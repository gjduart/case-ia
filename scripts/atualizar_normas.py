#!/usr/bin/env python3
"""Etapa de PREPARAÇÃO (rede permitida; NÃO é chamada por replicar.sh).

Atualiza cache/normas_senado.json (lido por extrair_leis.py e pelo HybridClassifier), sem
LLM e sem mapeamento manual: para cada norma de cache/lei_tema.csv, de
cache/leis_por_questao.csv (coluna leis_citadas) e da lista histórica NORMAS, consulta a API
de Dados Abertos do Senado (GET /dadosabertos/legislacao/{tipo}/{num}/{ano}) e grava os
metadados no formato (normaNome, apelido, ementa, urn, ok) mais indexacao, catalogo,
classes, edivs e consultado_em. Normas não encontradas ficam com ok=false e
nao_encontrada=true.

O mapa norma -> tema (cache/lei_tema.csv) é uma entrada curada, não gerada por este script.

Uso:
    python -B scripts/atualizar_normas.py            # consulta o Senado (rede)
    python -B scripts/atualizar_normas.py --forcar   # refaz todas as consultas
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE = BASE_DIR / "cache"

URL_DETALHE = "https://legis.senado.leg.br/dadosabertos/legislacao/{tipo}/{num}/{ano}"
UA = {"User-Agent": "Mozilla/5.0 (case-ia-classificacao/1.1; pesquisa academica)"}
PAUSA = 0.3

# Normas históricas do corpus (lista original do script); a união com os CSVs é feita em runtime.
NORMAS = [
    ("del", "2848", "1940"), ("del", "3688", "1941"), ("del", "3689", "1941"),
    ("lei", "1521", "1951"), ("lei", "2889", "1956"), ("lei", "4737", "1965"),
    ("lei", "6368", "1976"), ("lei", "7210", "1984"), ("lei", "7492", "1986"),
    ("lei", "7716", "1989"), ("lei", "8072", "1990"), ("lei", "8078", "1990"),
    ("lei", "8137", "1990"), ("lei", "8069", "1990"), ("lei", "8666", "1993"),
    ("lei", "9099", "1995"), ("lei", "9296", "1996"), ("lei", "9455", "1997"),
    ("lei", "9503", "1997"), ("lei", "9605", "1998"), ("lei", "9613", "1998"),
    ("lei", "10741", "2003"), ("lei", "10826", "2003"), ("lei", "11105", "2005"),
    ("lei", "11340", "2006"), ("lei", "11343", "2006"), ("lei", "12037", "2009"),
    ("lei", "12683", "2012"), ("lei", "12737", "2012"), ("lei", "12850", "2013"),
    ("lei", "13146", "2015"), ("lei", "13260", "2016"), ("lei", "13869", "2019"),
    ("lei", "13964", "2019"), ("lei", "14133", "2021"), ("lei", "14344", "2022"),
]


# ---------------------------------------------------------------------------
# utilidades de rede
# ---------------------------------------------------------------------------
def http_get(url: str, aceita: str = "*/*", timeout: int = 60, tentativas: int = 4) -> bytes:
    """Faz GET com timeout e retentativas (backoff linear).

    Args:
        url: endereço a consultar.
        aceita: valor do cabeçalho Accept.
        timeout: tempo limite por tentativa, em segundos.
        tentativas: número máximo de tentativas.

    Returns:
        Corpo da resposta, em bytes.

    Raises:
        urllib.error.HTTPError: imediatamente para HTTP 400/404.
        Exception: a última exceção, se todas as tentativas falharem.
    """
    ultimo: Exception | None = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": aceita})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):  # definitivo: não adianta repetir
                raise
            ultimo = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
            ultimo = exc
        time.sleep(1.5 * (i + 1))
    assert ultimo is not None
    raise ultimo


def partir_id(norma_id: str) -> tuple[str, str, str] | None:
    """Decompõe um norma_id (ex.: 'LEI-11340-2006').

    Args:
        norma_id: identificador da norma.

    Returns:
        Tupla (tipo em minúsculas, número, ano) ou None se o formato for inválido.
    """
    m = re.fullmatch(r"([A-Z]+)-(\d+)-(\d{4})", norma_id)
    if not m:
        return None
    return m.group(1).lower(), m.group(2), m.group(3)


def chave(tipo: str, num: str, ano: str) -> str:
    """Monta o norma_id a partir das partes.

    Args:
        tipo: tipo da norma ('lei', 'del').
        num: número da norma.
        ano: ano da norma.

    Returns:
        Identificador como 'LEI-11340-2006'.
    """
    return f"{tipo.upper()}-{num}-{ano}"


def universo_normas() -> list[str]:
    """Reúne todas as normas a consultar.

    União da lista histórica NORMAS, de cache/lei_tema.csv e das leis citadas em
    cache/leis_por_questao.csv.

    Returns:
        Lista ordenada de norma_id válidos.
    """
    ids = {chave(*t) for t in NORMAS}
    p = CACHE / "lei_tema.csv"
    if p.exists():
        import csv
        with p.open(encoding="utf-8") as f:
            ids |= {r["norma_id"] for r in csv.DictReader(f)}
    p = CACHE / "leis_por_questao.csv"
    if p.exists():
        import csv
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ids |= {x for x in (r.get("leis_citadas") or "").split(";") if x}
    return sorted(i for i in ids if partir_id(i))


# ---------------------------------------------------------------------------
# 1) metadados do Senado
# ---------------------------------------------------------------------------
def _frases_para_termos(frases) -> list[str]:
    """Converte frases de indexação em termos únicos.

    Args:
        frases: lista de frases separadas por vírgula (pode ser None).

    Returns:
        Lista de termos sem repetição, com espaços normalizados.
    """
    termos: list[str] = []
    for fr in frases or []:
        for t in str(fr).split(","):
            t = re.sub(r"\s+", " ", t).strip(" .")
            if t and t not in termos:
                termos.append(t)
    return termos


def _lista(x):
    """Garante que o valor seja uma lista.

    Args:
        x: valor, lista, None ou "".

    Returns:
        x se já for lista; [] se vazio; senão [x].
    """
    return x if isinstance(x, list) else ([] if x in (None, "") else [x])


def buscar_detalhe(tipo: str, num: str, ano: str) -> dict:
    """Consulta o detalhe de uma norma na API de Dados Abertos do Senado.

    Args:
        tipo: tipo da norma ('lei', 'del').
        num: número da norma.
        ano: ano da norma.

    Returns:
        Dicionário com ok, normaNome, apelido, ementa, urn, indexacao, catalogo,
        classes, edivs e consultado_em.
    """
    url = URL_DETALHE.format(tipo=tipo, num=num, ano=ano)
    bruto = http_get(url, aceita="application/json", timeout=30, tentativas=3)
    doc = json.loads(bruto)["DetalheDocumento"]["documentos"]["documento"]
    doc = doc[0] if isinstance(doc, list) else doc
    ident = doc.get("identificacao", {})
    edivs = []
    for e in _lista((doc.get("edivs") or {}).get("ediv")):
        itens = [i.get("dispositivo", "").strip() for i in _lista((e.get("itens") or {}).get("item"))]
        edivs.append({"norma": e.get("nomeNormaAnterior", ""), "cod": e.get("codnormaAnterior", ""),
                      "comentario": e.get("comentario", ""), "dispositivos": [i for i in itens if i]})
    return {
        "ok": True,
        "normaNome": ident.get("normaNome", ""),
        "apelido": ident.get("apelido", ""),
        "ementa": doc.get("ementa", ""),
        "urn": ident.get("urlDocumento", ""),
        "indexacao": _frases_para_termos((doc.get("indexacao") or {}).get("frase")),
        "catalogo": _frases_para_termos((doc.get("catalogo") or {}).get("frase")),
        "classes": [c for c in _lista((doc.get("classes") or {}).get("classe"))],
        "edivs": edivs,
        "consultado_em": time.strftime("%Y-%m-%d"),
    }


def atualizar_metadados(forcar: bool) -> dict:
    """Atualiza cache/normas_senado.json com os metadados do Senado.

    Args:
        forcar: se True, refaz também as normas já consultadas.

    Returns:
        O dicionário completo gravado no arquivo.
    """
    saida = CACHE / "normas_senado.json"
    if saida.exists():
        dados = json.loads(saida.read_text(encoding="utf-8"))
    else:
        dados = {"fonte": "https://legis.senado.leg.br/dadosabertos (API de Dados Abertos do Senado Federal)",
                 "endpoint": "/legislacao/{tipo}/{numero}/{ano}", "normas": {}}
    dados["observacao"] = ("Cache versionado. O pipeline de publicacao NAO faz chamada de rede nem de LLM; "
                           "le este arquivo. Campos novos: indexacao, catalogo, classes, edivs, consultado_em.")
    normas = dados.setdefault("normas", {})
    nao_achadas = []
    for k in universo_normas():
        atual = normas.get(k, {})
        if not forcar and atual.get("ok") and "indexacao" in atual:
            continue
        if not forcar and atual.get("nao_encontrada"):
            nao_achadas.append(k)
            continue
        tipo, num, ano = partir_id(k)  # type: ignore[misc]
        try:
            novo = buscar_detalhe(tipo, num, ano)
            normas[k] = novo
            print(f"OK  {k:16s} {novo['apelido'][:60]} | idx={len(novo['indexacao'])} edivs={len(novo['edivs'])}")
        except urllib.error.HTTPError as exc:
            normas[k] = {"ok": False, "nao_encontrada": exc.code in (400, 404),
                         "erro": f"HTTP {exc.code}", "consultado_em": time.strftime("%Y-%m-%d")}
            nao_achadas.append(k)
            print(f"ERR {k:16s} HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001  (rede/JSON: registrar e seguir)
            normas[k] = {"ok": False, "nao_encontrada": isinstance(exc, (KeyError, IndexError, ValueError)),
                         "erro": f"{type(exc).__name__}: {exc}"[:160], "consultado_em": time.strftime("%Y-%m-%d")}
            nao_achadas.append(k)
            print(f"ERR {k:16s} {type(exc).__name__}: {exc}")
        time.sleep(PAUSA)
    dados["consultado_em"] = time.strftime("%Y-%m-%d")
    dados["nao_encontradas"] = sorted(set(nao_achadas))
    saida.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in normas.values() if v.get("ok"))
    print(f"\nmetadados: {len(normas)} normas | {ok} ok | nao encontradas: {sorted(set(nao_achadas))}")
    return dados


def main() -> int:
    """Ponto de entrada: atualiza cache/normas_senado.json.

    Opção: --forcar (refaz as consultas).

    Returns:
        Código de saída 0.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcar", action="store_true", help="refaz todas as consultas")
    args = ap.parse_args()
    atualizar_metadados(args.forcar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
