#!/usr/bin/env python3
"""Extrai, por regex, as normas e artigos citados em cada questão.

Passo [2/3] de ./replicar.sh. Gera cache/leis_por_questao.csv com as colunas
id, leis_citadas, artigos_citados, n_leis, tem_lei, tema_lei_n1, disc_lei.

Entradas: questoes.jsonl, temas.csv, cache/lei_tema.csv e cache/normas_senado.json.
Saída: cache/leis_por_questao.csv (ou o caminho de --saida).

Sem rede, sem modelo de linguagem; determinístico. A versão original do CSV não
tinha gerador no repositório; este script é uma reconstrução por regex e NÃO
reproduz o CSV antigo linha a linha (ver README).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessor import TextCleaner  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
PREFIXO = {"lei": "LEI", "decreto-lei": "DEL", "decreto lei": "DEL"}

# Apelidos usuais (siglas e nomes) -> norma_id
ALIASES = {
    r"c[óo]digo penal\b(?!\s+militar)": "DEL-2848-1940",
    r"c[óo]digo de processo penal\b(?!\s+militar)": "DEL-3689-1941",
    r"lei maria da penha": "LEI-11340-2006",
    r"lei de drogas": "LEI-11343-2006",
    r"lei anticrime|pacote anticrime": "LEI-13964-2019",
    r"\bECA\b|estatuto da crian[çc]a": "LEI-8069-1990",
}
RE_NUM = re.compile(
    r"\b(lei|decreto[- ]lei)\s*(?:complementar\s*)?(?:federal\s*)?(?:n[ºo°.]*\s*)?(\d{1,2}(?:\.\d{3})|\d{3,5})\s*(?:/|,\s*de[^,]{0,25}de\s*|\s+de\s+(?:\d{1,2}\s+de\s+\w+\s+de\s+)?)?(\d{4}|\d{2})?",
    re.I)
RE_ART = re.compile(r"\bart(?:s|igos?)?\.?\s*(\d{1,4})(?:\s*-?\s*[A-Z]\b)?", re.I)


def carregar_mapa(lei_tema: pd.DataFrame, temas: pd.DataFrame):
    """Constrói os mapas norma -> temas, pai e nível.

    Args:
        lei_tema: DataFrame de cache/lei_tema.csv (norma_id, tema_id).
        temas: DataFrame de temas.csv.

    Returns:
        Tupla (por_norma, pai, nivel): conjunto de temas por norma, pai por tema
        e nível por tema.
    """
    pai = {int(r.tema_id): (None if pd.isna(r.parent_id) else int(r.parent_id)) for r in temas.itertuples()}
    nivel = {int(r.tema_id): int(r.nivel) for r in temas.itertuples()}
    por_norma = {}
    for r in lei_tema.itertuples():
        por_norma.setdefault(r.norma_id, set()).add(int(r.tema_id))
    return por_norma, pai, nivel


PRIORIDADE: set[str] = set()  # normas com tema mapeado em lei_tema.csv (têm precedência em duplicatas)


def normas_citadas(texto: str, conhecidas: set[str]) -> list[str]:
    """Identifica as normas citadas em um texto.

    Usa o número da lei/decreto-lei (tolera ano errado quando o número é
    conhecido) e apelidos usuais (Código Penal, Maria da Penha, ECA etc.).

    Args:
        texto: texto da questão.
        conhecidas: conjunto de norma_id conhecidos.

    Returns:
        Lista ordenada e sem repetição de norma_id.
    """
    achadas = []
    for m in RE_NUM.finditer(texto):
        tipo = PREFIXO[m.group(1).lower().replace("  ", " ")] if m.group(1).lower() in PREFIXO else "DEL"
        num = m.group(2).replace(".", "")
        ano = m.group(3)
        cands = sorted((k for k in conhecidas if k.startswith(f"{tipo}-{num}-")), key=lambda k: (k not in PRIORIDADE, k))
        if cands:  # número conhecido: tolera ano digitado errado (ex.: 11.340/2016)
            achadas.append(cands[0])
        elif ano and len(ano) == 4 and tipo == "LEI" and int(num) > 1000:
            achadas.append(f"{tipo}-{num}-{ano}")
    for padrao, norma in ALIASES.items():
        if re.search(padrao, texto, 0 if "ECA" in padrao else re.I) or re.search(padrao, texto):
            achadas.append(norma)
    return sorted(set(achadas))


def main():
    """Extrai as normas de todas as questões e grava o CSV.

    Opções de linha de comando: --saida (CSV de saída) e --comparar (CSV de
    referência para reportar diferenças).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(BASE_DIR / "cache" / "leis_por_questao.csv"))
    ap.add_argument("--comparar", default=None, help="CSV de referência para reportar diferenças")
    args = ap.parse_args()

    temas = pd.read_csv(BASE_DIR / "temas.csv")
    lei_tema = pd.read_csv(BASE_DIR / "cache" / "lei_tema.csv")
    por_norma, pai, nivel = carregar_mapa(lei_tema, temas)
    conhecidas = set(por_norma) | {k for k in json.loads((BASE_DIR / "cache" / "normas_senado.json").read_text(encoding="utf-8"))["normas"]}
    PRIORIDADE.update(por_norma)
    # temas de nível 1 = "especiais" (n1); raízes entram em disc_lei
    linhas = []
    for line in (BASE_DIR / "questoes.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = TextCleaner.extract_question_text(r.get("enunciado_html"), r.get("comando"), r.get("alternativas"), include_alternativas=True)
        normas = normas_citadas(t, conhecidas)
        arts = []
        if normas:
            for m in RE_ART.finditer(t):
                a = m.group(1)
                if a not in arts:
                    arts.append(a)
        n1, disc = [], []
        for n in normas:
            for tid in sorted(por_norma.get(n, [])):
                if nivel.get(tid) == 1:
                    n1.append(str(tid))
                    if pai.get(tid) is not None:
                        disc.append(str(pai[tid]))
                else:
                    disc.append(str(tid))
        dedup = lambda xs: [x for i, x in enumerate(xs) if x not in xs[:i]]
        linhas.append(dict(
            id=r["id"], leis_citadas=";".join(normas),
            artigos_citados=";".join(sorted(arts, key=int)[:10]) if arts else "",
            n_leis=len(normas), tem_lei=bool(normas),
            tema_lei_n1=";".join(dedup(n1)), disc_lei=";".join(sorted(dedup(disc))),
        ))
    out = pd.DataFrame(linhas)
    out.to_csv(args.saida, index=False)
    print(f"extrair_leis: {len(out)} questões, {int(out.tem_lei.sum())} com norma citada -> {args.saida}")
    if args.comparar:
        ref = pd.read_csv(args.comparar).set_index("id").fillna("")
        new = out.set_index("id").fillna("")
        for c in ["leis_citadas", "tema_lei_n1", "disc_lei", "artigos_citados", "tem_lei"]:
            print(f"  diff {c}: {(ref[c].astype(str) != new.loc[ref.index, c].astype(str)).sum()} linhas")
        print(f"  linhas com qualquer diferença: {(ref.astype(str).ne(new.loc[ref.index].astype(str))).any(axis=1).sum()} de {len(ref)}")


if __name__ == "__main__":
    main()
