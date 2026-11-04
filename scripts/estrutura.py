"""Detecção de nós de nível 1 duplicados (regra genérica, sem ids fixos).

Parte do pipeline de ./replicar.sh (importado por dados.py). Não lê nem grava
arquivos por conta própria.

Duplicado = nó de nível 1 ativo SEM filhos ativos e com descrição vazia/NaN
(não gerado a partir da fonte original) cujo título normalizado (minúsculas,
sem acento, sem "(parte N)"/sufixo após hífen, sem artigos/preposições, plural
removido) tem Jaccard de tokens >= LIMIAR_JACCARD com um IRMÃO ativo que TEM
filhos ativos. Alvo = irmão de maior Jaccard. Também entram (união) os nós
"(Parte N)" sem filhos que repetem o título de um irmão (mascara_estrutural). Um duplicado
nunca é previsto; seus exemplos/descrição sintéticos são reatribuídos ao alvo.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import pandas as pd

LIMIAR_JACCARD = 0.6  # fixado a priori
STOP = {"de", "do", "da", "dos", "das", "em", "e", "a", "o", "as", "os", "no", "na", "nos", "nas", "ao", "aos"}


def toks(titulo: str):
    """Tokeniza um título de tema para comparação por Jaccard.

    Args:
        titulo: título do tema.

    Returns:
        Conjunto de tokens normalizados (sem acento, sem stopwords, sem plural simples).
    """
    s = unicodedata.normalize("NFKD", str(titulo))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\(\s*parte\s*\d+\s*\)", " ", s)
    s = re.sub(r"\s-\s.*$", lambda m: " " + m.group(0)[2:], s)  # "x - especies" mantem as palavras
    out = set()
    for t in re.sub(r"[^a-z0-9]+", " ", s).split():
        if t in STOP:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.add(t)
    return out


def jaccard(a, b) -> float:
    """Calcula a similaridade de Jaccard entre dois conjuntos.

    Args:
        a: primeiro conjunto de tokens.
        b: segundo conjunto de tokens.

    Returns:
        |a inter b| / |a uniao b|; 0.0 se ambos forem vazios.
    """
    return len(a & b) / len(a | b) if a | b else 0.0


def duplicados(tax) -> Dict[int, int]:
    """Identifica os nós de nível 1 duplicados pela regra de Jaccard.

    Args:
        tax: TaxonomyManager com a árvore de temas.

    Returns:
        Dicionário {id_duplicado: id_alvo}.
    """
    dup: Dict[int, int] = {}
    for tid in tax.act_l1:
        ativos = [c for c in tax.children[tid] if tax.status.get(c) == "active"]
        if ativos:
            continue
        desc = tax.rows[tid].descricao
        vazia = pd.isna(desc) or not str(desc).strip() or str(desc).strip().lower() == "nan"
        if not vazia:
            continue
        irm = [s for s in tax.act_l1 if s != tid and tax.pai[s] == tax.pai[tid]]
        com_filhos = [s for s in irm if any(tax.status.get(c) == "active" for c in tax.children[s])]
        if not com_filhos:
            continue
        tt = toks(tax.titulo[tid])
        best = max(com_filhos, key=lambda s: (jaccard(tt, toks(tax.titulo[s])), -s))
        if jaccard(tt, toks(tax.titulo[best])) >= LIMIAR_JACCARD:
            dup[tid] = best
    return dup


def norm_titulo(s: str) -> str:
    """Normaliza um título (sem acento, minúsculo, só alfanumérico, espaços únicos).

    Args:
        s: título.

    Returns:
        Título normalizado.
    """
    sem = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", sem.lower())).strip()


def mascara_estrutural(tax) -> List[int]:
    """Lista os nós de nível 1 sem filhos cujo título é "<título de irmão> (Parte N)".

    Args:
        tax: TaxonomyManager com a árvore de temas.

    Returns:
        Ids dos nós a excluir (têm um irmão de mesmo título base).
    """
    excl = []
    rx = re.compile(r"^(.*?)\s*\(\s*parte\s*\d+\s*\)\s*(?:[-:–].*)?$", re.I)
    for tid in tax.act_l1:
        if any(tax.status.get(c) == "active" for c in tax.children[tid]):
            continue
        m = rx.match(tax.titulo[tid])
        if not m:
            continue
        base = norm_titulo(m.group(1))
        irmaos = [s for s in tax.children.get(tax.pai[tid], []) if s != tid and norm_titulo(tax.titulo[s]) == base]
        if irmaos:
            excl.append(tid)
    return excl


def nos_duplicados(tax) -> Dict[int, int]:
    """Une os duplicados por Jaccard com os nós "(Parte N)" (mascara_estrutural).

    Args:
        tax: TaxonomyManager com a árvore de temas.

    Returns:
        Dicionário {id_duplicado: id_alvo}, alvo = irmão de maior Jaccard.
    """
    excl = mascara_estrutural(tax)
    dup = duplicados(tax)
    for tid in excl:
        if tid in dup:
            continue
        irm = [s for s in tax.act_l1 if s != tid and tax.pai[s] == tax.pai[tid] and s not in excl]
        if irm:
            tt = toks(tax.titulo[tid])
            dup[tid] = max(irm, key=lambda s: (jaccard(tt, toks(tax.titulo[s])), -s))
    return dup
