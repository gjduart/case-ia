#!/usr/bin/env python3
"""Etapa de PREPARAÇÃO (não roda no replicar.sh): filtro automático dos exemplos sintéticos.

Gera cache/sinteticos_filtrados.jsonl (lido pela entrega), cache/sinteticos_filtro_relatorio.json
e, se ainda não existir, cache/sinteticos_amostra_revisao.csv (30 exemplos para revisão humana,
estratificados por raiz; uma planilha já revisada nunca é sobrescrita).

Entradas: cache/sinteticos/ (exemplos gerados), cache/embeddings/e5/ (vetores e5 de sint,
sint_desc e pseudo; ver scripts/embeddings.py --modelo e5) e temas.csv.
Método (round-trip, fixado antes de qualquer classificador de decisão): uma LR sobre os
embeddings e5, treinada nos OUTROS exemplos (6-fold estratificado por tema) + descrições +
pseudo-documentos, tem de colocar o tema do próprio exemplo no top-2; depois removem-se as
quase-duplicatas (cosseno >= 0,98). Os nós duplicados da taxonomia são mesclados no alvo.
Uso: python -B scripts/filtrar_sinteticos.py [--saida DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dados import BASE_DIR, C_LR, Contexto  # noqa: E402

COS_DUP = 0.98  # quase-duplicatas (0,95 removeria 346 de ~885: no e5 o cosseno basal entre textos jurídicos é ~0,91)
SEED = 0


def registros_sinteticos(ctx: Contexto) -> list:
    """Lê o registro bruto (enunciado, comando, alternativas, estilo) de cada exemplo sintético.

    Args:
        ctx: contexto de dados com sint_ids.

    Returns:
        Lista de registros na ordem de ctx.sint_ids.
    """
    out = []
    for s in ctx.sint_ids:
        t, i = s.split("#")
        L = [json.loads(l) for l in (BASE_DIR / "cache" / "sinteticos" / f"tema_{t}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        out.append([r for r in L if r.get("tipo") == "exemplo"][int(i)])
    return out


def filtrar(ctx: Contexto, saida: Path) -> dict:
    """Filtra os sintéticos e grava os artefatos de revisão.

    Args:
        ctx: contexto com os vetores e5 (aplicar_filtro=False).
        saida: diretório de saída (cache/ na entrega).

    Returns:
        Dicionário-relatório com n, top1, top2, removidos, quase-duplicatas e mantidos.
    """
    rec = registros_sinteticos(ctx)
    y = np.array([ctx.cls_idx[t] for t in ctx.sint_tema])
    n = len(y)
    top1 = np.zeros(n, bool)
    top2 = np.zeros(n, bool)
    Xe = np.vstack([ctx.E_desc, ctx.E_pseudo[ctx.pseudo_rows]])
    ye = np.concatenate([[ctx.cls_idx[t] for t in ctx.desc_tema], ctx.pseudo_rows])
    for a, b in StratifiedKFold(6, shuffle=True, random_state=SEED).split(ctx.E_sint, y):
        m = LogisticRegression(C=C_LR, max_iter=500).fit(np.vstack([ctx.E_sint[a], Xe]), np.concatenate([y[a], ye]))
        P = m.predict_proba(ctx.E_sint[b])
        cl = m.classes_[np.argsort(-P, axis=1)[:, :2]]
        top1[b] = cl[:, 0] == y[b]
        top2[b] = (cl == y[b][:, None]).any(axis=1)
    keep = top2.copy()
    nd = 0
    S = ctx.E_sint @ ctx.E_sint.T
    kept_idx = []
    for j in range(n):
        if not keep[j]:
            continue
        if any(S[j, k] >= COS_DUP for k in kept_idx):
            keep[j] = False
            nd += 1
        else:
            kept_idx.append(j)
    rep = dict(n=n, top1=float(top1.mean()), top2=float(top2.mean()), removidos_top2=int((~top2).sum()), quase_dup=nd, mantidos=int(keep.sum()), pct=float(keep.mean()))
    (saida / "sinteticos_filtro_relatorio.json").write_text(json.dumps(dict(rep, cos_dup=COS_DUP), indent=1), encoding="utf-8")
    print(f"Filtro sintetico: {n} exemplos; top-1 correto {rep['top1']:.1%}; tema no top-2 {rep['top2']:.1%} ({rep['removidos_top2']} removidos); "
          f"quase-duplicatas cos>={COS_DUP}: {nd} removidas; mantidos {rep['mantidos']}/{n} = {rep['pct']:.1%}")
    rows = []
    for j in range(n):
        if not keep[j]:
            continue
        r = rec[j]
        rows.append(dict(emb_id=ctx.sint_ids[j], tema_id=int(ctx.sint_ids[j].split("#")[0]), tema_id_efetivo=int(ctx.sint_tema[j]), estilo=r.get("estilo"),
                         enunciado=r.get("enunciado"), comando=r.get("comando"), alternativas=r.get("alternativas"), top1_roundtrip=bool(top1[j])))
    (saida / "sinteticos_filtrados.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    if (saida / "sinteticos_amostra_revisao.csv").exists():
        print("sinteticos_amostra_revisao.csv ja existe (revisao humana preenchida): mantido sem alteracao")
    else:
        amostra_revisao(ctx, pd.DataFrame(rows), saida)
    return rep


def amostra_revisao(ctx: Contexto, kept: pd.DataFrame, saida: Path) -> None:
    """Grava a amostra de 30 mantidos (proporcional por raiz, >= 1 por raiz) para revisão humana.

    Args:
        ctx: contexto de dados.
        kept: DataFrame dos sintéticos mantidos.
        saida: diretório de saída.
    """
    kept["raiz"] = kept.tema_id_efetivo.map(lambda t: ctx.tax.titulo[ctx.tax.pai[int(t)]])
    rng = np.random.RandomState(SEED)
    raizes = sorted(kept.raiz.unique())
    cota = {r: max(1, int(round(30 * (kept.raiz == r).mean()))) for r in raizes}
    while sum(cota.values()) > 30:
        cota[max(cota, key=cota.get)] -= 1
    while sum(cota.values()) < 30:
        cota[max(cota, key=lambda r: (kept.raiz == r).sum() - cota[r])] += 1
    pick = []
    for r in raizes:
        idx = np.where(kept.raiz.values == r)[0]
        pick += list(rng.choice(idx, size=min(cota[r], len(idx)), replace=False))
    am = kept.iloc[sorted(pick)]
    def caminho(rw):
        """Formata o caminho do tema de um exemplo, indicando a mesclagem.

        Args:
            rw: linha (namedtuple) com tema_id e tema_id_efetivo.

        Returns:
            Caminho do tema, com o destino da mesclagem quando houver.
        """
        c = ctx.tax.caminho[int(rw.tema_id)]
        return c if rw.tema_id == rw.tema_id_efetivo else f"{c} (mesclado em: {ctx.tax.caminho[int(rw.tema_id_efetivo)]})"
    pd.DataFrame(dict(tema_caminho=[caminho(r) for r in am.itertuples()], estilo=am.estilo.values, enunciado=am.enunciado.values,
                      alternativas=[" | ".join(f"{k}) {v}" for k, v in (a or {}).items()) for a in am.alternativas],
                      ok_humano="", comentario="")).to_csv(saida / "sinteticos_amostra_revisao.csv", index=False, encoding="utf-8")


def main():
    """Ponto de entrada: regenera o filtro dos sintéticos (opção --saida)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(BASE_DIR / "cache"))
    a = ap.parse_args()
    filtrar(Contexto("e5", aplicar_filtro=False), Path(a.saida))


if __name__ == "__main__":
    main()
