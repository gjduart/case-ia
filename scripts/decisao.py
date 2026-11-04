"""Regra S4 de Direito Penal e Processo Penal: consenso NB = BM25 = LSA + bagging + piso de score.

Parte do pipeline de ./replicar.sh (importado por validacao.py e gerar_publicacao.py).
Lê apenas objetos em memória (dados.Contexto); não grava arquivos.

Para cada questão a decisão é: tema = argmax do blend 0,55 NB + 0,30 BM25 + 0,15 LSA
(forçado pela norma unívoca citada, quando existe); a questão só é publicada se NB, BM25 e
LSA concordam no tema (consenso), se o consenso se repete em >= 80% de 25 subamostras de 80%
do treino (bagging do NB; 20 sacos) e se o score do consenso (logit do blend com bônus de
2,8 para norma concordante) passa o piso do fold (quartil inferior do score do consenso,
calculado por CV interna só nas linhas de treino). Tudo é aprendido apenas no treino do fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dados import Contexto, renorm

B_BAGS, FRAC_BAG, TAU = 25, 0.8, 0.8
SEED_BAG = 4000
PESOS = (0.55, 0.30, 0.15)  # NB, BM25, LSA
BONUS_NORMA = 2.8


def sacos_necessarios(tau: float = TAU) -> int:
    """Converte tau no número mínimo de sacos de bagging concordantes.

    Args:
        tau: fração mínima de sacos (0 a 1).

    Returns:
        ceil(tau * B_BAGS), com tolerância numérica (20 para tau = 0,8).
    """
    return int(np.ceil(tau * B_BAGS - 1e-9))


def tem_norma(ctx: Contexto, ti: np.ndarray) -> np.ndarray:
    """Indica as questões cuja norma citada aponta um único tema não duplicado.

    Args:
        ctx: contexto de dados.
        ti: índices das questões em ctx.q.

    Returns:
        Array booleano.
    """
    return np.array([ctx.law_cls[i] >= 0 and ctx.mask[ctx.law_cls[i]] > 0 for i in ti])


def nucleo(ctx: Contexto, ids, P_nb: np.ndarray, ti: np.ndarray):
    """Calcula o tema, o consenso NB = BM25 = LSA e o score do consenso.

    Args:
        ctx: contexto de dados.
        ids: ids das questões.
        P_nb: probabilidades do NB hierárquico.
        ti: índices das questões em ctx.q.

    Returns:
        Tupla (coluna do tema, indicador de consenso, score logit com bônus de norma).
    """
    P = PESOS[0] * P_nb + PESOS[1] * ctx.S_bm[ti] + PESOS[2] * ctx.S_ls[ti]
    b = ctx.lei_decide(ids, P)
    ar = np.arange(len(b))
    cons = ((P_nb.sum(axis=1) > 0) & (P_nb.argmax(1) == b) & (ctx.S_bm[ti].argmax(1) == b) & (ctx.S_ls[ti].argmax(1) == b))
    p = np.clip(P[ar, b], 1e-6, 1 - 1e-6)
    return b, cons, np.log(p / (1 - p)) + BONUS_NORMA * tem_norma(ctx, ti)


def decidir_fold(ctx: Contexto, trn_ids, tst_ids, seed: int) -> pd.DataFrame:
    """Aplica a regra S4 a um fold: treina no treino e decide o teste.

    Args:
        ctx: contexto de dados.
        trn_ids: ids da folha usados no treino (as linhas "fora" só entram no piso).
        tst_ids: ids a decidir.
        seed: semente da CV interna do piso e do bagging.

    Returns:
        DataFrame por questão com id, pred (tema), r0 (consenso), cnt (sacos com
        consenso), score, qg (piso), law (norma concordante), ag_nb0/ag_bm/ag_ls
        (concordância de cada classificador com o tema) e root.
    """
    clf, tax = ctx.clf, ctx.tax
    lab = ctx.sup[ctx.sup.id.isin(trn_ids) & (ctx.sup.y != -1)]
    ti = np.array([ctx.qidx[i] for i in tst_ids])
    tst_df = ctx.q.iloc[ti]
    nb = clf.fit_predict_hierarchy(lab, tst_df)
    # blend e consenso com BM25/LSA renormalizados (mesma conta da avaliação original)
    bm, ls = renorm(ctx.S_bm[ti] * np.ones(ctx.K)), renorm(ctx.S_ls[ti] * np.ones(ctx.K))
    P = PESOS[0] * nb + PESOS[1] * bm + PESOS[2] * ls
    b = ctx.lei_decide(list(tst_ids), P)
    ag = {"ag_nb0": (nb.sum(axis=1) > 0) & (nb.argmax(axis=1) == b),
          "ag_bm": (bm.sum(axis=1) > 0) & (bm.argmax(axis=1) == b),
          "ag_ls": (ls.sum(axis=1) > 0) & (ls.argmax(axis=1) == b)}
    pred = [tax.act_l1[x] for x in b]
    # bagging: o NB é retreinado em B_BAGS subamostras de 80% do treino
    rng = np.random.RandomState(SEED_BAG + seed)
    cnt = np.zeros(len(b), int)
    for _ in range(B_BAGS):
        idx = rng.choice(len(lab), int(round(FRAC_BAG * len(lab))), replace=False)
        Pb = clf.fit_predict_hierarchy(lab.iloc[idx], tst_df)
        cnt += ((Pb.sum(axis=1) > 0) & (Pb.argmax(1) == b))
    # score do consenso e piso (quartil inferior) por CV interna das linhas de treino
    _, _, score = nucleo(ctx, list(tst_ids), nb, ti)
    sup_tr = ctx.sup[ctx.sup.id.isin(trn_ids)].reset_index(drop=True)
    sc_in = []
    for a, bb in StratifiedKFold(5, shuffle=True, random_state=seed).split(sup_tr, sup_tr.estr):
        li = sup_tr.iloc[a]
        li = li[li.y != -1]
        te_ids = list(sup_tr.id.iloc[bb])
        tj = np.array([ctx.qidx[i] for i in te_ids])
        Pn = clf.fit_predict_hierarchy(li, ctx.q.iloc[tj])
        _, ci, si = nucleo(ctx, te_ids, Pn, tj)
        sc_in += list(si[ci])
    qg = float(np.quantile(np.array(sc_in), 0.25))
    return pd.DataFrame(dict(id=list(tst_ids), pred=pred, r0=ag["ag_nb0"] & ag["ag_bm"] & ag["ag_ls"], cnt=cnt, score=score, qg=qg,
                             law=tem_norma(ctx, ti), root=[tax.pai[int(p)] for p in pred], **ag))
