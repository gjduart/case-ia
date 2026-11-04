"""Protocolo de validação OOF 5x5 e regra de publicação da entrega (A1).

Parte do pipeline de ./replicar.sh (importado por gerar_publicacao.py). Não lê nem grava
arquivos: trabalha sobre o dados.Contexto.

Protocolo: StratifiedKFold(5, shuffle, random_state = 100 + repetição) sobre as 100
linhas da folha, 5 repetições, id-disjunto; tudo (roteador, NB, qwen-LR, bagging, pisos)
é aprendido só no treino do fold. A linha "prod" treina com a folha inteira e decide as
659 questões (as 559 fora da folha são as publicadas na entrega).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import decisao
import roteador_area as ra
from dados import Contexto

N_REPS, N_FOLDS = 5, 5


def decidir_fold(ctx: Contexto, chave, trn_ids, tst_ids, seed: int):
    """Decide um fold (S4 + áreas novas) limitando as threads, para uso em paralelo.

    Args:
        ctx: contexto de dados.
        chave: (repetição, fold) ou "prod".
        trn_ids: ids de treino.
        tst_ids: ids a decidir.
        seed: semente do fold.

    Returns:
        Tupla (chave, DataFrame combinado, modelos do roteador ou None).
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        s4 = decisao.decidir_fold(ctx, trn_ids, tst_ids, seed)
        n, mod = ra.decidir_fold(ctx, trn_ids, tst_ids, seed, guardar=(chave == "prod"))
        return chave, ra.combinar(ctx, s4, n), mod


def particoes(ctx: Contexto):
    """Lista os jobs de validação: 25 folds externos e a produção.

    Args:
        ctx: contexto de dados.

    Returns:
        Lista de tuplas (chave, ids de treino, ids de teste, semente).
    """
    jobs = []
    for rep in range(N_REPS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 + rep)
        for k, (a, b) in enumerate(skf.split(ctx.sup, ctx.sup.estr)):
            jobs.append(((rep, k), list(ctx.sup.id.iloc[a]), list(ctx.sup.id.iloc[b]), 100 + rep * 10 + k))
    jobs.append(("prod", list(ctx.sup.id), list(ctx.q.id), 999))
    return jobs


def executar(ctx: Contexto, n_jobs: int = 8):
    """Roda os 25 folds e a produção, em paralelo (resultado idêntico ao serial).

    Args:
        ctx: contexto de dados.
        n_jobs: número de processos.

    Returns:
        Tupla ({(rep, fold) ou "prod": DataFrame}, modelos do roteador da produção).
    """
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs)(delayed(decidir_fold)(ctx, *j) for j in particoes(ctx))
    return {k: d for k, d, _ in out}, next(m for k, _, m in out if k == "prod")


def publicar(d: pd.DataFrame) -> np.ndarray:
    """Calcula a máscara de publicação: consenso, bagging estável e acima do piso.

    Args:
        d: DataFrame de um fold com r0, cnt, score e qg.

    Returns:
        Array booleano de publicação.
    """
    return d.r0.values & (d.cnt.values >= decisao.sacos_necessarios()) & (d.score.values >= d.qg.values)


def avaliar(ex: dict) -> dict:
    """Avalia a regra por OOF 5x5 e pela produção.

    Args:
        ex: dicionário de folds de executar.

    Returns:
        Dicionário com oof (DataFrame com pub, rep, fold), por_rep (acertos, publicadas,
        total por repetição), prod (DataFrame com pub) e res (K, N, T, prec, n_eff, cob,
        pior; K e N somados nas repetições).
    """
    folds = []
    for rep in range(N_REPS):
        for k in range(N_FOLDS):
            d = ex[(rep, k)].copy()
            d["pub"] = publicar(ex[(rep, k)])
            d["rep"], d["fold"] = rep, k
            folds.append(d)
    oof = pd.concat(folds, ignore_index=True)
    por_rep = [(int(oof.ramo[(oof.rep == r) & oof.pub].sum()), int(((oof.rep == r) & oof.pub).sum()), int((oof.rep == r).sum())) for r in range(N_REPS)]
    prod = ex["prod"].copy()
    prod["pub"] = publicar(ex["prod"])
    return dict(oof=oof, por_rep=por_rep, prod=prod, res=resumo(por_rep))


def resumo(por_rep) -> dict:
    """Resume as contagens por repetição.

    Args:
        por_rep: lista de tuplas (acertos, publicadas, total) por repetição.

    Returns:
        Dicionário com K, N, T (somas), prec, n_eff (publicadas por repetição, i.e.
        por questão distinta), cob (cobertura OOF) e pior (pior precisão entre as repetições).
    """
    K = sum(r[0] for r in por_rep)
    N = sum(r[1] for r in por_rep)
    T = sum(r[2] for r in por_rep)
    return dict(K=K, N=N, T=T, prec=K / N if N else 0.0, n_eff=N / N_REPS, cob=N / T,
                pior=min((r[0] / r[1]) if r[1] else 1.0 for r in por_rep))


def cobertura_559(ctx: Contexto, prod: pd.DataFrame) -> dict:
    """Conta as publicações nas 559 questões fora da folha.

    Args:
        ctx: contexto de dados.
        prod: DataFrame de produção com a coluna pub.

    Returns:
        Dicionário com p559/n559 (total), pb/nb (banco Q0001-Q0359) e pn/nn (novas Q0360+).
    """
    fo = ~prod.id.isin(set(ctx.sup.id))
    nv = np.array([bool(ctx.novo[ctx.qidx[i]]) for i in prod.id])
    return dict(p559=int((prod.pub & fo).sum()), n559=int(fo.sum()), pb=int((prod.pub & fo & ~nv).sum()), nb=int((fo & ~nv).sum()),
                pn=int((prod.pub & fo & nv).sum()), nn=int((fo & nv).sum()))


def por_area(ctx: Contexto, av: dict) -> dict:
    """Desempenho por raiz (área): OOF e produção nas 559.

    Args:
        ctx: contexto de dados.
        av: resultado de avaliar.

    Returns:
        Dicionário {raiz: {oof_h, oof_p, oof_t, prod_p, prod_t}} (acertos, publicadas e
        total no OOF; publicadas e total nas 559).
    """
    oof, prod = av["oof"], av["prod"]
    fo = ~prod.id.isin(set(ctx.sup.id))
    out = {}
    for r in ctx.tax.roots:
        mo, mp = oof.root == r, fo & (prod.root == r)
        out[r] = dict(oof_h=int(oof.ramo[oof.pub & mo].sum()), oof_p=int((oof.pub & mo).sum()), oof_t=int(mo.sum()),
                      prod_p=int((prod.pub & mp).sum()), prod_t=int(mp.sum()))
    return out
