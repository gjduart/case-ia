"""Calibração da coluna `confianca` (P(ramo do tema sugerido correto)). Sem LLM, sem rede.

Parte do pipeline de ./replicar.sh (importado por gerar_publicacao.py e
pela API). Não lê nem grava arquivos por conta própria.

Escolha a priori: regressão logística (C=1) sobre [publicada, consenso
NB=BM25=LSA, fração de sacos do bagging com consenso, score do blend sem o bônus
de norma] -> escore 1-D monotônico -> regressão isotônica crescente nesse
escore. Alvo: ramo correto do tema sugerido (linhas `fora` = 0). Linhas: todas as
100 da folha x 5 repetições. Avaliação aninhada: para cada (rep, fold) o
calibrador é ajustado só nas linhas cujo id NÃO está no fold de teste. O
calibrador é guardado como arrays (coef, intercepto, limiares isotônicos) e
aplicado com np.interp, igual no CSV e na API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

B_BAGS = 25
LAW_BONUS = 2.8
CLIP = (0.01, 0.99)


def feats(pub, cons, cnt, score_nl):
    """Empilha as features de calibração.

    Args:
        pub: indicador de publicada (0/1).
        cons: indicador de consenso entre NB, BM25 e LSA.
        cnt: contagem de sacos do bagging com consenso (dividida por B_BAGS).
        score_nl: score do blend sem o bônus de norma.

    Returns:
        Matriz (n, 4) de floats.
    """
    return np.column_stack([np.asarray(pub, float), np.asarray(cons, float), np.asarray(cnt, float) / B_BAGS, np.asarray(score_nl, float)])


def feats_df(d: pd.DataFrame) -> np.ndarray:
    """Constrói a matriz de features a partir de um DataFrame OOF.

    Args:
        d: DataFrame com colunas score, law, pub, r0 e cnt.

    Returns:
        Matriz (n, 4) de features (score com o bônus de norma removido).
    """
    sc = d.score.values - LAW_BONUS * d.law.values.astype(float)
    return feats(d.pub.values, d.r0.values, d.cnt.values, sc)


def ajustar(X: np.ndarray, y: np.ndarray) -> dict:
    """Ajusta o calibrador: regressão logística seguida de isotônica.

    Args:
        X: matriz de features.
        y: alvo binário (ramo correto).

    Returns:
        Dicionário com coef, intercept, zx e zy (limiares isotônicos).
    """
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    z = lr.decision_function(X)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(z, y)
    return dict(coef=lr.coef_[0].copy(), intercept=float(lr.intercept_[0]), zx=iso.X_thresholds_.copy(), zy=iso.y_thresholds_.copy())


def aplicar(cal: dict, X: np.ndarray) -> np.ndarray:
    """Aplica o calibrador a uma matriz de features.

    Args:
        cal: dicionário retornado por ajustar.
        X: matriz de features.

    Returns:
        Probabilidades calibradas, cortadas ao intervalo CLIP.
    """
    z = X @ cal["coef"] + cal["intercept"]
    return np.clip(np.interp(z, cal["zx"], cal["zy"]), *CLIP)


def oof_aninhado(ex: dict, pub_mask_fn, n_reps: int, n_folds: int):
    """Calibra em validação aninhada e ajusta o calibrador de produção.

    Args:
        ex: dicionário {(rep, fold): DataFrame} com as linhas por questão.
        pub_mask_fn: função que recebe o DataFrame e devolve a máscara de publicadas.
        n_reps: número de repetições.
        n_folds: número de folds.

    Returns:
        Tupla (oof, cal): DataFrame OOF com a coluna cal e o calibrador
        de produção ajustado em todas as linhas.
    """
    rows = []
    for rep in range(n_reps):
        for k in range(n_folds):
            d = ex[(rep, k)].copy()
            d["pub"] = pub_mask_fn(ex[(rep, k)])
            d["rep"], d["fold"] = rep, k
            rows.append(d)
    oof = pd.concat(rows, ignore_index=True)
    oof["y"] = oof.ramo.clip(lower=0).astype(int)
    X = feats_df(oof)
    oof["cal"] = np.nan
    for rep in range(n_reps):
        for k in range(n_folds):
            te = (oof.rep == rep) & (oof.fold == k)
            ids_te = set(oof.id[te])
            tr = ~oof.id.isin(ids_te)
            cal = ajustar(X[tr.values], oof.y.values[tr.values])
            oof.loc[te, "cal"] = aplicar(cal, X[te.values])
    return oof, ajustar(X, oof.y.values)


def brier(p, y):
    """Calcula o escore de Brier.

    Args:
        p: probabilidades previstas.
        y: resultados observados (0/1).

    Returns:
        Erro quadrático médio.
    """
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def relatorio(oof: pd.DataFrame) -> dict:
    """Gera a tabela de confiabilidade e o Brier da confiança calibrada.

    Args:
        oof: DataFrame OOF com colunas cal, y e pub.

    Returns:
        Dicionário com a tabela por faixa, Brier calibrado (todas e publicadas),
        médias e precisões das publicadas e das revisar.
    """
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0001]
    lin = []
    cut = pd.cut(oof.cal, bins, right=False)
    for iv, g in oof.groupby(cut, observed=True):
        lin.append((f"{iv.left:.2f}-{min(iv.right, 1.0):.2f}", len(g), float(g.cal.mean()), float(g.y.mean())))
    pub = oof[oof.pub]
    return dict(tabela=lin, brier_new=brier(oof.cal, oof.y), brier_new_pub=brier(pub.cal, pub.y),
                mean_pub=float(pub.cal.mean()), prec_pub=float(pub.y.mean()), n_pub=len(pub), n=len(oof),
                mean_rev=float(oof[~oof.pub].cal.mean()), prec_rev=float(oof[~oof.pub].y.mean()))


def texto_relatorio(r: dict) -> str:
    """Formata o relatório de calibração em texto.

    Args:
        r: dicionário retornado por relatorio.

    Returns:
        Texto descritivo da calibração (para o relatório técnico).
    """
    t = "; ".join(f"[{b}] n={n} prevista {m:.2f} x observada {o:.2f}" for b, n, m, o in r["tabela"])
    return (f"Calibracao da confianca (OOF aninhado, 100 questoes x 5 reps, publicadas e revisar; LR[publicada, consenso, fracao bagging, score] + isotonica; alvo = ramo correto): {t}.\n"
            f"Brier da confianca calibrada: {r['brier_new']:.3f} (so publicadas: {r['brier_new_pub']:.3f}). "
            f"Media da confianca calibrada nas publicadas OOF {r['mean_pub']:.3f} x precisao observada {r['prec_pub']:.3f} (n={r['n_pub']}); nas revisar {r['mean_rev']:.3f} x {r['prec_rev']:.3f}.")
