"""Configuração A1: roteador por área + regras por área. Sem LLM e sem rede.

Parte do pipeline de ./replicar.sh (importado por validacao.py, gerar_publicacao.py e
pela API). Lê os embeddings Qwen3 de cache/embeddings/qwen (via dados.Contexto); não
grava arquivos próprios.

Motivo: o estágio 1 do NB usa a disciplina_origem como metadado e as questões novas
(Q0360+) nunca trazem "Direito Penal - Legislação Especial"; logo o NB roteia para Penal e
o consenso NB = BM25 = LSA falha: Legislação Especial e Criminologia ficavam 100% em
revisar na S4 (configuração anterior, só Penal/Processo).

Regras (fixadas antes de rodar, tudo aprendido só no treino do fold):
  Roteador = LogisticRegression(C=10) só sobre o embedding Qwen3 (SEM disciplina/assunto),
  alvo = RAIZ, treinada em sintéticos filtrados + descrições + linhas não-fora da folha
  do treino (peso 4).
  Penal / Processo Penal -> decisão da S4 (decisao.py), inalterada.
  Legislação Especial -> publica sse BM25 (docs enriquecidos com leis) top-1 == qwen-LR
  top-1 == tema (restritos à raiz roteada) E (norma unívoca == tema OU nenhuma norma
  citada) E bagging (25 subamostras de 80%) tau = 0,8 E piso de score da área.
  Criminologia -> publica sse BM25 top-1 == qwen-LR top-1 E bagging tau = 0,8 E piso.
Score = logit do blend 0,55 qwen-LR + 0,30 BM25 + 0,15 LSA, mais 2,8 se a norma concorda.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dados import C_LR, W_FOLHA, Contexto
from decisao import B_BAGS, BONUS_NORMA, FRAC_BAG, PESOS, SEED_BAG

warnings.filterwarnings("ignore")
PEN, PROC, LEG, CRIM = 5062, 5465, 4683, 4365
AREAS_S4 = (PEN, PROC)
AREAS_NOVAS = (LEG, CRIM)
INSTRUCAO = "Dada uma questao de prova juridica brasileira, recuperar o tema juridico (assunto) ao qual ela pertence"


def roteador_ajustar(ctx: Contexto, lab_ids, ytrue) -> LogisticRegression:
    """Treina o roteador de raiz (área) só sobre embeddings de texto.

    Args:
        ctx: contexto de dados.
        lab_ids: ids das linhas rotuladas do treino.
        ytrue: mapa id -> tema de nível 1 verdadeiro.

    Returns:
        LogisticRegression ajustada, cujas classes são os ids das raízes.
    """
    k, tax = ctx.keep, ctx.tax
    X = [ctx.E_sint[k], ctx.E_desc, ctx.E_q[[ctx.qidx[i] for i in lab_ids]]]
    y = [[tax.pai[int(t)] for t in ctx.sint_tema[k]], [tax.pai[int(t)] for t in ctx.desc_tema], [tax.pai[int(ytrue[i])] for i in lab_ids]]
    w = [np.ones(int(k.sum())), np.ones(len(ctx.desc_tema)), np.full(len(lab_ids), W_FOLHA)]
    return LogisticRegression(C=C_LR, max_iter=500).fit(np.vstack(X), np.concatenate([np.asarray(a, int) for a in y]), sample_weight=np.concatenate(w))


def tema_ajustar(ctx: Contexto, lab_ids) -> LogisticRegression:
    """Treina a regressão logística de tema (qwen-LR) sobre embeddings.

    Usa sintéticos filtrados, descrições, pseudo-documentos dos temas e as linhas da
    folha dadas (peso W_FOLHA).

    Args:
        ctx: contexto de dados.
        lab_ids: ids das linhas rotuladas da folha.

    Returns:
        LogisticRegression ajustada.
    """
    k = ctx.keep
    X = [ctx.E_sint[k], ctx.E_desc, ctx.E_pseudo[ctx.pseudo_rows]]
    y = [[ctx.cls_idx[t] for t in ctx.sint_tema[k]], [ctx.cls_idx[t] for t in ctx.desc_tema], list(ctx.pseudo_rows)]
    w = [np.ones(k.sum()), np.ones(len(ctx.desc_tema)), np.ones(len(ctx.pseudo_rows))]
    if len(lab_ids):
        X.append(ctx.E_q[[ctx.qidx[i] for i in lab_ids]])
        y.append([ctx.ylab[i] for i in lab_ids])
        w.append(np.full(len(lab_ids), W_FOLHA))
    X, y, w = np.vstack(X), np.concatenate([np.asarray(a, int) for a in y]), np.concatenate(w)
    return LogisticRegression(C=C_LR, max_iter=500).fit(X, y, sample_weight=w)


def prob_tema(ctx: Contexto, m: LogisticRegression, E: np.ndarray) -> np.ndarray:
    """Calcula as probabilidades de tema expandidas a todos os temas ativos.

    Args:
        ctx: contexto de dados.
        m: modelo de tema ajustado.
        E: embeddings das questões.

    Returns:
        Matriz (n x temas de nível 1 ativos).
    """
    return ctx.expandir(m, m.predict_proba(E))


def colunas_por_raiz(ctx: Contexto) -> dict:
    """Lista as colunas (índices em act_l1) dos temas não duplicados de cada raiz.

    Args:
        ctx: contexto de dados.

    Returns:
        Dicionário {raiz: lista de índices de coluna}.
    """
    return {int(r): [ctx.cls_idx[c] for c in ctx.tax.kids[r] if ctx.mask[ctx.cls_idx[c]] > 0] for r in ctx.tax.roots}


def restringir(cols: dict, P: np.ndarray, raizes) -> np.ndarray:
    """Escolhe o tema de maior probabilidade restrito à raiz de cada questão.

    Args:
        cols: saída de colunas_por_raiz.
        P: matriz de scores (n x temas).
        raizes: raiz atribuída a cada questão.

    Returns:
        Array com o índice de coluna escolhido por questão.
    """
    out = np.zeros(len(raizes), int)
    for j, r in enumerate(raizes):
        c = cols[int(r)]
        out[j] = c[int(np.argmax(P[j, c]))]
    return out


def nucleo(cols: dict, bm: np.ndarray, ls: np.ndarray, law_cls: np.ndarray, rt: np.ndarray, Pq: np.ndarray):
    """Calcula o núcleo de decisão das áreas novas.

    Args:
        cols: saída de colunas_por_raiz.
        bm: scores BM25 normalizados.
        ls: scores LSA normalizados.
        law_cls: coluna do tema indicado pela norma unívoca (ou -1).
        rt: raiz roteada de cada questão.
        Pq: probabilidades qwen-LR.

    Returns:
        Tupla (tema, BM25 == tema, score logit com bônus de norma, norma concorda).
    """
    th, bt = restringir(cols, Pq, rt), restringir(cols, bm, rt)
    P7 = PESOS[0] * Pq + PESOS[1] * bm + PESOS[2] * ls
    law_ag = law_cls == th
    p = np.clip(P7[np.arange(len(th)), th], 1e-6, 1 - 1e-6)
    return th, bt == th, np.log(p / (1 - p)) + BONUS_NORMA * law_ag, law_ag


def decidir_fold(ctx: Contexto, trn_ids, tst_ids, seed: int, guardar: bool = False):
    """Roteia, treina a qwen-LR, faz o bagging e calcula o piso de score de um fold.

    Args:
        ctx: contexto de dados.
        trn_ids: ids de treino.
        tst_ids: ids a decidir.
        seed: semente da CV interna e do bagging (os mesmos sacos da S4).
        guardar: se True, devolve também os modelos (produção).

    Returns:
        Tupla (DataFrame com rt, th, gate, cnt, score, floor, cited, lawag e thid;
        dicionário de modelos ou None).
    """
    tax, cols = ctx.tax, colunas_por_raiz(ctx)
    ytrue = dict(zip(ctx.sup.id, ctx.sup.y))
    lab = ctx.sup[ctx.sup.id.isin(trn_ids) & (ctx.sup.y != -1)]
    ids = list(tst_ids)
    ti = np.array([ctx.qidx[i] for i in ids])
    bm, ls, lw = ctx.S_bm[ti], ctx.S_ls[ti], ctx.law_cls[ti]
    rm = roteador_ajustar(ctx, list(lab.id), ytrue)
    rt = rm.classes_[rm.predict_proba(ctx.E_q[ti]).argmax(1)]
    full = tema_ajustar(ctx, lab.id)
    th, gate, sc, law_ag = nucleo(cols, bm, ls, lw, rt, prob_tema(ctx, full, ctx.E_q[ti]))
    rng = np.random.RandomState(SEED_BAG + seed)
    cnt = np.zeros(len(ids), int)
    bags = []
    for _ in range(B_BAGS):
        idx = rng.choice(len(lab), int(round(FRAC_BAG * len(lab))), replace=False)
        mb = tema_ajustar(ctx, lab.iloc[idx].id)
        cnt += (restringir(cols, prob_tema(ctx, mb, ctx.E_q[ti]), rt) == th)
        if guardar:
            bags.append(mb)
    sup_tr = ctx.sup[ctx.sup.id.isin(trn_ids)].reset_index(drop=True)
    sc_in, rt_in = [], []
    for a, bb in StratifiedKFold(5, shuffle=True, random_state=seed).split(sup_tr, sup_tr.estr):
        li = sup_tr.iloc[a]
        li = li[li.y != -1]
        te = list(sup_tr.id.iloc[bb])
        tj = np.array([ctx.qidx[i] for i in te])
        rmi = roteador_ajustar(ctx, list(li.id), ytrue)
        rti = rmi.classes_[rmi.predict_proba(ctx.E_q[tj]).argmax(1)]
        _, gi, si, _ = nucleo(cols, ctx.S_bm[tj], ctx.S_ls[tj], ctx.law_cls[tj], rti, prob_tema(ctx, tema_ajustar(ctx, li.id), ctx.E_q[tj]))
        sc_in += list(si[gi])
        rt_in += list(rti[gi])
    sc_in, rt_in = np.array(sc_in), np.array(rt_in)
    qg = float(np.quantile(sc_in, 0.25)) if len(sc_in) else np.inf
    qa = {int(r): (float(np.quantile(sc_in[rt_in == r], 0.25)) if (rt_in == r).sum() >= 4 else qg) for r in tax.roots}
    L = ctx.clf.leis_df
    cited = np.array([bool(L.loc[i, "tem_lei"]) if i in L.index else False for i in ids])
    d = pd.DataFrame(dict(id=ids, rt=rt, th=th, gate=gate, cnt=cnt, score=sc, floor=[qa[int(r)] for r in rt],
                          cited=cited, lawag=law_ag, thid=[tax.act_l1[x] for x in th]))
    mod = dict(router=rm, full=full, bags=bags, qa=qa, cols=cols) if guardar else None
    return d, mod


def combinar(ctx: Contexto, s4: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """Combina a decisão S4 com a das áreas novas conforme a raiz roteada.

    Nas questões roteadas a Legislação Especial ou Criminologia vale a regra das áreas
    novas (r0 = BM25 == qwen-LR e norma ok; cnt, score e piso do roteador); nas demais,
    a S4 inalterada.

    Args:
        ctx: contexto de dados.
        s4: saída de decisao.decidir_fold.
        n: saída (DataFrame) de decidir_fold deste módulo, na mesma ordem de ids.

    Returns:
        DataFrame com pred, r0, cnt, score, qg, qg_s4, law, root, ramo, rt, nova_area,
        gate_bm, lawok e ag_* por questão.
    """
    tax = ctx.tax
    assert list(s4.id) == list(n.id)
    d = s4.copy()
    d["qg_s4"] = s4.qg.values  # piso da S4 (a API usa nas áreas Penal/Processo)
    nova = n.rt.isin(AREAS_NOVAS).values
    lawok = np.where(n.rt == LEG, n.lawag.values | ~n.cited.values, True)
    d["rt"] = n.rt.values
    d["nova_area"] = nova
    d["gate_bm"] = n.gate.values
    d["lawok"] = lawok
    for col, v in (("pred", n.thid.values), ("r0", n.gate.values & lawok), ("cnt", n.cnt.values), ("score", n.score.values), ("qg", n.floor.values),
                   ("law", n.lawag.values)):
        d[col] = np.where(nova, v, s4[col].values)
    d["r0"] = d.r0.astype(bool)
    d["root"] = [tax.pai[int(p)] for p in d.pred]
    d["ramo"] = [ctx.ramo(int(p), i) for p, i in zip(d.pred, d.id)]
    return d
