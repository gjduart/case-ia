"""Dados da entrega: questões normalizadas, folha, taxonomia, BM25/LSA, normas e embeddings.

Parte do pipeline de ./replicar.sh (importado por roteador_area, decisao, validacao e
gerar_publicacao). Lê questoes.jsonl, temas.csv, folha_verdade.csv, cache/leis_por_questao.csv,
cache/lei_tema.csv, cache/normas_senado.json, cache/embeddings/<modelo>/ e
cache/sinteticos_filtrados.jsonl. Não grava arquivos.

A classe Contexto reúne tudo que as etapas seguintes precisam: texto normalizado das 659
questões (preprocessor.TextCleaner), taxonomia, nós duplicados mesclados (estrutura.py),
similaridades BM25 e LSA (transdutivas, sem rótulos), tema indicado por norma unívoca,
partição da folha em estratos e os vetores (Qwen3) das questões, dos sintéticos filtrados,
das descrições e dos pseudo-documentos dos temas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from estrutura import nos_duplicados
from models import HybridClassifier
from preprocessor import TextCleaner

BASE_DIR = Path(__file__).resolve().parents[1]
EMB = BASE_DIR / "cache" / "embeddings"
ID_NOVO_MIN = 360  # Q0360+ = concursos recentes (fora da distribuição da folha)
W_FOLHA = 4.0      # peso das linhas da folha nos modelos sobre embeddings
C_LR = 10.0        # regularização das regressões logísticas (embeddings normalizados)


def renorm(a: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para somar 1 (linhas nulas permanecem nulas).

    Args:
        a: matriz não negativa.

    Returns:
        Matriz com linhas de soma 1.
    """
    return a / np.maximum(a.sum(axis=1, keepdims=True), 1e-9)


class Contexto:
    """Dados já carregados e transformados, compartilhados por todas as etapas.

    Attributes:
        clf: HybridClassifier (NB hierárquico, BM25, LSA, normas).
        tax: TaxonomyManager.
        q: DataFrame das 659 questões, com a coluna texto_modelo (texto normalizado).
        qidx: mapa id da questão -> linha em q.
        n: número de questões.
        K: número de temas de nível 1 ativos.
        cls_idx: mapa id do tema -> coluna em tax.act_l1.
        dup: nós de nível 1 duplicados -> tema alvo da mesclagem.
        mask: 0 nos temas duplicados, 1 nos demais (colunas de act_l1).
        sup: folha_verdade unida às questões, com y (tema de nível 1; -1 = fora) e estr (estrato).
        folha: folha_verdade indexada por id.
        ylab: mapa id -> coluna do tema de nível 1 (-1 para "fora").
        novo: indicador de questão nova (Q0360+).
        law_cls: coluna do tema indicado por norma unívoca (-1 se não há).
        S_bm: similaridade BM25 normalizada (n x K).
        S_ls: similaridade LSA normalizada (n x K).
        E_q, E_pseudo, E_sint, E_desc: embeddings de questões, temas, sintéticos e descrições.
        sint_ids, desc_ids: ids dos vetores sintéticos e das descrições.
        sint_tema, desc_tema: tema efetivo (com duplicados mesclados) de cada vetor.
        keep: máscara dos sintéticos mantidos pelo filtro automático.
        pseudo_rows: colunas dos temas não duplicados.
    """

    def __init__(self, modelo: str = "qwen", aplicar_filtro: bool = True):
        """Carrega os dados, ajusta BM25/LSA e lê os embeddings do cache.

        Args:
            modelo: subpasta de cache/embeddings ("qwen" na entrega; "e5" só no filtro).
            aplicar_filtro: se False, mantém todos os sintéticos (usado ao regenerar o filtro).
        """
        clf = HybridClassifier(BASE_DIR)
        self.clf, self.tax = clf, clf.tax
        raw = [json.loads(l) for l in (BASE_DIR / "questoes.jsonl").read_text(encoding="utf-8").splitlines()]
        q = pd.DataFrame(raw)
        q["texto_modelo"] = [TextCleaner.extract_question_text(r.get("enunciado_html"), r.get("comando"), r.get("alternativas"), include_alternativas=True) for r in raw]
        self.q = q
        self.qidx = {qid: i for i, qid in enumerate(q.id)}
        self.n, self.K = len(q), len(self.tax.act_l1)
        self.cls_idx = {c: i for i, c in enumerate(self.tax.act_l1)}
        # nós duplicados (regra genérica; scripts/estrutura.py)
        self.dup = nos_duplicados(self.tax)
        self.mask = np.array([0.0 if t in self.dup else 1.0 for t in self.tax.act_l1])
        # folha e estratos da validação cruzada
        sup = clf.folha_df.merge(q[["id", "texto_modelo", "disciplina_origem", "assunto_origem"]], on="id", how="left")
        sup["y"] = [-1 if pd.isna(c) else self.tax.get_parent_l1(int(c)) for c in sup.tema_central_id]
        sup["y"] = sup["y"].astype(int)
        cont = sup.y.value_counts().to_dict()
        sup["estr"] = sup.y.map(lambda y: "FORA" if y == -1 else (f"y:{y}" if cont.get(y, 0) >= 5 else f"disc:{self.tax.pai.get(y, -1)}"))
        self.sup = sup
        self.folha = clf.folha_df.set_index("id")
        self.ylab = {r.id: (self.cls_idx[r.y] if r.y != -1 else -1) for r in sup.itertuples()}
        self.novo = np.array([int(i[1:]) >= ID_NOVO_MIN for i in q.id])
        # norma citada que indica um único tema de nível 1 (fast-track)
        L = clf.leis_df
        self.law_cls = np.full(self.n, -1)
        for qid, i in self.qidx.items():
            if qid in L.index and pd.notna(L.loc[qid, "tema_lei_n1"]):
                tids = [int(x) for x in str(L.loc[qid, "tema_lei_n1"]).split(";") if x.strip()]
                if len(tids) == 1 and tids[0] in self.cls_idx:
                    self.law_cls[i] = self.cls_idx[tids[0]]
        # BM25 / LSA (transdutivos, independentes de rótulos)
        texts = list(q.texto_modelo)
        clf.fit_lsa(texts)
        self.S_bm = renorm(clf.compute_bm25_matrix(texts))
        self.S_ls = renorm(clf.compute_lsa_matrix(texts))
        # embeddings
        d = EMB / modelo
        self.E_pseudo = np.load(d / "pseudo.npy")
        assert json.loads((d / "pseudo_ids.json").read_text()) == [str(t) for t in self.tax.act_l1]
        self.E_q = None
        if (d / "questoes.npy").exists():
            self.E_q = np.load(d / "questoes.npy")
            assert json.loads((d / "questoes_ids.json").read_text()) == list(q.id)
        self.E_sint = np.load(d / "sint.npy")
        self.sint_ids = json.loads((d / "sint_ids.json").read_text())
        self.E_desc = np.load(d / "sint_desc.npy")
        self.desc_ids = json.loads((d / "sint_desc_ids.json").read_text())
        self.sint_tema = np.array([self.eff(int(s.split("#")[0])) for s in self.sint_ids])
        self.desc_tema = np.array([self.eff(int(s.split("#")[0])) for s in self.desc_ids])
        self.keep = self._carregar_filtro() if aplicar_filtro else np.ones(len(self.sint_ids), dtype=bool)
        self.pseudo_rows = np.array([i for i, t in enumerate(self.tax.act_l1) if t not in self.dup])

    def eff(self, tid: int) -> int:
        """Resolve o tema efetivo, mesclando duplicados no alvo.

        Args:
            tid: id do tema.

        Returns:
            Id do alvo se for duplicado; senão o próprio id.
        """
        return self.dup.get(tid, tid)

    def _carregar_filtro(self) -> np.ndarray:
        """Carrega a máscara de sintéticos mantidos pelo filtro automático.

        Returns:
            Vetor booleano na ordem de sint_ids (cache/sinteticos_filtrados.jsonl).
        """
        p = BASE_DIR / "cache" / "sinteticos_filtrados.jsonl"
        ok = {json.loads(l)["emb_id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
        return np.array([s in ok for s in self.sint_ids])

    def ramo(self, pred_id: int, qid: str) -> int:
        """Indica se a predição está no ramo correto da folha.

        Args:
            pred_id: id do tema predito.
            qid: id da questão.

        Returns:
            1 se correta, 0 se errada, -1 se a questão não está na folha.
        """
        if qid not in self.folha.index:
            return -1
        r = self.folha.loc[qid]
        return int(self.tax.is_correct_branch(pred_id, r.tema_central_id, r.extras_ids))

    def expandir(self, m, P: np.ndarray) -> np.ndarray:
        """Expande probabilidades de um modelo às K colunas de temas.

        Args:
            m: modelo ajustado (com classes_).
            P: probabilidades nas classes do modelo.

        Returns:
            Matriz (n x K) com zeros nas classes ausentes.
        """
        out = np.zeros((P.shape[0], self.K))
        out[:, m.classes_] = P
        return out

    def lei_decide(self, ids, P: np.ndarray) -> np.ndarray:
        """Escolhe o tema de maior probabilidade, forçado pela norma unívoca quando há.

        Args:
            ids: ids das questões (linhas de P).
            P: matriz de probabilidades (n x K).

        Returns:
            Coluna do tema escolhido por questão.
        """
        b = P.argmax(axis=1).copy()
        for j, qid in enumerate(ids):
            lc = self.law_cls[self.qidx[qid]]
            if lc >= 0 and self.mask[lc] > 0:
                b[j] = lc
        return b
