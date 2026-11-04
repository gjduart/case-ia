"""Pipeline híbrido assimétrico: ComplementNB + BM25 + LSA + normas.

Parte do pipeline de ./replicar.sh (usado por dados.py, embeddings.py e pela
API). Lê temas.csv, folha_verdade.csv e cache/{leis_por_questao,lei_tema}.csv e
cache/normas_senado.json (relativos a base_dir); não grava arquivos.

Arquitetura:
- Etapa 1: classificador hierárquico de raiz (metadados de origem).
- Etapa 2: ComplementNB supervisionado condicionado na árvore.
- Etapa 3: busca não supervisionada Okapi BM25 sobre ementas do Senado Federal.
- Etapa 4: projeção latente LSA (TruncatedSVD) sobre n-gramas do corpus.
- Etapa 5: injeção de normas unívocas (fast-track determinístico).
- Etapa 6: regra de publicação por consenso (ver scripts/decisao.py e scripts/roteador_area.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, normalize

from bm25 import OkapiBM25
from taxonomy import TaxonomyManager


class LawFeatureExtractor:
    """Extrai features categóricas e quantitativas das leis identificadas.

    Attributes:
        map: mapa id da questão -> {nl: nº de leis, tl: tem lei, tn: temas de
            nível 1 das normas, dl: disciplinas das normas}.
        mlb_tn: binarizador dos temas de nível 1 (ajustado em fit_transform).
        mlb_dl: binarizador das disciplinas (ajustado em fit_transform).
    """

    def __init__(self, leis_df: pd.DataFrame):
        """Indexa as leis por questão.

        Args:
            leis_df: DataFrame de cache/leis_por_questao.csv (id, n_leis, tem_lei,
                tema_lei_n1, disc_lei).
        """
        self.map: Dict[str, Dict[str, Any]] = {}
        for r in leis_df.itertuples():
            self.map[str(r.id)] = {
                "nl": float(r.n_leis) if not pd.isna(r.n_leis) else 0.0,
                "tl": float(bool(r.tem_lei)),
                "tn": [x.strip() for x in (str(r.tema_lei_n1) if pd.notna(r.tema_lei_n1) else "").split(";") if x.strip()],
                "dl": [x.strip() for x in (str(r.disc_lei) if pd.notna(r.disc_lei) else "").split(";") if x.strip()],
            }
        self.mlb_tn: MultiLabelBinarizer | None = None
        self.mlb_dl: MultiLabelBinarizer | None = None

    def _raw(self, ids: List[str]) -> Tuple[np.ndarray, List[List[str]], List[List[str]]]:
        """Monta as features brutas para uma lista de questões.

        Args:
            ids: ids das questões.

        Returns:
            Tupla (num, tn, dl): matriz log1p de [n_leis, tem_lei], listas de temas
            e listas de disciplinas por questão (vazias se o id for desconhecido).
        """
        num = np.log1p(np.array([[self.map.get(i, {}).get("nl", 0.0), self.map.get(i, {}).get("tl", 0.0)] for i in ids], dtype=float))
        tn = [self.map.get(i, {}).get("tn", []) for i in ids]
        dl = [self.map.get(i, {}).get("dl", []) for i in ids]
        return num, tn, dl

    def fit_transform(self, ids: List[str]) -> csr_matrix:
        """Ajusta os binarizadores e devolve a matriz de features.

        Args:
            ids: ids das questões de treino.

        Returns:
            Matriz esparsa CSR [numéricas | temas de nível 1 | disciplinas].
        """
        self.mlb_tn, self.mlb_dl = MultiLabelBinarizer(), MultiLabelBinarizer()
        num, tn, dl = self._raw(ids)
        return hstack([
            csr_matrix(num),
            csr_matrix(self.mlb_tn.fit_transform(tn)),
            csr_matrix(self.mlb_dl.fit_transform(dl))
        ]).tocsr()

    def transform(self, ids: List[str]) -> csr_matrix:
        """Transforma ids em features com os binarizadores já ajustados.

        Args:
            ids: ids das questões.

        Returns:
            Matriz esparsa CSR com as mesmas colunas de fit_transform.
        """
        num, tn, dl = self._raw(ids)
        return hstack([
            csr_matrix(num),
            csr_matrix(self.mlb_tn.transform(tn)),
            csr_matrix(self.mlb_dl.transform(dl))
        ]).tocsr()


class HybridClassifier:
    """Orquestrador do pipeline híbrido assimétrico.

    Attributes:
        base_dir: raiz do projeto.
        temas_df: DataFrame de temas.csv.
        folha_df: DataFrame de folha_verdade.csv.
        leis_df: leis por questão, indexado por id.
        lei_tema_df: mapa norma -> tema (cache/lei_tema.csv).
        normas_senado: dicionário de normas (cache/normas_senado.json).
        tax: TaxonomyManager da árvore.
        law_fe: LawFeatureExtractor.
        senado_por_tema: texto de apelido e ementa por tema de nível 1.
        class_texts: documento enriquecido de cada tema de nível 1 ativo.
        bm25: índice OkapiBM25 dos temas.
        vec_lsa: vetorizador TF-IDF do LSA (após fit_lsa).
        svd: TruncatedSVD do LSA (após fit_lsa).
        D_themes_lsa: vetores LSA dos temas (após fit_lsa).
    """

    def __init__(self, base_dir: Path):
        """Carrega os dados e constrói taxonomia, textos de classe e índice BM25.

        Args:
            base_dir: raiz do projeto (contém temas.csv, folha_verdade.csv e cache/).
        """
        self.base_dir = base_dir
        self.temas_df = pd.read_csv(base_dir / "temas.csv")
        self.folha_df = pd.read_csv(base_dir / "folha_verdade.csv")
        self.leis_df = pd.read_csv(base_dir / "cache" / "leis_por_questao.csv").set_index("id")
        self.lei_tema_df = pd.read_csv(base_dir / "cache" / "lei_tema.csv")
        
        with open(base_dir / "cache" / "normas_senado.json", "r", encoding="utf-8") as f:
            self.normas_senado = json.load(f)["normas"]

        self.tax = TaxonomyManager(self.temas_df)
        self.law_fe = LawFeatureExtractor(pd.read_csv(base_dir / "cache" / "leis_por_questao.csv"))

        # Mapa de enriquecimento do Senado
        self.senado_por_tema: Dict[int, str] = {}
        for r in self.lei_tema_df.itertuples():
            tid = int(r.tema_id)
            if tid in self.tax.act_l1:
                info = f"{r.apelido or ''} {r.ementa or ''}"
                self.senado_por_tema[tid] = self.senado_por_tema.get(tid, "") + " " + info

        # Constrói textos de classes enriquecidos com ementas oficiais do Senado
        self.class_texts = [
            self.tax.build_pseudo_document(c, self.senado_por_tema.get(c, ""))
            for c in self.tax.act_l1
        ]
        
        # Indexador Okapi BM25 sobre a taxonomia expandida
        theme_tokens = [OkapiBM25.tokenize(t) for t in self.class_texts]
        self.bm25 = OkapiBM25(theme_tokens)

        # Projeção LSA (TruncatedSVD)
        self.vec_lsa: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.D_themes_lsa: np.ndarray | None = None


    def fit_lsa(self, corpus_texts: List[str]):
        """Ajusta o espaço LSA (d=50) com questões e textos dos temas.

        Args:
            corpus_texts: textos das questões do corpus.
        """
        all_texts = corpus_texts + self.class_texts
        self.vec_lsa = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
        self.vec_lsa.fit(all_texts)
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        X_all = normalize(self.svd.fit_transform(self.vec_lsa.transform(all_texts)))
        self.D_themes_lsa = X_all[len(corpus_texts):]

    def compute_bm25_matrix(self, query_texts: List[str]) -> np.ndarray:
        """Calcula a similaridade BM25 normalizada pelo máximo de cada consulta.

        Args:
            query_texts: textos das questões.

        Returns:
            Matriz (n_consultas x n_temas de nível 1) com valores em [0, 1].
        """
        q_tokens = [OkapiBM25.tokenize(t) for t in query_texts]
        raw_scores = self.bm25.score_all_queries(q_tokens)
        norm_scores = np.zeros_like(raw_scores)
        for i in range(len(raw_scores)):
            m = raw_scores[i].max()
            if m > 0:
                norm_scores[i] = raw_scores[i] / m
        return norm_scores

    def compute_lsa_matrix(self, query_texts: List[str]) -> np.ndarray:
        """Calcula a similaridade semântica latente (LSA) com os temas.

        Args:
            query_texts: textos das questões.

        Returns:
            Matriz (n_consultas x n_temas de nível 1) de cossenos truncados em 0.

        Raises:
            ValueError: se fit_lsa ainda não foi chamado.
        """
        if self.vec_lsa is None or self.svd is None or self.D_themes_lsa is None:
            raise ValueError("LSA não inicializado.")
        X_q = normalize(self.svd.transform(self.vec_lsa.transform(query_texts)))
        sim = X_q @ self.D_themes_lsa.T
        return np.maximum(sim, 0.0)

    def fit_predict_hierarchy(self, trn_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
        """Treina e infere a hierarquia em duas etapas num só passo.

        Args:
            trn_df: DataFrame de treino (colunas y, texto_modelo, id, metadados).
            pred_df: DataFrame a prever.

        Returns:
            Matriz de probabilidades (n x n temas de nível 1 ativos).
        """
        return self.predict_hierarchy(self.fit_hierarchy(trn_df), pred_df)

    def fit_hierarchy(self, trn_df: pd.DataFrame) -> Dict[str, Any]:
        """Ajusta a hierarquia e devolve o estado ajustado (picklável).

        Etapa 1: disciplina-raiz por metadados (ComplementNB). Etapa 2: tema de
        nível 1 por ComplementNB sobre TF-IDF de palavras e caracteres, metadados e
        features de leis, aumentado com o texto de cada classe.

        Args:
            trn_df: DataFrame de treino com y (tema nível 1), texto_modelo, id,
                disciplina_origem e assunto_origem.

        Returns:
            Dicionário com ohe1, m1, word, char, ohe2, law_fe e m2.
        """
        import copy
        cls_idx = {c: i for i, c in enumerate(self.tax.act_l1)}
        root_idx = {r: i for i, r in enumerate(self.tax.roots)}
        cols = ["disciplina_origem", "assunto_origem"]

        # Etapa 1: Disciplina-Raiz via Metadados de Origem
        ohe1 = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        S_tr = normalize(ohe1.fit_transform(trn_df[cols].fillna("NA")))
        ytr1 = np.array([root_idx[int(self.tax.pai[int(y)])] for y in trn_df.y])
        m1 = ComplementNB(alpha=1.0).fit(S_tr, ytr1)

        # Etapa 2: Tema Nível 1 condicionado
        txt_tr = trn_df.texto_modelo.fillna("").astype(str).values
        classes = sorted(int(x) for x in trn_df.y.unique())
        aug_txt = [self.class_texts[cls_idx[c]] for c in classes]
        fit_txt = np.concatenate([txt_tr, np.array(aug_txt, dtype=object)])

        word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode", lowercase=True, max_features=20000).fit(fit_txt)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, sublinear_tf=True, lowercase=True, max_features=20000).fit(fit_txt)

        W_tr, C_tr = word.transform(txt_tr), char.transform(txt_tr)
        W_aug, C_aug = word.transform(aug_txt), char.transform(aug_txt)

        ohe2 = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        O_tr = ohe2.fit_transform(trn_df[cols].fillna("NA"))

        law_tr = self.law_fe.fit_transform(trn_df.id.values)
        law_fe = copy.copy(self.law_fe)  # snapshot dos binarizadores ajustados (reatribuidos, nao mutados, a cada novo fit)

        X_tr = normalize(hstack([W_tr, C_tr, O_tr, law_tr]).tocsr())
        text_aug = normalize(hstack([W_aug, C_aug]).tocsr())
        pad = csr_matrix((len(classes), X_tr.shape[1] - text_aug.shape[1]))
        X_aug = normalize(vstack([X_tr, hstack([text_aug, pad]).tocsr()]).tocsr())
        y_aug = np.concatenate([trn_df.y.values.astype(int), np.array(classes, dtype=int)])

        m2 = ComplementNB(alpha=0.05).fit(X_aug, y_aug)
        return dict(ohe1=ohe1, m1=m1, word=word, char=char, ohe2=ohe2, law_fe=law_fe, m2=m2)

    def predict_hierarchy(self, st: Dict[str, Any], pred_df: pd.DataFrame) -> np.ndarray:
        """Infere com o estado devolvido por fit_hierarchy.

        A raiz prevista restringe os temas permitidos e a massa de probabilidade da
        etapa 2 é renormalizada dentro dela.

        Args:
            st: estado retornado por fit_hierarchy.
            pred_df: DataFrame a prever.

        Returns:
            Matriz de probabilidades (n x n temas de nível 1 ativos).
        """
        n = len(pred_df)
        cls_idx = {c: i for i, c in enumerate(self.tax.act_l1)}
        cols = ["disciplina_origem", "assunto_origem"]
        P = np.zeros((n, len(self.tax.act_l1)), dtype=float)

        S_pr = normalize(st["ohe1"].transform(pred_df[cols].fillna("NA")))
        p1 = st["m1"].predict_proba(S_pr)
        P1 = np.zeros((n, len(self.tax.roots)))
        for j, c in enumerate(st["m1"].classes_):
            P1[:, int(c)] = p1[:, j]
        pred_root = P1.argmax(axis=1)

        txt_pr = pred_df.texto_modelo.fillna("").astype(str).values
        W_pr, C_pr = st["word"].transform(txt_pr), st["char"].transform(txt_pr)
        O_pr = st["ohe2"].transform(pred_df[cols].fillna("NA"))
        law_pr = st["law_fe"].transform(pred_df.id.values)
        X_pr = normalize(hstack([W_pr, C_pr, O_pr, law_pr]).tocsr())

        m2 = st["m2"]
        p2 = m2.predict_proba(X_pr)
        col2 = {int(c): j for j, c in enumerate(m2.classes_)}

        for j in range(n):
            r = self.tax.roots[int(pred_root[j])]
            allowed = [c for c in self.tax.kids[r] if c in col2 and c in cls_idx]
            mass = sum(float(p2[j, col2[c]]) for c in allowed)
            if mass > 0:
                for c in allowed:
                    P[j, cls_idx[c]] = float(p2[j, col2[c]]) / mass

        return P
