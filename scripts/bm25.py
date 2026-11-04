"""Mecanismo de busca e relevância léxica (Okapi BM25).

Parte do pipeline de ./replicar.sh (importado por models.py). Componente de recuperação de informação desacoplado,
sem entradas/saídas em arquivo.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List
import numpy as np


class OkapiBM25:
    """Implementação vetorial de Okapi BM25 para alinhamento temático.

    Attributes:
        k1: saturação da frequência do termo.
        b: normalização pelo tamanho do documento.
        corpus_size: número de documentos indexados.
        doc_lens: comprimento (em tokens) de cada documento.
        avgdl: comprimento médio dos documentos.
        df: frequência de documentos por termo.
        doc_term_freqs: lista de frequências de termo por documento.
        idf: IDF de cada termo.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """Indexa o corpus tokenizado.

        Args:
            corpus_tokens: lista de documentos, cada um uma lista de tokens.
            k1: parâmetro de saturação do termo (padrão 1.5).
            b: parâmetro de normalização de tamanho (padrão 0.75).
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if self.corpus_size > 0 else 1.0

        self.df: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.doc_term_freqs.append(tf)
            for term in tf:
                self.df[term] = self.df.get(term, 0) + 1

        self.idf: Dict[str, float] = {
            t: float(np.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0))
            for t, n in self.df.items()
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Extrai tokens alfanuméricos relevantes (mínimo 3 caracteres).

        Args:
            text: texto bruto.

        Returns:
            Lista de tokens em ASCII minúsculo, sem acentos.
        """
        s = unicodedata.normalize("NFKD", str(text or "")).encode("ASCII", "ignore").decode("ASCII").lower()
        return re.findall(r"[a-z0-9]{3,}", s)

    def score_all_queries(self, queries_tokens: List[List[str]]) -> np.ndarray:
        """Calcula a matriz de pontuação BM25 (consultas x documentos).

        Args:
            queries_tokens: lista de consultas, cada uma uma lista de tokens.

        Returns:
            Matriz numpy de forma (n_consultas, n_documentos); linha zerada se
            nenhum termo da consulta existir no vocabulário.
        """
        n_q = len(queries_tokens)
        scores = np.zeros((n_q, self.corpus_size), dtype=float)

        for j, q_toks in enumerate(queries_tokens):
            q_terms = [t for t in q_toks if t in self.idf]
            if not q_terms:
                continue
            for i in range(self.corpus_size):
                tf_dict = self.doc_term_freqs[i]
                doc_len = self.doc_lens[i]
                s = 0.0
                for q in q_terms:
                    if q in tf_dict:
                        f = tf_dict[q]
                        denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                        s += self.idf[q] * (f * (self.k1 + 1.0)) / denom
                scores[j, i] = s

        return scores
