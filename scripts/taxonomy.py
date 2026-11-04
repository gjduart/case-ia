"""Taxonomia e validação do domínio jurídico.

Parte do pipeline de ./replicar.sh (usada por praticamente todos os scripts).
Encapsula as regras estruturais da árvore da plataforma (raiz, tema nível 1,
subtema nível 2) e a métrica da banca (acerto por ramo pai/filho; erro por irmão).

Entradas: DataFrame lido de temas.csv.
Saídas: nenhuma em arquivo; fornece a classe TaxonomyManager.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set
import pandas as pd


class TaxonomyManager:
    """Gerencia a árvore de temas e a navegação hierárquica.

    Attributes:
        rows: mapa tema_id -> linha (namedtuple) de temas.csv.
        pai: mapa tema_id -> id do pai (None para raízes).
        status: mapa tema_id -> status do tema ("active" etc.).
        nivel: mapa tema_id -> nível (0 raiz, 1 tema, 2 subtema).
        titulo: mapa tema_id -> título.
        caminho: mapa tema_id -> caminho completo do tema.
        roots: ids das raízes (nível 0) ativas, ordenados.
        act_l1: ids dos temas de nível 1 ativos, ordenados.
        kids: mapa raiz -> temas de nível 1 ativos filhos dela.
        children: mapa tema_id -> todos os filhos diretos (qualquer status).
    """

    def __init__(self, temas_df: pd.DataFrame):
        """Indexa a tabela de temas em dicionários de acesso rápido.

        Args:
            temas_df: DataFrame de temas.csv com as colunas tema_id, parent_id,
                status, nivel, titulo, caminho e descricao.
        """
        self.rows: Dict[int, Any] = {int(r.tema_id): r for r in temas_df.itertuples()}
        self.pai: Dict[int, Optional[int]] = {
            tid: (None if pd.isna(r.parent_id) else int(r.parent_id))
            for tid, r in self.rows.items()
        }
        self.status: Dict[int, str] = {tid: r.status for tid, r in self.rows.items()}
        self.nivel: Dict[int, int] = {tid: int(r.nivel) for tid, r in self.rows.items()}
        self.titulo: Dict[int, str] = {tid: str(r.titulo) for tid, r in self.rows.items()}
        self.caminho: Dict[int, str] = {tid: str(r.caminho) for tid, r in self.rows.items()}

        self.roots: List[int] = sorted(
            tid for tid, r in self.rows.items()
            if int(r.nivel) == 0 and r.status == "active"
        )
        self.act_l1: List[int] = sorted(
            tid for tid, r in self.rows.items()
            if int(r.nivel) == 1 and r.status == "active"
        )
        self.kids: Dict[int, List[int]] = {
            r: [c for c in self.act_l1 if self.pai.get(c) == r]
            for r in self.roots
        }
        self.children: Dict[int, List[int]] = {tid: [] for tid in self.rows}
        for c, p in self.pai.items():
            if p is not None:
                self.children[p].append(c)

    def get_ramo(self, tema_id: int) -> Set[int]:
        """Devolve o conjunto de nós aceitos pela régua do case.

        O ramo de um tema é {tema, pai, filhos ativos}.

        Args:
            tema_id: identificador do tema de referência.

        Returns:
            Conjunto de ids de temas que contam como acerto para esse tema.
        """
        tid = int(tema_id)
        out = {tid}
        p = self.pai.get(tid)
        if p is not None:
            out.add(p)
        out.update(c for c in self.children.get(tid, []) if self.status.get(c) == "active")
        return out

    def get_parent_l1(self, tema_id: int) -> int:
        """Devolve o tema de nível 1 correspondente ao nó.

        Args:
            tema_id: id de um nó de nível 1 ou 2.

        Returns:
            O próprio id se o nó for de nível 1; caso contrário o id do pai
            (ou o próprio id se não houver pai).
        """
        tid = int(tema_id)
        if self.nivel.get(tid) == 1:
            return tid
        p = self.pai.get(tid)
        return int(p) if p is not None else tid

    def build_pseudo_document(self, tid: int, senado_enrichment: str = "") -> str:
        """Constrói a descrição rica do nó para busca lexical/semântica.

        Concatena título, títulos de até 15 filhos, título do pai e a descrição
        (sem a frase "N questões no source" e sem caracteres não latinos), mais o
        enriquecimento opcional.

        Args:
            tid: id do tema.
            senado_enrichment: texto extra (ex.: normas) anexado ao documento.

        Returns:
            Texto único usado como documento do tema em BM25, TF-IDF e LSA.
        """
        r = self.rows[tid]
        parts = [str(r.titulo)]
        parts += [str(self.rows[k].titulo) for k in self.children.get(tid, [])[:15]]
        p = self.pai.get(tid)
        if p is not None and p in self.rows:
            parts.append(str(self.rows[p].titulo))
            
        desc = re.sub(r"[\u0590-\u08ff]+", " ", re.sub(r"\d+\s*questões no source", "", str(r.descricao or "")))
        parts.append(desc)
        if senado_enrichment:
            parts.append(senado_enrichment)
            
        return " ".join(parts).strip()

    @staticmethod
    def parse_extras(extras_val: Any) -> List[int]:
        """Decodifica os extras_ids separados por ponto-e-vírgula.

        Args:
            extras_val: valor bruto da coluna extras_ids (str, NaN ou None).

        Returns:
            Lista de ids inteiros; vazia se o valor for nulo ou vazio.
        """
        if extras_val is None or (isinstance(extras_val, float) and math.isnan(extras_val)):
            return []
        s = str(extras_val).strip()
        return [int(x) for x in s.split(";") if x.strip()] if s else []

    def is_correct_branch(self, pred_id: int, true_central: Any, true_extras: Any) -> bool:
        """Avalia se a predição conta como correta pela régua oficial do case.

        Args:
            pred_id: id do tema predito.
            true_central: id do tema central verdadeiro (pode ser NaN).
            true_extras: extras_ids verdadeiros (formato "a;b;c", NaN ou None).

        Returns:
            True se pred_id está no ramo do tema central ou de algum extra.
        """
        if pd.isna(true_central):
            return False
        c = int(true_central)
        if pred_id in self.get_ramo(c):
            return True
        for e in self.parse_extras(true_extras):
            if pred_id in self.get_ramo(e):
                return True
        return False
