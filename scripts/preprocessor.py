"""Pré-processamento e limpeza textual das questões.

Parte do pipeline de ./replicar.sh (importado por dados, embeddings
e extrair_leis). Normaliza caracteres, remove HTML/comentários do MS Word,
decodifica entidades e padroniza enunciados, comandos e alternativas.

Entradas: textos brutos em memória. Saídas: textos limpos (sem arquivos).
"""
from __future__ import annotations

import re
import unicodedata
from html import unescape
from typing import Any, Dict


class TextCleaner:
    """Higieniza e normaliza textos jurídicos mantendo a integridade léxica.

    Attributes:
        _RE_*: expressões regulares compiladas para tags, comentários, estilos,
            espaços, quebras e caracteres não latinos.
        _HOMOGLIFOS: tabela de tradução de letras cirílicas/gregas para latinas.
    """

    _RE_QUEBRA = re.compile(r"<\s*(br|/p|p|/div|div|/li|li)\b[^>]*>", re.I)
    _RE_IMG = re.compile(r"<\s*img\b[^>]*>", re.I)
    _RE_TAG = re.compile(r"<[^>]+>")
    _RE_ESPACOS = re.compile(r"[ \t ]+")
    _RE_LINHAS = re.compile(r"\n{3,}")
    _RE_LITERAL = re.compile(r"\\n")
    # Normalizacao motivada pela auditoria dos dados (Q0099: HTML do MS Word com comentarios condicionais e <style>;
    # Q0490, Q0544: letras cirilicas/gregas no lugar das latinas; temas.csv: palavra em arabe na descricao).
    _RE_MSO = re.compile(r"<!--\[if.*?<!\[endif\]-->", re.S | re.I)
    _RE_COMENT = re.compile(r"<!--.*?-->", re.S)
    _RE_STYLE = re.compile(r"<style\b.*?</style>", re.S | re.I)
    _HOMOGLIFOS = str.maketrans("аеорсухіАВЕКМНОРСТХІοΟρΡαΑνΚτΤ", "aeopcyxiABEKMHOPCTXIoOpPaAvKtT")
    _RE_NAO_LATINO = re.compile(r"[֐-ࣿЀ-ӿͰ-Ͽ]+")

    @classmethod
    def clean(cls, text: Any) -> str:
        """Limpa marcações HTML e padroniza quebras de linha.

        Args:
            text: texto bruto (qualquer tipo; não-string ou vazio gera "").

        Returns:
            Texto sem HTML, com homóglifos corrigidos e espaços normalizados.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        s = cls._RE_MSO.sub(" ", text)
        s = cls._RE_STYLE.sub(" ", s)
        s = cls._RE_COMENT.sub(" ", s)
        s = cls._RE_LITERAL.sub("\n", s)
        s = cls._RE_QUEBRA.sub("\n", s)
        s = cls._RE_IMG.sub(" [imagem] ", s)
        s = cls._RE_TAG.sub(" ", s)
        s = unescape(s)
        s = unicodedata.normalize("NFC", s).translate(cls._HOMOGLIFOS)
        s = cls._RE_NAO_LATINO.sub(" ", s)
        s = cls._RE_ESPACOS.sub(" ", s)
        s = cls._RE_LINHAS.sub("\n\n", s)
        return s.strip()

    @classmethod
    def to_single_line(cls, text: Any) -> str:
        """Converte texto higienizado em uma linha contínua.

        Args:
            text: texto bruto.

        Returns:
            Texto limpo com todo espaço em branco colapsado em um espaço.
        """
        cleaned = cls.clean(text)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def extract_question_text(
        cls,
        enunciado: Any,
        comando: Any,
        alternativas: Dict[str, str] | None = None,
        include_alternativas: bool = True
    ) -> str:
        """Monta o documento textual completo da questão.

        Args:
            enunciado: enunciado bruto.
            comando: comando da questão (prefixado com "[COMANDO]").
            alternativas: dicionário letra -> texto; opcional.
            include_alternativas: se True, anexa "[ALTERNATIVAS]" (até 3000 caracteres).

        Returns:
            Texto único em uma linha: enunciado + comando + alternativas.
        """
        en_str = cls.to_single_line(enunciado)
        cm_str = cls.to_single_line(comando)
        
        parts = [en_str]
        if cm_str:
            parts.append(f"[COMANDO] {cm_str}")
            
        if include_alternativas and isinstance(alternativas, dict):
            alt_texts = [cls.to_single_line(v) for v in alternativas.values() if v]
            if alt_texts:
                parts.append("[ALTERNATIVAS] " + " ".join(alt_texts)[:3000])
                
        return " ".join(parts).strip()
