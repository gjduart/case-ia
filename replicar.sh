#!/usr/bin/env bash
# ==============================================================================
# Script de Replicação Oficial - Case IA Classificação de Tópicos (Final Ótimo)
# Executa o pipeline sem LLM e sem rede (embeddings e sintéticos lidos do cache) e
# gera publicacao.csv e previsao.txt (configuração entregue: A1).
# ==============================================================================
set -euo pipefail

DIR_ATUAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR_ATUAL"

echo "======================================================================"
echo "    MÉTODO VDE - CASE DE IA: CLASSIFICAÇÃO DE TÓPICOS JURÍDICOS"
echo "    Entrega: A1 = roteador por area (qwen-LR de raiz) + S4 em Penal/Processo + consenso BM25=qwen-LR em Leg.Esp./Crim. (sem LLM, sem rede)"
echo "======================================================================"

# Verifica ambiente virtual ou python do sistema
PYTHON_BIN="python3"
if [ -d "../venv/bin" ]; then
    PYTHON_BIN="../venv/bin/python"
elif [ -d "venv/bin" ]; then
    PYTHON_BIN="venv/bin/python"
fi

echo "[1/3] Usando interpretador: $($PYTHON_BIN --version) em $PYTHON_BIN"

echo "[2/3] Extraindo normas citadas (regex, cache/leis_por_questao.csv)..."
$PYTHON_BIN -B scripts/extrair_leis.py

echo "[3/3] Executando o pipeline completo (OOF 5x5 + produção + calibração; lê cache/embeddings; offline, ~5 min em CPU)..."
$PYTHON_BIN -B scripts/gerar_publicacao.py \
    --saida publicacao.csv \
    --previsao previsao.txt

echo "Validação dos artefatos gerados:"
if [ -f "publicacao.csv" ] && [ -f "previsao.txt" ]; then
    echo "  -> publicacao.csv gerado com sucesso: $(wc -l < publicacao.csv) linhas"
    echo "  -> previsao.txt gerado com sucesso:"
    head -n 6 previsao.txt
    echo "======================================================================"
    echo "Sucesso! Replicação concluída em conformidade com as regras do case."
    echo "======================================================================"
else
    echo "Erro: falha na geração dos arquivos."
    exit 1
fi
