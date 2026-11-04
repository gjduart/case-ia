# Testado nesta maquina (Docker 29.9.2): o alvo base reproduz publicacao.csv/previsao.txt com o mesmo md5. Python = o do venv que gerou os resultados (3.14).
FROM python:3.14-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x replicar.sh
# padrao: replica publicacao.csv e previsao.txt (offline, sem modelo de linguagem)
CMD ["./replicar.sh"]

FROM base AS api
COPY requirements-api.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-api.txt
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
