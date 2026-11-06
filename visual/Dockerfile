FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependências primeiro (melhor uso de cache). scikit-learn fixado em 1.8.0 por causa do joblib.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Falha o build se os modelos não reproduzirem as métricas do teste original.
RUN python verificar_modelos.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# Padrão: app Streamlit. Para o pipeline em lote: docker run ... python pipeline.py <comando>
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
