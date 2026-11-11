# FinRag API image. Python 3.13 matches the local .venv the tests ran on (3.13.3).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FASTEMBED_CACHE_PATH=/opt/fastembed
WORKDIR /app

RUN useradd --create-home --uid 1000 finrag

COPY requirements.txt .
RUN pip install -r requirements.txt

# Embedding model (mxbai-embed-large-v1, about 0.64 GB) baked in so the container starts offline.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1')" \
    && chmod -R a+rX /opt/fastembed

# Prebuilt index v2 (built offline by src/jobs/index_corpus.py; never rebuilt in the image).
COPY --chown=finrag:finrag .cache/index/v2 .cache/index/v2
COPY --chown=finrag:finrag data/eval/docs.json data/eval/docs.json

COPY --chown=finrag:finrag config.yaml main.py ./
COPY --chown=finrag:finrag src src
RUN mkdir -p logs .cache/llm && chown -R finrag:finrag /app

USER finrag
EXPOSE 8000
# 0.0.0.0 here, not in config.yaml (api.host stays 127.0.0.1 for local runs)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
