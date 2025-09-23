# syntax=docker/dockerfile:1.6
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r requirements.txt

ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
        /opt/venv/bin/python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"; \
    fi

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/home/app/.cache/huggingface" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libopenblas0-pthread \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY streamlit_app.py ./
COPY scripts ./scripts
COPY static ./static
COPY README.md ./
COPY requirements.txt ./

RUN mkdir -p /app/logs /app/qdrant_data "$HF_HOME" && \
    chown -R app:app /app /home/app

USER app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
