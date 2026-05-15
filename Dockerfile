ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV FINWAVE_LISTEN_HOST=0.0.0.0 \
    FINWAVE_LISTEN_PORT=5003 \
    FINWAVE_MODEL_STORE_PATH=/var/lib/finwave/models

EXPOSE 5003

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:${FINWAVE_LISTEN_PORT}/health').raise_for_status()" || exit 1

CMD ["sh", "-c", "exec uvicorn finwave_inference_server.main:app --host \"$FINWAVE_LISTEN_HOST\" --port \"$FINWAVE_LISTEN_PORT\""]
