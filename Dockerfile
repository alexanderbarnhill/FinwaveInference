ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm

WORKDIR /app

# Pillow's manylinux wheel is self-contained; onnxruntime CPU wheel ships its own
# libs. If we later add a runtime dep needing system libs (e.g. opencv-python),
# add the matching apt install here.

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
