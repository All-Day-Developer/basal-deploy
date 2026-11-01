# syntax=docker/dockerfile:1.10
ARG UV_VERSION=0.9.28
ARG PYTHON_VERSION=3.12
ARG VLLM_IMAGE=vllm/vllm-openai:v0.11.0
ARG LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
ARG WEIGHTS=slim

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM scratch AS compile-cache

FROM debian:bookworm-slim AS builder
ARG PYTHON_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/basal/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project --python ${PYTHON_VERSION}
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --python ${PYTHON_VERSION}

FROM builder AS models-slim
RUN mkdir -p /models/model

FROM builder AS models-weights
ARG MODEL
ARG REVISION=main
ARG GGUF_FILE=""
RUN test -n "${MODEL}" || (echo "MODEL build-arg is required" >&2; exit 1)
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    /opt/basal/venv/bin/basal-images download \
      --model "${MODEL}" --revision "${REVISION}" --gguf-file "${GGUF_FILE}" --dest /models/model

FROM models-${WEIGHTS} AS models

FROM ${VLLM_IMAGE} AS vllm
ARG MODEL
ARG GGUF_FILE=""
ARG EXTRA_ARGS=""
ARG PORT=8000
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/basal/venv /opt/basal/venv
COPY --from=models /models /models
ENV BASAL_RUNTIME=vllm \
    BASAL_MODEL="${MODEL}" \
    BASAL_GGUF_FILE="${GGUF_FILE}" \
    BASAL_EXTRA_ARGS="${EXTRA_ARGS}" \
    BASAL_PORT="${PORT}" \
    HF_HOME=/models/cache/hf \
    VLLM_CACHE_ROOT=/opt/vllm-cache \
    TRITON_CACHE_DIR=/opt/vllm-cache/triton
EXPOSE ${PORT}
ENTRYPOINT ["/opt/basal/venv/bin/basal-images", "serve"]
CMD []

FROM vllm AS vllm-compiled
COPY --from=compile-cache / /opt/vllm-cache/

FROM ${LLAMACPP_IMAGE} AS llamacpp
ARG MODEL
ARG GGUF_FILE=""
ARG EXTRA_ARGS=""
ARG PORT=8080
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/basal/venv /opt/basal/venv
COPY --from=models /models /models
ENV BASAL_RUNTIME=llamacpp \
    BASAL_MODEL="${MODEL}" \
    BASAL_GGUF_FILE="${GGUF_FILE}" \
    BASAL_EXTRA_ARGS="${EXTRA_ARGS}" \
    BASAL_PORT="${PORT}" \
    LLAMA_SERVER=/app/llama-server \
    LLAMA_CACHE=/models/cache/llama.cpp
EXPOSE ${PORT}
ENTRYPOINT ["/opt/basal/venv/bin/basal-images", "serve"]
CMD []
