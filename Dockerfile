# syntax=docker/dockerfile:1.10
ARG UV_VERSION=0.9.28
ARG PYTHON_VERSION=3.12
ARG ENGINE_IMAGE=debian:bookworm-slim
ARG LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

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
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

FROM builder AS venv-tools
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM builder AS venv-engine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra engine

FROM builder AS venv-llamacpp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra llamacpp

FROM venv-tools AS models
ARG MODEL
ARG REVISION=main
ARG GGUF_FILE=""
RUN test -n "${MODEL}" || (echo "MODEL build-arg is required" >&2; exit 1)
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    /opt/basal/venv/bin/basal-images download \
      --model "${MODEL}" --revision "${REVISION}" --gguf-file "${GGUF_FILE}" --dest /models/model

FROM ${ENGINE_IMAGE} AS engine-runner
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=venv-engine /opt/python /opt/python
COPY --from=venv-engine /opt/basal/venv /opt/basal/venv
RUN mkdir -p /models/model /models/cache /opt/compile-cache
ENV PATH=/opt/basal/venv/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    BASAL_PORT=8000 \
    HF_HOME=/models/cache/hf \
    TORCHINDUCTOR_CACHE_DIR=/opt/compile-cache/inductor \
    TRITON_CACHE_DIR=/opt/compile-cache/triton
EXPOSE 8000
ENTRYPOINT ["/opt/basal/venv/bin/basal-images", "serve"]
CMD []

FROM engine-runner AS engine
ARG MODEL
ARG MODE
ARG REVISION=main
ARG SERVE_ARGS=""
ENV BASAL_MODEL="${MODEL}" \
    BASAL_MODE="${MODE}" \
    BASAL_REVISION="${REVISION}" \
    BASAL_SERVE_ARGS="${SERVE_ARGS}"

FROM engine AS engine-offline
COPY --from=models /models/model /models/model

FROM ${LLAMACPP_IMAGE} AS llamacpp-runner
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=venv-llamacpp /opt/python /opt/python
COPY --from=venv-llamacpp /opt/basal/venv /opt/basal/venv
RUN mkdir -p /models/model /models/cache
ENV PATH=/opt/basal/venv/bin:${PATH} \
    BASAL_MODE=llamacpp \
    BASAL_PORT=8000 \
    HF_HOME=/models/cache/hf \
    LLAMA_SERVER=/app/llama-server \
    LLAMA_PORT=8081
EXPOSE 8000
ENTRYPOINT ["/opt/basal/venv/bin/basal-images", "serve"]
CMD []

FROM llamacpp-runner AS llamacpp
ARG MODEL
ARG REVISION=main
ARG GGUF_FILE
ARG CONTEXT=8192
ARG SERVE_ARGS=""
ENV BASAL_MODEL="${MODEL}" \
    BASAL_REVISION="${REVISION}" \
    BASAL_GGUF_FILE="${GGUF_FILE}" \
    BASAL_CONTEXT="${CONTEXT}" \
    BASAL_SERVE_ARGS="${SERVE_ARGS}"

FROM llamacpp AS llamacpp-offline
COPY --from=models /models/model /models/model
