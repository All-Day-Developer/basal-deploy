# basal-images

Container images for the `Remek/basal-*` models, built from `images.yaml`.

## Images

| Image | Tags | Contents |
|-------|------|----------|
| `basal-vllm`, `basal-llamacpp` | `latest`, `sha-<commit>` | runner only; bring your own model |
| `<model>` | `latest`, `online`, `online-sha-<commit>` | model preset; weights downloaded on first start |
| `<model>` | `offline`, `offline-sha-<commit>` | weights baked in, runs with no network (`HF_HUB_OFFLINE=1`) |

## Run

Online, with persistent weights and compile cache:

```bash
docker run --gpus all -p 8000:8000 \
  -v basal-hf:/models/cache \
  -v basal-vllm-cache:/opt/vllm-cache \
  ghcr.io/all-day-developer/basal-2.0-mini:latest
```

The first start compiles the torch graphs into `basal-vllm-cache`; later starts on the same
GPU, vLLM version and server args reuse it.

Offline:

```bash
docker run --gpus all -p 8080:8080 ghcr.io/all-day-developer/basal-2.0-mini-gguf:offline
```

Runner with any model:

```bash
docker run --gpus all -p 8000:8000 -e BASAL_MODEL=Remek/basal-2.0-4.5B \
  -v basal-hf:/models/cache -v basal-vllm-cache:/opt/vllm-cache \
  ghcr.io/all-day-developer/basal-vllm:latest

docker run --gpus all -p 8080:8080 \
  -e BASAL_MODEL=Remek/basal-2.0-max-GGUF -e BASAL_GGUF_FILE=basal-2.0-max-Q8_0.gguf \
  -v basal-hf:/models/cache \
  ghcr.io/all-day-developer/basal-llamacpp:latest
```

Local weights can be mounted read-only at `/models/model` instead of setting `BASAL_MODEL`.
Extra server flags go after the image name or in `BASAL_EXTRA_ARGS`.

## Development

```bash
uv sync
uv run pytest
uv run basal-images matrix
```

## Dockerfile targets

| Target | Purpose |
|--------|---------|
| `vllm-runner`, `llamacpp-runner` | runner images |
| `vllm`, `llamacpp` | model images, `VARIANT=online\|offline`, `MODEL`, `GGUF_FILE`, `EXTRA_ARGS` |

Pass a Hugging Face token with `--secret id=hf_token,env=HF_TOKEN` for gated repos.

## CI secrets

| Name | Required |
|------|----------|
| `HF_TOKEN` | only for gated or private model repos |
