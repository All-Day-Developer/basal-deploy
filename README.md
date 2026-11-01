# basal-images

Container images for the `Remek/basal-*` models, built from `images.yaml`.

Each image is published in two variants:

| Tag | Contents |
|-----|----------|
| `latest`, `sha-<commit>` | no weights; downloaded on first start into `/models/cache` |
| `weights`, `weights-sha-<commit>` | weights baked into `/models/model` |

```bash
docker run --gpus all -p 8000:8000 -v basal-cache:/models/cache ghcr.io/<owner>/basal-2.0-mini:latest
docker run --gpus all -p 8080:8080 ghcr.io/<owner>/basal-2.0-mini-gguf:weights
```

## Development

```bash
uv sync
uv run pytest
uv run basal-images matrix
```

## Local build

```bash
docker build --target llamacpp \
  --build-arg WEIGHTS=weights \
  --build-arg MODEL=Remek/basal-2.0-mini-GGUF \
  --build-arg GGUF_FILE=basal-2.0-mini-Q5_K_M.gguf \
  -t basal-2.0-mini-gguf:weights .
```

`--target` is the runtime (`vllm` or `llamacpp`), `WEIGHTS` is `slim` or `weights`.
Pass a Hugging Face token with `--secret id=hf_token,env=HF_TOKEN` for gated repos.
