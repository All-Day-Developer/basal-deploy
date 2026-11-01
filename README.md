# basal-images

Container images that serve the `Remek/basal-*` typed-decision models with
[`basal-serve`](https://github.com/rkinas/basal), built from `images.yaml`.

basal models are not chat models: they read a `state`, answer typed `questions` and return a calibrated probability
for every allowed answer, in one forward pass. The API is `POST /v1/systemone`.

## Images

| Image | Tags | Contents |
|-------|------|----------|
| `basal-engine` | `latest`, `sha-<commit>` | basal engine (PyTorch, CUDA 12.8): modes `fast`, `fp8`, `eager`; bring your own model |
| `basal-engine-llamacpp` | `latest`, `sha-<commit>` | `llama-server` + `basal-serve --mode llamacpp`; bring your own GGUF |
| `<model>` | `latest`, `online`, `online-sha-<commit>` | model preset; weights downloaded on first start |
| `<model>` | `offline`, `offline-sha-<commit>` | weights baked in, runs with no network |

| Model image | Mode |
|-------------|------|
| `basal-2.0-{mini,4.5b,max}` | `fast` (bf16, sm80+) |
| `basal-2.0-{mini,4.5b,max}-fp8` | `fp8` (sm89+: RTX 40xx, Hopper, Blackwell) |
| `basal-2.0-{mini,4.5b,max}-gguf` | `llamacpp` (Q8_0) |
| `basal-omni-1.1-max` | `eager --media on` |

## Run

```bash
docker run --gpus all -p 8000:8000 \
  -v basal-compile:/opt/compile-cache \
  ghcr.io/all-day-developer/basal-2.0-max-fp8:offline
```

The first start of `fast` / `fp8` compiles the model and captures CUDA graphs (several minutes); the
`basal-compile` volume keeps the compile cache for later starts. Online images also want
`-v basal-hf:/models/cache` so weights are downloaded once.

Arguments after the image name go to `basal-serve`:

```bash
docker run --gpus all -p 8000:8000 ghcr.io/all-day-developer/basal-2.0-mini:offline --deterministic
```

Runner with any model:

```bash
docker run --gpus all -p 8000:8000 -e BASAL_MODEL=Remek/basal-2.0-4.5B -e BASAL_MODE=fast \
  -e BASAL_SERVE_ARGS="--template-prefix off" \
  -v basal-hf:/models/cache -v basal-compile:/opt/compile-cache \
  ghcr.io/all-day-developer/basal-engine:latest

docker run --gpus all -p 8000:8000 -e BASAL_MODEL=Remek/basal-2.0-max-GGUF \
  -e BASAL_GGUF_FILE=basal-2.0-max-Q5_K_M.gguf -e BASAL_CONTEXT=32768 \
  -v basal-hf:/models/cache \
  ghcr.io/all-day-developer/basal-engine-llamacpp:latest
```

A local model folder can be mounted read-only at `/models/model` instead of setting `BASAL_MODEL`.

## Ask

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

## Environment

| Variable | Meaning |
|----------|---------|
| `BASAL_MODEL` | Hugging Face repo id (ignored when `/models/model` has files) |
| `BASAL_MODE` | `fast`, `fast-nocompile`, `fp8`, `eager`, `llamacpp` |
| `BASAL_SERVE_ARGS` | extra `basal-serve` flags |
| `BASAL_GGUF_FILE`, `BASAL_CONTEXT` | llama.cpp images: GGUF file and `llama-server -c` |
| `N_GPU_LAYERS`, `LLAMA_ARGS` | llama.cpp images: `-ngl` and extra `llama-server` flags |
| `PORT` | listen port (default 8000) |
| `HF_TOKEN` | for gated repos |

## Development

```bash
uv sync
uv run pytest
uv run basal-images matrix
```

The engine dependencies are the `engine` and `llamacpp` extras of `pyproject.toml`, locked in `uv.lock`
(torch from the CUDA 12.8 index for `engine`, the CPU index for `llamacpp`).

## Dockerfile targets

| Target | Purpose |
|--------|---------|
| `engine-runner`, `llamacpp-runner` | runner images |
| `engine`, `llamacpp` | model images: `VARIANT=online\|offline`, `MODEL`, `MODE`, `GGUF_FILE`, `CONTEXT`, `SERVE_ARGS` |

## CI secrets

| Name | Required |
|------|----------|
| `HF_TOKEN` | only for gated or private model repos |
