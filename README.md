# basal-deploy

Docker images for the [basal](https://github.com/rkinas/basal) decision models (`Remek/basal-*` on Hugging Face).

basal is not a chat model. You send it a piece of text (`state`) and a few questions with fixed answers, and it
returns a probability for each answer. The server is `basal-serve` and the endpoint is `POST /v1/systemone`.
OpenAI-style `/v1/chat/completions` does not exist here, and running these weights in vLLM or plain llama.cpp gives
nonsense.

## Quick start

```bash
docker run --gpus all -p 8000:8000 -v basal-compile:/opt/compile-cache \
  ghcr.io/all-day-developer/basal-2.0-4.5b-fp8:offline
```

The first start compiles the model and takes a few minutes. The `basal-compile` volume keeps the result, so the next
start is quick. Then:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

## Which image

Every model has two tags:

- `offline` has the weights inside the image. Big, but starts without network access.
- `online` (also `latest`) downloads the weights on first start. Mount `-v basal-hf:/models/cache` so that happens
  only once.

Pick the model by GPU:

- `basal-2.0-{mini,4.5b,max}-fp8` for RTX 40xx, Hopper and Blackwell. Best choice if your card supports it.
- `basal-2.0-{mini,4.5b,max}` (bf16) for Ampere and newer.
- `basal-2.0-{mini,4.5b,max}-gguf` runs through llama.cpp (Q8_0). Slower, but uses less memory.
- `basal-omni-1.1-max` also accepts images and audio.

The `max` models need a lot of memory. On a 24 GB card the default CUDA graph sizes don't fit, so shrink them:

```bash
docker run --gpus all -p 8000:8000 -v basal-compile:/opt/compile-cache \
  -e BASAL_TOKEN_BUDGET=4096 -e BASAL_GRAPH_LEN=4096 -e BASAL_TREE_ROWS=2 \
  ghcr.io/all-day-developer/basal-2.0-max-fp8:offline
```

Prompts longer than `BASAL_GRAPH_LEN` still work, just a bit slower.

Anything after the image name is passed to `basal-serve`, for example `--deterministic`.

## Your own model

`basal-engine` and `basal-engine-llamacpp` are the same images without a model. Tell them what to load:

```bash
docker run --gpus all -p 8000:8000 \
  -e BASAL_MODEL=Remek/basal-2.0-4.5B -e BASAL_MODE=fast -e BASAL_SERVE_ARGS="--template-prefix off" \
  -v basal-hf:/models/cache -v basal-compile:/opt/compile-cache \
  ghcr.io/all-day-developer/basal-engine:latest

docker run --gpus all -p 8000:8000 \
  -e BASAL_MODEL=Remek/basal-2.0-max-GGUF -e BASAL_GGUF_FILE=basal-2.0-max-Q5_K_M.gguf -e BASAL_CONTEXT=32768 \
  -v basal-hf:/models/cache \
  ghcr.io/all-day-developer/basal-engine-llamacpp:latest
```

You can also mount a local model folder at `/models/model` instead of setting `BASAL_MODEL`.

Settings:

```
BASAL_MODEL         Hugging Face repo, ignored when /models/model is not empty
BASAL_MODE          fast, fast-nocompile, fp8, eager or llamacpp
BASAL_SERVE_ARGS    extra basal-serve flags
BASAL_TOKEN_BUDGET  fast/fp8: tokens per batch (default 12288)
BASAL_GRAPH_LEN     fast/fp8: longest prompt with a CUDA graph (default 8192)
BASAL_TREE_ROWS     fast/fp8: requests per second-pass graph (default 8)
BASAL_GGUF_FILE     llamacpp: which .gguf file to load
BASAL_CONTEXT       llamacpp: llama-server context size
N_GPU_LAYERS        llamacpp: layers on the GPU (default all)
LLAMA_ARGS          llamacpp: extra llama-server flags
PORT                listen port (default 8000)
HF_TOKEN            only for gated repos
```

## Adding a model

Add an entry to `images.yaml` and push to `master`. CI builds one job per image (online and offline in the same
job) and pushes to `ghcr.io/<owner>/<name>`. Run it by hand from the Actions tab to build only some images.

If a model repo is gated, add an `HF_TOKEN` secret to the repository.

## Working on this repo

```bash
uv sync
uv run pytest
uv run basal-images matrix
```

The Python dependencies of the images live in the `engine` and `llamacpp` extras in `pyproject.toml` and are pinned
in `uv.lock`. The `Dockerfile` targets are `engine-runner` and `llamacpp-runner` for the empty images, `engine` and `llamacpp`
for the online model images, and `engine-offline` and `llamacpp-offline`, which are the online image plus one layer
with the weights.
