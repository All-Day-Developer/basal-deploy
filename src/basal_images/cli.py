from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config

MODEL_DIR = Path("/models/model")


def cmd_matrix(args: argparse.Namespace) -> int:
    cfg = config.load(args.config)
    only = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None
    unknown = (only or set()) - {i.name for i in cfg.images}
    if unknown:
        print(f"unknown image(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    include = []
    for image in cfg.images:
        if only is not None:
            if image.name not in only:
                continue
        elif not image.enabled:
            continue
        precompile = image.should_precompile and not args.no_precompile
        if precompile and cfg.precompile is None:
            print(f"{image.name}: precompile requires `precompile.runner`", file=sys.stderr)
            return 2
        runner = cfg.precompile.runner if precompile else image.runner
        timeout = cfg.precompile.timeout_minutes if precompile else 60
        for variant in image.variants:
            if args.variant != "all" and variant != args.variant:
                continue
            include.append(
                {
                    "name": image.name,
                    "variant": variant,
                    "model": image.model,
                    "revision": image.revision,
                    "gguf_file": image.gguf_file or "",
                    "runtime": image.runtime.name,
                    "base_image": image.runtime.base_image,
                    "port": image.runtime.port,
                    "extra_args": shlex.join(image.extra_args),
                    "platforms": ",".join(image.platforms),
                    "precompile": precompile,
                    "precompile_timeout": timeout,
                    "runner": json.dumps(runner),
                }
            )

    out = {
        "registry": cfg.registry,
        "namespace": cfg.namespace,
        "matrix": {"include": include},
    }
    print(json.dumps(out, separators=(",", ":")) if args.compact else json.dumps(out, indent=2))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    allow = [args.gguf_file] if args.gguf_file else None

    dest = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=args.dest,
            allow_patterns=allow,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)

    files = [p for p in dest.rglob("*") if p.is_file()]
    if not files:
        print(f"nothing downloaded from {args.model} ({args.gguf_file})", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in files) / 1e9
    print(f"downloaded {len(files)} files, {size:.2f} GB -> {dest}")
    return 0


def _baked() -> bool:
    return MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())


def _vllm_cmd(model: str, port: str, extra: list[str]) -> list[str]:
    target = str(MODEL_DIR) if _baked() else model
    return [
        "vllm", "serve", target,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", port,
        *extra,
    ]  # fmt: skip


def _llamacpp_cmd(model: str, gguf_file: str, port: str, extra: list[str]) -> list[str]:
    binary = os.environ.get("LLAMA_SERVER", "/app/llama-server")
    if _baked():
        path = MODEL_DIR / gguf_file
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        source = ["-m", str(path)]
    else:
        source = ["--hf-repo", model, "--hf-file", gguf_file]
    return [
        binary, *source,
        "--alias", model,
        "--host", "0.0.0.0",
        "--port", port,
        "-ngl", os.environ.get("N_GPU_LAYERS", "999"),
        *extra,
    ]  # fmt: skip


def _server_argv(extra: list[str]) -> tuple[list[str], str, str]:
    env = os.environ
    runtime = env["BASAL_RUNTIME"]
    model = env["BASAL_MODEL"]
    port = env.get("PORT") or env["BASAL_PORT"]
    extra = shlex.split(env.get("BASAL_EXTRA_ARGS", "")) + extra

    if runtime == "vllm":
        return _vllm_cmd(model, port, extra), model, port
    if runtime == "llamacpp":
        return _llamacpp_cmd(model, env["BASAL_GGUF_FILE"], port, extra), model, port
    raise SystemExit(f"unknown BASAL_RUNTIME {runtime!r}")


def cmd_serve(args: argparse.Namespace) -> int:
    argv, _, _ = _server_argv(args.extra)
    print("exec:", shlex.join(argv), flush=True)
    os.execvp(argv[0], argv)
    return 0


def _wait_healthy(proc: subprocess.Popen, url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    return False


def _complete(base: str, model: str, max_tokens: int) -> None:
    body = json.dumps({"model": model, "prompt": "Hello", "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()


def cmd_warmup(args: argparse.Namespace) -> int:
    argv, model, port = _server_argv(args.extra)
    base = f"http://127.0.0.1:{port}"
    print("warmup:", shlex.join(argv), flush=True)
    proc = subprocess.Popen(argv)
    try:
        if not _wait_healthy(proc, f"{base}/health", time.monotonic() + args.timeout):
            print("server did not become healthy", file=sys.stderr)
            return 1
        for _ in range(args.requests):
            _complete(base, model, args.max_tokens)
        print("warmup requests done", flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    cache = Path(os.environ.get("VLLM_CACHE_ROOT", "/opt/vllm-cache"))
    files = [p for p in cache.rglob("*") if p.is_file()] if cache.is_dir() else []
    if not files:
        print(f"no compile artifacts under {cache}", file=sys.stderr)
        return 1
    print(f"compile cache: {len(files)} files under {cache}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basal-images")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("matrix", help="emit the CI build matrix as JSON")
    p.add_argument("--config", default="images.yaml")
    p.add_argument("--only", help="comma-separated image names (overrides `enabled`)")
    p.add_argument("--variant", choices=["all", *config.VARIANTS], default="all")
    p.add_argument("--no-precompile", action="store_true", help="skip GPU precompilation")
    p.add_argument("--compact", action="store_true", help="single-line JSON")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("download", help="download model weights")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--gguf-file", default="", help="download only this GGUF file")
    p.add_argument("--dest", default=str(MODEL_DIR))
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("serve", help="exec the inference server (reads BASAL_* env)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for the server")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("warmup", help="start the server once to populate the compile cache")
    p.add_argument("--timeout", type=int, default=3600, help="seconds to wait for /health")
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for the server")
    p.set_defaults(func=cmd_warmup)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
