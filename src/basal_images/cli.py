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
from dataclasses import dataclass
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

    images = []
    for image in cfg.images:
        if only is not None:
            if image.name not in only:
                continue
        elif not image.enabled:
            continue
        variants = [v for v in image.variants if args.variant in ("all", v)]
        if not variants:
            continue
        images.append(
            {
                "name": image.name,
                "variants": ",".join(variants),
                **{v: v in variants for v in config.VARIANTS},
                "model": image.model,
                "revision": image.revision,
                "mode": image.mode,
                "gguf_file": image.gguf_file or "",
                "context": image.context or "",
                "serve_args": shlex.join(image.serve_args),
                "runtime": image.runtime.name,
                "runner_target": f"{image.runtime.extra}-runner",
                "target": image.runtime.extra,
                "base_image": image.runtime.base_image,
                "platforms": ",".join(image.platforms),
                "runner": json.dumps(image.runner),
            }
        )

    used = {row["runtime"] for row in images}
    runners = [
        {
            "name": rt.image,
            "runtime": rt.name,
            "target": f"{rt.extra}-runner",
            "base_image": rt.base_image,
            "platforms": ",".join(
                sorted({p for i in cfg.images if i.runtime is rt for p in i.platforms})
                or ["linux/amd64"]
            ),
        }
        for rt in cfg.runtimes.values()
        if only is None or rt.name in used
    ]

    out = {
        "registry": cfg.registry,
        "namespace": cfg.namespace,
        "runners": {"include": runners},
        "images": {"include": images},
    }
    print(json.dumps(out, separators=(",", ":")) if args.compact else json.dumps(out, indent=2))
    return 0


def fetch(model: str, revision: str, gguf_file: str, dest: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    folder = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_dir=dest,
            ignore_patterns=["*.gguf"] if gguf_file else None,
            token=token,
        )
    )
    if gguf_file:
        hf_hub_download(
            repo_id=model,
            filename=gguf_file,
            revision=revision,
            local_dir=dest,
            token=token,
        )
    if dest:
        shutil.rmtree(folder / ".cache", ignore_errors=True)
    return folder


def cmd_download(args: argparse.Namespace) -> int:
    folder = fetch(args.model, args.revision, args.gguf_file, args.dest)
    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        print(f"nothing downloaded from {args.model}", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in files) / 1e9
    print(f"downloaded {len(files)} files, {size:.2f} GB -> {folder}")
    return 0


def baked() -> bool:
    return MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())


def _bin(name: str) -> str:
    local = Path(sys.executable).parent / name
    return str(local) if local.exists() else (shutil.which(name) or name)


@dataclass
class Plan:
    serve: list[str]
    llama: list[str] | None = None
    llama_url: str | None = None


def plan(extra: list[str]) -> Plan:
    env = os.environ
    model = env.get("BASAL_MODEL", "")
    mode = env.get("BASAL_MODE", "")
    port = env.get("PORT") or env.get("BASAL_PORT", "8000")
    gguf_file = env.get("BASAL_GGUF_FILE", "")
    serve_args = shlex.split(env.get("BASAL_SERVE_ARGS", "")) + extra

    if not mode:
        raise SystemExit("set BASAL_MODE (fast, fp8, eager, llamacpp, ...)")
    if not model and not baked():
        raise SystemExit(f"set BASAL_MODEL or mount a model folder at {MODEL_DIR}")

    source = str(MODEL_DIR) if baked() else model
    serve = [
        _bin("basal-serve"),
        "--model", source,
        "--mode", mode,
        "--host", "0.0.0.0",
        "--port", port,
    ]  # fmt: skip

    if mode != "llamacpp":
        return Plan(serve=serve + serve_args)

    if not gguf_file:
        raise SystemExit("BASAL_GGUF_FILE is required for mode llamacpp")
    folder = MODEL_DIR if baked() else fetch(model, env.get("BASAL_REVISION", "main"), gguf_file)
    gguf = folder / gguf_file
    if not gguf.is_file():
        raise SystemExit(f"{gguf} not found")

    llama_port = env.get("LLAMA_PORT", "8081")
    llama_url = f"http://127.0.0.1:{llama_port}"
    llama = [
        env.get("LLAMA_SERVER", "/app/llama-server"),
        "-m", str(gguf),
        "-c", env.get("BASAL_CONTEXT") or "8192",
        "-ngl", env.get("N_GPU_LAYERS", "999"),
        "--host", "127.0.0.1",
        "--port", llama_port,
        *shlex.split(env.get("LLAMA_ARGS", "")),
    ]  # fmt: skip
    serve[serve.index("--model") + 1] = str(folder)
    return Plan(
        serve=serve + ["--llamacpp-url", llama_url] + serve_args, llama=llama, llama_url=llama_url
    )


def _wait_healthy(proc: subprocess.Popen, url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def _supervise(p: Plan) -> int:
    assert p.llama and p.llama_url
    print("llama-server:", shlex.join(p.llama), flush=True)
    llama = subprocess.Popen(p.llama)
    procs = [llama]

    def stop(signum: int, _frame: object) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    if not _wait_healthy(
        llama, f"{p.llama_url}/health", float(os.environ.get("LLAMA_TIMEOUT", 900))
    ):
        print("llama-server did not become healthy", file=sys.stderr)
        stop(signal.SIGTERM, None)
        llama.wait()
        return 1

    print("basal-serve:", shlex.join(p.serve), flush=True)
    procs.append(subprocess.Popen(p.serve))
    while all(proc.poll() is None for proc in procs):
        time.sleep(1)
    code = next(proc.returncode for proc in procs if proc.poll() is not None)
    stop(signal.SIGTERM, None)
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return code


def cmd_serve(extra: list[str]) -> int:
    if baked():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    p = plan(extra)
    if p.llama is None:
        argv = [sys.executable, "-m", "basal_images.launch", *p.serve[1:]]
        print("exec:", shlex.join(argv), flush=True)
        os.execv(argv[0], argv)
        return 0
    return _supervise(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basal-images")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("matrix", help="emit the CI build matrices as JSON")
    p.add_argument("--config", default="images.yaml")
    p.add_argument("--only", help="comma-separated image names (overrides `enabled`)")
    p.add_argument("--variant", choices=["all", *config.VARIANTS], default="all")
    p.add_argument("--compact", action="store_true", help="single-line JSON")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("download", help="download a model repository")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--gguf-file", default="", help="for -GGUF repos: the only .gguf to fetch")
    p.add_argument("--dest", default=str(MODEL_DIR))
    p.set_defaults(func=cmd_download)

    sub.add_parser("serve", help="start basal-serve (reads BASAL_* env), args go to basal-serve")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"] and argv[1:2] not in (["-h"], ["--help"]):
        return cmd_serve(argv[1:])
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
