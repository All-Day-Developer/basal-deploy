from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
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
        for variant in image.variants:
            if args.variant != "all" and variant != args.variant:
                continue
            images.append(
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
                    "runner": json.dumps(image.runner),
                }
            )

    used = {row["runtime"] for row in images}
    runners = [
        {
            "name": rt.image,
            "runtime": rt.name,
            "base_image": rt.base_image,
            "port": rt.port,
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
        "--served-model-name", model or "model",
        "--host", "0.0.0.0",
        "--port", port,
        *extra,
    ]  # fmt: skip


def _llamacpp_cmd(model: str, gguf_file: str, port: str, extra: list[str]) -> list[str]:
    binary = os.environ.get("LLAMA_SERVER", "/app/llama-server")
    if _baked():
        if gguf_file:
            path = MODEL_DIR / gguf_file
        else:
            ggufs = sorted(MODEL_DIR.glob("*.gguf"))
            if len(ggufs) != 1:
                raise SystemExit(f"set BASAL_GGUF_FILE; found {len(ggufs)} .gguf in {MODEL_DIR}")
            path = ggufs[0]
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        source = ["-m", str(path)]
    else:
        if not gguf_file:
            raise SystemExit("BASAL_GGUF_FILE is required")
        source = ["--hf-repo", model, "--hf-file", gguf_file]
    return [
        binary, *source,
        "--alias", model or "model",
        "--host", "0.0.0.0",
        "--port", port,
        "-ngl", os.environ.get("N_GPU_LAYERS", "999"),
        *extra,
    ]  # fmt: skip


def server_argv(extra: list[str]) -> list[str]:
    env = os.environ
    runtime = env["BASAL_RUNTIME"]
    model = env.get("BASAL_MODEL", "")
    port = env.get("PORT") or env["BASAL_PORT"]
    extra = shlex.split(env.get("BASAL_EXTRA_ARGS", "")) + extra

    if not model and not _baked():
        raise SystemExit(f"set BASAL_MODEL or mount weights at {MODEL_DIR}")
    if runtime == "vllm":
        return _vllm_cmd(model, port, extra)
    if runtime == "llamacpp":
        return _llamacpp_cmd(model, env.get("BASAL_GGUF_FILE", ""), port, extra)
    raise SystemExit(f"unknown BASAL_RUNTIME {runtime!r}")


def cmd_serve(args: argparse.Namespace) -> int:
    argv = server_argv(args.extra)
    if _baked():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("exec:", shlex.join(argv), flush=True)
    os.execvp(argv[0], argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basal-images")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("matrix", help="emit the CI build matrices as JSON")
    p.add_argument("--config", default="images.yaml")
    p.add_argument("--only", help="comma-separated image names (overrides `enabled`)")
    p.add_argument("--variant", choices=["all", *config.VARIANTS], default="all")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
