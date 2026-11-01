from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VARIANTS = ("online", "offline")
MODES = ("fast", "fast-nocompile", "fp8", "eager", "llamacpp", "mlx")

Runner = str | list[str]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Runtime:
    name: str
    image: str
    base_image: str
    extra: str
    port: int


@dataclass(frozen=True)
class Image:
    name: str
    model: str
    runtime: Runtime
    mode: str
    revision: str = "main"
    gguf_file: str | None = None
    context: int | None = None
    serve_args: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    platforms: list[str] = field(default_factory=lambda: ["linux/amd64"])
    runner: Runner = "ubuntu-24.04"
    enabled: bool = True


@dataclass(frozen=True)
class Config:
    registry: str
    namespace: str
    runtimes: dict[str, Runtime]
    images: list[Image]

    def get(self, name: str) -> Image:
        for image in self.images:
            if image.name == name:
                return image
        raise ConfigError(f"no image named {name!r}")


def _runtime(key: str, val: dict[str, Any]) -> Runtime:
    try:
        return Runtime(
            name=key,
            image=val.get("image", f"basal-{key}"),
            base_image=val["base_image"],
            extra=val.get("extra", key),
            port=int(val["port"]),
        )
    except KeyError as exc:
        raise ConfigError(f"runtime {key}: missing {exc}") from exc


def load(path: str | Path = "images.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    runtimes = {k: _runtime(k, v) for k, v in (raw.get("runtimes") or {}).items()}
    defaults: dict[str, Any] = raw.get("defaults") or {}

    seen: set[str] = {r.image for r in runtimes.values()}
    if len(seen) != len(runtimes):
        raise ConfigError("runtime image names must be unique")

    images: list[Image] = []
    for entry in raw.get("images") or []:
        merged = {**defaults, **entry}
        name = merged.get("name")
        if not name or name != name.lower():
            raise ConfigError(f"image name must be set and lowercase: {name!r}")
        if name in seen:
            raise ConfigError(f"duplicate image name {name!r}")
        seen.add(name)

        runtime_key = merged.pop("runtime", None)
        if runtime_key not in runtimes:
            raise ConfigError(f"{name}: unknown runtime {runtime_key!r}")
        runtime = runtimes[runtime_key]

        mode = merged.get("mode")
        if mode not in MODES:
            raise ConfigError(f"{name}: mode must be one of {MODES}, got {mode!r}")
        bad = set(merged.get("variants", VARIANTS)) - set(VARIANTS)
        if bad:
            raise ConfigError(f"{name}: unknown variants {sorted(bad)}")
        if mode == "llamacpp":
            if runtime.extra != "llamacpp":
                raise ConfigError(f"{name}: mode llamacpp needs a llamacpp runtime")
            if not merged.get("gguf_file"):
                raise ConfigError(f"{name}: mode llamacpp needs a `gguf_file`")
        elif runtime.extra == "llamacpp":
            raise ConfigError(f"{name}: a llamacpp runtime only serves mode llamacpp")
        if mode == "mlx" and merged.get("enabled", True):
            raise ConfigError(f"{name}: mlx runs only on Apple Silicon, set enabled: false")

        merged["serve_args"] = [str(a) for a in merged.get("serve_args") or []]
        try:
            images.append(Image(runtime=runtime, **merged))
        except TypeError as exc:
            raise ConfigError(f"{name}: {exc}") from exc

    return Config(
        registry=raw.get("registry", "ghcr.io"),
        namespace=raw.get("namespace") or "",
        runtimes=runtimes,
        images=images,
    )
