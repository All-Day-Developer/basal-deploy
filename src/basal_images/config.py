from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VARIANTS = ("slim", "weights")

Runner = str | list[str]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Runtime:
    name: str
    base_image: str
    port: int
    precompile: bool = False


@dataclass(frozen=True)
class Precompile:
    runner: Runner
    timeout_minutes: int = 60


@dataclass(frozen=True)
class Image:
    name: str
    model: str
    runtime: Runtime
    revision: str = "main"
    gguf_file: str | None = None
    extra_args: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    platforms: list[str] = field(default_factory=lambda: ["linux/amd64"])
    runner: Runner = "ubuntu-latest"
    precompile: bool | None = None
    enabled: bool = True

    @property
    def should_precompile(self) -> bool:
        return self.runtime.precompile if self.precompile is None else self.precompile


@dataclass(frozen=True)
class Config:
    registry: str
    namespace: str
    images: list[Image]
    precompile: Precompile | None = None

    def get(self, name: str) -> Image:
        for image in self.images:
            if image.name == name:
                return image
        raise ConfigError(f"no image named {name!r}")


def load(path: str | Path = "images.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    runtimes = {
        key: Runtime(
            name=key,
            base_image=val["base_image"],
            port=int(val["port"]),
            precompile=bool(val.get("precompile", False)),
        )
        for key, val in (raw.get("runtimes") or {}).items()
    }
    defaults: dict[str, Any] = raw.get("defaults") or {}

    precompile = None
    if raw.get("precompile"):
        precompile = Precompile(**raw["precompile"])

    images: list[Image] = []
    seen: set[str] = set()
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

        bad = set(merged.get("variants", VARIANTS)) - set(VARIANTS)
        if bad:
            raise ConfigError(f"{name}: unknown variants {sorted(bad)}")
        if runtime.name == "llamacpp" and not merged.get("gguf_file"):
            raise ConfigError(f"{name}: llamacpp images need a `gguf_file`")

        try:
            image = Image(runtime=runtime, **merged)
        except TypeError as exc:
            raise ConfigError(f"{name}: {exc}") from exc
        if image.should_precompile and runtime.name != "vllm":
            raise ConfigError(f"{name}: precompile is only supported for vllm")
        if image.enabled and image.should_precompile and precompile is None:
            raise ConfigError(f"{name}: precompile requested but no `precompile.runner` set")
        images.append(image)

    return Config(
        registry=raw.get("registry", "ghcr.io"),
        namespace=raw.get("namespace") or "",
        images=images,
        precompile=precompile,
    )
