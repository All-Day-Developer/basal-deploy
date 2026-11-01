import json
from pathlib import Path

import pytest

from basal_images import cli, config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "images.yaml")


def _plan(capsys, *extra: str) -> dict:
    assert cli.main(["matrix", "--config", CONFIG, "--compact", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "images.yaml"
    path.write_text(body)
    return path


def test_repo_config_loads():
    cfg = config.load(CONFIG)
    assert cfg.images
    assert all(i.runtime.name in cfg.runtimes for i in cfg.images)


def test_modes_match_weights():
    cfg = config.load(CONFIG)
    for image in cfg.images:
        if image.model.endswith("-FP8"):
            assert image.mode == "fp8", image.name
        elif image.model.endswith("-GGUF"):
            assert image.mode == "llamacpp", image.name
        elif "MLX" in image.model:
            assert not image.enabled, image.name


def test_matrix_skips_disabled_and_expands_variants(capsys):
    images = _plan(capsys)["images"]["include"]
    assert not any("mlx" in row["name"] for row in images)
    variants = {row["variant"] for row in images if row["name"] == "basal-2.0-mini"}
    assert variants == {"online", "offline"}


def test_matrix_only_and_variant_filters(capsys):
    plan = _plan(capsys, "--only", "basal-2.0-mini-gguf", "--variant", "offline")
    images = plan["images"]["include"]
    assert [(r["name"], r["variant"]) for r in images] == [("basal-2.0-mini-gguf", "offline")]
    assert images[0]["gguf_file"] == "basal-2.0-mini-Q8_0.gguf"
    assert images[0]["target"] == "llamacpp"
    assert [r["target"] for r in plan["runners"]["include"]] == ["llamacpp-runner"]


def test_runner_images_listed(capsys):
    runners = _plan(capsys)["runners"]["include"]
    assert {(r["name"], r["target"]) for r in runners} == {
        ("basal-engine", "engine-runner"),
        ("basal-engine-llamacpp", "llamacpp-runner"),
    }


def test_serve_args_are_shell_joined(capsys):
    images = _plan(capsys, "--only", "basal-2.0-4.5b-fp8")["images"]["include"]
    assert {r["serve_args"] for r in images} == {"--template-prefix off"}


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "runtimes: {llamacpp: {base_image: x, port: 1}}\n"
            "images: [{name: a, model: m, runtime: llamacpp, mode: llamacpp}]\n",
            "gguf_file",
        ),
        (
            "runtimes: {engine: {base_image: x, port: 1}}\n"
            "images: [{name: a, model: m, runtime: engine, mode: llamacpp, gguf_file: f}]\n",
            "llamacpp runtime",
        ),
        (
            "runtimes: {engine: {base_image: x, port: 1}}\n"
            "images: [{name: a, model: m, runtime: engine, mode: turbo}]\n",
            "mode",
        ),
        (
            "runtimes: {engine: {base_image: x, port: 1}}\n"
            "images: [{name: a, model: m, runtime: engine, mode: mlx}]\n",
            "Apple",
        ),
        ("runtimes: {}\nimages: [{name: a, model: m, runtime: nope, mode: fast}]\n", "runtime"),
        (
            "runtimes: {engine: {image: basal-engine, base_image: x, port: 1}}\n"
            "images: [{name: basal-engine, model: m, runtime: engine, mode: fast}]\n",
            "duplicate",
        ),
    ],
)
def test_invalid_configs(tmp_path, body, match):
    with pytest.raises(config.ConfigError, match=match):
        config.load(_write(tmp_path, body))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "MODEL_DIR", tmp_path / "model")
    for key in ("BASAL_MODEL", "BASAL_MODE", "BASAL_GGUF_FILE", "BASAL_SERVE_ARGS", "PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BASAL_PORT", "8000")
    return monkeypatch


def _bake(path: Path, *files: str) -> None:
    path.mkdir()
    for name in files:
        (path / name).write_text("x")


def test_engine_online_uses_repo_id(env):
    env.setenv("BASAL_MODEL", "Remek/basal-2.0-max-FP8")
    env.setenv("BASAL_MODE", "fp8")
    p = cli.plan(["--deterministic"])
    assert p.llama is None
    assert p.serve[1:5] == ["--model", "Remek/basal-2.0-max-FP8", "--mode", "fp8"]
    assert p.serve[-1] == "--deterministic"


def test_engine_offline_uses_baked_folder(env):
    _bake(cli.MODEL_DIR, "config.json")
    env.setenv("BASAL_MODEL", "Remek/basal-2.0-4.5B")
    env.setenv("BASAL_MODE", "fast")
    env.setenv("BASAL_SERVE_ARGS", "--template-prefix off")
    p = cli.plan([])
    assert p.serve[1:3] == ["--model", str(cli.MODEL_DIR)]
    assert p.serve[-2:] == ["--template-prefix", "off"]


def test_llamacpp_offline_starts_llama_server_and_points_basal_at_it(env):
    _bake(cli.MODEL_DIR, "basal-2.0-max-Q8_0.gguf", "tokenizer.json")
    env.setenv("BASAL_MODEL", "Remek/basal-2.0-max-GGUF")
    env.setenv("BASAL_MODE", "llamacpp")
    env.setenv("BASAL_GGUF_FILE", "basal-2.0-max-Q8_0.gguf")
    env.setenv("BASAL_CONTEXT", "32768")
    p = cli.plan([])
    assert p.llama is not None
    assert p.llama[p.llama.index("-m") + 1] == str(cli.MODEL_DIR / "basal-2.0-max-Q8_0.gguf")
    assert p.llama[p.llama.index("-c") + 1] == "32768"
    assert p.serve[p.serve.index("--llamacpp-url") + 1] == p.llama_url
    assert p.serve[p.serve.index("--model") + 1] == str(cli.MODEL_DIR)


def test_llamacpp_online_fetches_repo(env, tmp_path):
    folder = tmp_path / "snap"
    _bake(folder, "basal-2.0-mini-Q8_0.gguf")
    calls = []
    env.setattr(cli, "fetch", lambda *a: calls.append(a) or folder)
    env.setenv("BASAL_MODEL", "Remek/basal-2.0-mini-GGUF")
    env.setenv("BASAL_MODE", "llamacpp")
    env.setenv("BASAL_GGUF_FILE", "basal-2.0-mini-Q8_0.gguf")
    p = cli.plan([])
    assert calls == [("Remek/basal-2.0-mini-GGUF", "main", "basal-2.0-mini-Q8_0.gguf")]
    assert p.serve[p.serve.index("--model") + 1] == str(folder)


def test_serve_passes_dash_args_through(env):
    env.setenv("BASAL_MODEL", "Remek/basal-2.0-mini")
    env.setenv("BASAL_MODE", "fast")
    captured = {}
    env.setattr(cli.os, "execv", lambda f, a: captured.setdefault("argv", a))
    cli.main(["serve", "--orders", "2"])
    assert captured["argv"][-2:] == ["--orders", "2"]


def test_runner_without_model_or_mode_fails(env):
    env.setenv("BASAL_MODE", "fast")
    with pytest.raises(SystemExit, match="BASAL_MODEL"):
        cli.plan([])
    env.delenv("BASAL_MODE")
    env.setenv("BASAL_MODEL", "x")
    with pytest.raises(SystemExit, match="BASAL_MODE"):
        cli.plan([])
