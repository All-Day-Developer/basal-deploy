import json
from pathlib import Path

import pytest

from basal_images import cli, config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "images.yaml")


def _plan(capsys, *extra: str) -> dict:
    assert cli.main(["matrix", "--config", CONFIG, "--compact", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_repo_config_loads():
    cfg = config.load(CONFIG)
    assert cfg.images
    assert all(i.runtime.name in cfg.runtimes for i in cfg.images)


def test_matrix_skips_disabled_and_expands_variants(capsys):
    images = _plan(capsys)["images"]["include"]
    names = {row["name"] for row in images}
    assert "basal-2.0-mini-mlx-8bit" not in names
    variants = {row["variant"] for row in images if row["name"] == "basal-2.0-mini"}
    assert variants == {"online", "offline"}


def test_matrix_only_and_variant_filters(capsys):
    plan = _plan(capsys, "--only", "basal-2.0-mini-gguf", "--variant", "offline")
    images = plan["images"]["include"]
    assert [(r["name"], r["variant"]) for r in images] == [("basal-2.0-mini-gguf", "offline")]
    assert images[0]["gguf_file"] == "basal-2.0-mini-Q5_K_M.gguf"
    assert [r["runtime"] for r in plan["runners"]["include"]] == ["llamacpp"]


def test_runner_images_listed(capsys):
    runners = _plan(capsys)["runners"]["include"]
    assert {(r["name"], r["runtime"]) for r in runners} == {
        ("basal-vllm", "vllm"),
        ("basal-llamacpp", "llamacpp"),
    }


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "images.yaml"
    path.write_text(body)
    return path


def test_llamacpp_requires_gguf_file(tmp_path):
    path = _write(
        tmp_path,
        """
runtimes: {llamacpp: {base_image: x, port: 1}}
images: [{name: a, model: m, runtime: llamacpp}]
""",
    )
    with pytest.raises(config.ConfigError, match="gguf_file"):
        config.load(path)


def test_rejects_unknown_runtime_and_duplicates(tmp_path):
    path = _write(tmp_path, "runtimes: {}\nimages: [{name: a, model: m, runtime: nope}]\n")
    with pytest.raises(config.ConfigError, match="runtime"):
        config.load(path)

    path = _write(
        tmp_path,
        """
runtimes: {vllm: {base_image: x, port: 1}}
images:
  - {name: a, model: m, runtime: vllm}
  - {name: a, model: m, runtime: vllm}
""",
    )
    with pytest.raises(config.ConfigError, match="duplicate"):
        config.load(path)


def test_image_name_cannot_shadow_runner(tmp_path):
    path = _write(
        tmp_path,
        """
runtimes: {vllm: {image: basal-vllm, base_image: x, port: 1}}
images: [{name: basal-vllm, model: m, runtime: vllm}]
""",
    )
    with pytest.raises(config.ConfigError, match="duplicate"):
        config.load(path)


def test_serve_offline_vllm(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr(cli, "MODEL_DIR", model_dir)
    monkeypatch.setenv("BASAL_RUNTIME", "vllm")
    monkeypatch.setenv("BASAL_MODEL", "Remek/basal-2.0-mini")
    monkeypatch.setenv("BASAL_PORT", "8000")
    argv = cli.server_argv([])
    assert argv[:3] == ["vllm", "serve", str(model_dir)]


def test_serve_online_llamacpp(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "MODEL_DIR", tmp_path / "missing")
    monkeypatch.setenv("BASAL_RUNTIME", "llamacpp")
    monkeypatch.setenv("BASAL_MODEL", "Remek/basal-2.0-mini-GGUF")
    monkeypatch.setenv("BASAL_GGUF_FILE", "basal-2.0-mini-Q5_K_M.gguf")
    monkeypatch.setenv("BASAL_PORT", "8080")
    argv = cli.server_argv([])
    assert "--hf-repo" in argv and "basal-2.0-mini-Q5_K_M.gguf" in argv


def test_runner_without_model_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "MODEL_DIR", tmp_path / "missing")
    monkeypatch.setenv("BASAL_RUNTIME", "vllm")
    monkeypatch.delenv("BASAL_MODEL", raising=False)
    monkeypatch.setenv("BASAL_PORT", "8000")
    with pytest.raises(SystemExit, match="BASAL_MODEL"):
        cli.server_argv([])
