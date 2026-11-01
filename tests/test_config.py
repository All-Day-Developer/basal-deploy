import json
from pathlib import Path

import pytest

from basal_images import cli, config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "images.yaml")


def _matrix(capsys, *extra: str) -> list[dict]:
    assert cli.main(["matrix", "--config", CONFIG, "--compact", *extra]) == 0
    return json.loads(capsys.readouterr().out)["matrix"]["include"]


def test_repo_config_loads():
    cfg = config.load(CONFIG)
    assert cfg.images
    assert all(i.runtime.name in {"vllm", "llamacpp"} for i in cfg.images)


def test_matrix_skips_disabled_and_expands_variants(capsys):
    include = _matrix(capsys)
    names = {row["name"] for row in include}
    assert "basal-2.0-mini-mlx-8bit" not in names
    variants = {row["variant"] for row in include if row["name"] == "basal-2.0-mini"}
    assert variants == {"slim", "weights"}


def test_matrix_only_and_variant_filters(capsys):
    include = _matrix(capsys, "--only", "basal-2.0-mini-gguf", "--variant", "weights")
    assert [(r["name"], r["variant"]) for r in include] == [("basal-2.0-mini-gguf", "weights")]
    assert include[0]["gguf_file"] == "basal-2.0-mini-Q5_K_M.gguf"


def test_precompile_routes_vllm_to_gpu_runner(capsys):
    cfg = config.load(CONFIG)
    include = _matrix(capsys)
    for row in include:
        if row["runtime"] == "vllm":
            assert row["precompile"] is True
            assert json.loads(row["runner"]) == cfg.precompile.runner
        else:
            assert row["precompile"] is False
            assert json.loads(row["runner"]) == "ubuntu-latest"


def test_no_precompile_flag(capsys):
    include = _matrix(capsys, "--no-precompile")
    assert not any(row["precompile"] for row in include)
    assert {json.loads(row["runner"]) for row in include} == {"ubuntu-latest"}


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


def test_precompile_requires_runner(tmp_path):
    path = _write(
        tmp_path,
        """
runtimes: {vllm: {base_image: x, port: 1, precompile: true}}
images: [{name: a, model: m, runtime: vllm}]
""",
    )
    with pytest.raises(config.ConfigError, match="precompile"):
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
