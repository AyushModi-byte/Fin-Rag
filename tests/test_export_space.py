"""The Hugging Face Space export: right files in, secrets and bulk out, correct front matter and LFS rules."""
import json
from pathlib import Path

import pytest

from src.jobs.export_space import FRONT_MATTER_KEYS, build_space_tree


def make_repo(root: Path) -> Path:
    files = {
        "Dockerfile": "FROM python:3.13-slim\n",
        "requirements.txt": "fastapi==1\n",
        "config.yaml": "index: {version: v2}\n",
        "config.v1.yaml": "index: {version: v1}\n",
        "main.py": "app = None\n",
        ".dockerignore": ".env\n",
        ".env": "GROQ_API_KEY_1=gsk_secret_value\n",
        "src/__init__.py": "",
        "src/api/routes.py": "x = 1\n",
        "src/api/__pycache__/routes.cpython-313.pyc": "binary",
        "tests/test_x.py": "def test_x(): pass\n",
        "frontend/index.html": "<html>",
        "data/raw/BIG.pdf": "pdf",
        "data/eval/docs.json": json.dumps({"A_2022_10K": {}}),
        "data/eval/runs/run.json": "{}",
        "logs/app.log": "log",
        ".cache/llm/budget.json": "{}",
        ".cache/semantic_cache.json": "{}",
        ".cache/index/v1/manifest.json": "{}",
        ".cache/index/v2/manifest.json": "{}",
        ".cache/index/v2/parents.jsonl": "{}\n",
        ".cache/index/v2/qdrant/collection/storage.sqlite": "sqlite",
        ".cache/index/v2/bm25/vocab.json": "{}",
        ".cache/index/v2/shards/shard_00000.npz": "npz",
    }
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


@pytest.fixture()
def tree(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "space"
    copied = build_space_tree(repo, out, index_version="v2")
    return out, copied


def names(out: Path) -> set[str]:
    return {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_serving_files_and_the_v2_index_are_included(tree):
    out, _ = tree
    got = names(out)
    for want in ["Dockerfile", "requirements.txt", "config.yaml", "main.py", ".dockerignore", "src/api/routes.py",
                 "data/eval/docs.json", ".cache/index/v2/manifest.json", ".cache/index/v2/parents.jsonl",
                 ".cache/index/v2/qdrant/collection/storage.sqlite", ".cache/index/v2/bm25/vocab.json"]:
        assert want in got, want


def test_secrets_caches_data_tests_frontend_and_shards_are_left_out(tree):
    out, _ = tree
    got = names(out)
    assert ".env" not in got
    assert not any(n.startswith(("tests/", "frontend/", "data/raw/", "data/eval/runs/", "logs/", ".cache/llm/")) for n in got)
    assert ".cache/semantic_cache.json" not in got and ".cache/index/v1/manifest.json" not in got
    assert not any("shards" in n or n.endswith(".pyc") or "__pycache__" in n for n in got)
    assert "config.v1.yaml" not in got  # the rollback config is not served


def test_no_secret_value_appears_anywhere_in_the_export(tree):
    out, _ = tree
    for p in out.rglob("*"):
        if p.is_file():
            assert b"gsk_secret_value" not in p.read_bytes(), p


def test_readme_has_docker_space_front_matter(tree):
    out, _ = tree
    text = (out / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head = text.split("---\n")[1]
    fields = dict(line.split(": ", 1) for line in head.strip().splitlines())
    assert fields["sdk"] == "docker" and fields["app_port"] == "8000" and fields["title"] == "FinRag"
    assert set(fields) == set(FRONT_MATTER_KEYS)


def test_index_files_are_tracked_with_lfs(tree):
    out, _ = tree
    attrs = (out / ".gitattributes").read_text(encoding="utf-8")
    assert ".cache/index/v2/**" in attrs and "filter=lfs" in attrs


def test_export_reports_what_it_copied_and_refuses_a_missing_index(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "space2"
    copied = build_space_tree(repo, out, index_version="v2")
    assert "Dockerfile" in copied and not any(".env" == c for c in copied)
    with pytest.raises(FileNotFoundError):
        build_space_tree(repo, tmp_path / "space3", index_version="v9")


def test_export_refuses_to_write_into_a_non_empty_directory(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "space"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_space_tree(repo, out, index_version="v2")
