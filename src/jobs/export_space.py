"""Assemble a Hugging Face Space (Docker) from this repo. Offline, no network, no LLM.

    python -m src.jobs.export_space --out ../finrag-space

Writes only what the image needs (the Dockerfile COPYs these paths): Dockerfile, requirements, config.yaml,
main.py, src/, data/eval/docs.json and the serving files of ONE index version. Never copied: .env, any cache,
tests, the frontend, raw data, run files, index shards, other index versions. The Space gets a README with the
Spaces front matter and a .gitattributes that sends the index through Git LFS (HF rejects big plain files).
The output directory must be empty or new. Pushing it is a separate, manual step (docs/DEPLOYMENT.md).
"""
import argparse
import shutil
import sys
from pathlib import Path

FRONT_MATTER_KEYS = ["title", "emoji", "colorFrom", "colorTo", "sdk", "app_port", "short_description"]

TOP_LEVEL_FILES = ["Dockerfile", "requirements.txt", "config.yaml", "main.py", ".dockerignore"]
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc"}
INDEX_SKIP_DIRS = {"shards"}  # embedding shards are only needed to rebuild the index, not to serve it

README = """---
title: FinRag
emoji: 📈
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
short_description: Cited answers from SEC 10-K filings
---

# FinRag API

Backend of FinRag: retrieval-augmented question answering over SEC 10-K filings. This Space serves the API only
(`GET /health`, `POST /query`); the web UI is a separate static site.

Secrets (Space settings): `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, `FINRAG_CORS_ORIGINS`;
optional `FINRAG_API_KEY`, `FINRAG_LOG_SALT`.
"""

GITATTRIBUTES = """.cache/index/{version}/** filter=lfs diff=lfs merge=lfs -text
"""


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _walk(root: Path, skip_dirs: set[str]):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES:
            continue
        if any(part in skip_dirs for part in p.relative_to(root).parts[:-1]):
            continue
        yield p


def build_space_tree(repo: Path, out: Path, index_version: str = "v2") -> list[str]:
    repo, out = Path(repo), Path(out)
    index_src = repo / ".cache" / "index" / index_version
    if not (index_src / "manifest.json").is_file():
        raise FileNotFoundError(f"no built index at {index_src}: build it first (python -m src.jobs.index_corpus)")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; export into a new or empty directory")

    copied: list[str] = []

    def add(src: Path, rel: str) -> None:
        _copy(src, out / rel)
        copied.append(rel)

    for name in TOP_LEVEL_FILES:
        if (repo / name).is_file():
            add(repo / name, name)
    for p in _walk(repo / "src", SKIP_DIRS):
        add(p, p.relative_to(repo).as_posix())
    add(repo / "data" / "eval" / "docs.json", "data/eval/docs.json")
    for p in _walk(index_src, INDEX_SKIP_DIRS):
        add(p, p.relative_to(repo).as_posix())

    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES.format(version=index_version), encoding="utf-8")
    copied += ["README.md", ".gitattributes"]
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="new or empty directory for the Space repo")
    ap.add_argument("--index-version", default="v2")
    args = ap.parse_args(argv)
    try:
        copied = build_space_tree(Path.cwd(), Path(args.out), args.index_version)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in Path(args.out).rglob("*") if p.is_file())
    print(f"exported {len(copied)} files, {size / 1e6:.0f} MB, to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
