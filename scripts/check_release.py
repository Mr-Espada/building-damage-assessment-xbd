"""Inspect the candidate tree without loading imagery/models or contacting GitHub."""

import ast
import json
from pathlib import Path
import re
import sys


def check(root):
    errors = []
    allowed = {".md", ".py", ".json", ".ipynb", ".toml", ".txt", ".bib", ".png"}
    ignored_generated = {"__pycache__", ".git", ".venv", ".pytest_cache"}
    sensitive = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bxox[baprs]-[A-Za-z0-9-]{16,}\b")
    count = 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(x in ignored_generated or x.endswith(".egg-info") for x in rel.parts):
            continue
        if p.is_symlink():
            errors.append(f"External-link risk: {rel}")
            continue
        if not p.is_file():
            continue
        count += 1
        if rel not in {Path(".gitignore"), Path("LICENSE")} and p.suffix not in allowed:
            errors.append(f"Unexpected/data/binary file: {rel}")
        if p.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Oversized file: {rel}")
            continue
        if p.suffix == ".png":
            if rel != Path("results/reported_metrics.png"):
                errors.append(f"Unreviewed image: {rel}")
            continue
        text = p.read_text(errors="replace")
        if sensitive.search(text):
            errors.append(f"Potential secret pattern: {rel} (value not printed)")
        if p.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as e:
                errors.append(f"Syntax error: {rel}:{e.lineno}")
        if p.suffix == ".ipynb":
            nb = json.loads(text)
            if any(c.get("outputs") or c.get("execution_count") is not None for c in nb["cells"] if c["cell_type"] == "code"):
                errors.append(f"Notebook outputs/counts retained: {rel}")
        if p.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                local = target.split("#", 1)[0]
                if not (p.parent / local).exists():
                    errors.append(f"Broken local link: {rel} -> {local}")
    print(json.dumps({"checked_files": count, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(check(Path(__file__).resolve().parents[1]))
