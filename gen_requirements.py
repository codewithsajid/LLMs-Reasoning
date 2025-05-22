#!/usr/bin/env python
"""
Scan cfg-dataset/ and nanoGPT/ for 'import xxx' lines,
cross-reference with `pip freeze`, and write:

  • python_env.txt   – minimal freeze (only needed pkgs)
  • requirements.txt – installable subset

Usage
-----
  python gen_requirements.py
"""
from pathlib import Path
import re, subprocess, sys, textwrap

ROOT = Path(__file__).resolve().parent
TARGET_DIRS = ["cfg-dataset", "nanoGPT"]
# mapping from import name → pip package (when they differ)
ALIASES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "nltk": "nltk",
    "pandas": "pandas",
    "numpy": "numpy",
    "tiktoken": "tiktoken",
    "transformers": "transformers",
    "datasets": "datasets",
}

# 1) collect import names
imports = set()
rx = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)")
for d in TARGET_DIRS:
    for py in (ROOT / d).rglob("*.py"):
        for line in py.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = rx.match(line)
            if m:
                top = m.group(1).split(".")[0]
                if top in ALIASES:
                    imports.add(top)

# 2) grab versions from pip freeze
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
ver = {}
for ln in freeze.splitlines():
    if "@" in ln:          # editable install or VCS pin
        name = ln.split("@")[0]
    else:
        name = ln.split("==")[0].lower()
    ver[name] = ln

pkgs = [ALIASES[x] for x in sorted(imports)]
req_lines = [ver[p] if p in ver else p for p in pkgs]

# 3) write files
(ROOT / "python_env.txt").write_text(
    "Python minimal environment for cfg-dataset + nanoGPT\n\n"
    + "\n".join(req_lines) + "\n"
)
(ROOT / "requirements.txt").write_text("\n".join(req_lines) + "\n")

print("✓  wrote python_env.txt and requirements.txt with:\n", "\n ".join(pkgs))
