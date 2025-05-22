#!/usr/bin/env python
"""
collect_env_info.py
────────────────────────────────────────────────────────
Writes three files in the current directory:

  • python_env.txt       – exact Python version + pip freeze
  • cuda_info.txt        – GPU + CUDA toolkit versions (if torch is present)
  • local_diff.txt       – diff of any files you CHANGED inside nanoGPT

No datasets or checkpoints are touched.
"""

from pathlib import Path
import json, os, subprocess, sys, textwrap, difflib

ROOT = Path(__file__).resolve().parent
OUT  = {
    "python_env": ROOT / "python_env.txt",
    "cuda_info" : ROOT / "cuda_info.txt",
    "local_diff": ROOT / "local_diff.txt",
}

# 1) Python version + pip freeze
with OUT["python_env"].open("w") as fh:
    fh.write(f"Python: {sys.version}\n\n")
    fh.write("# pip freeze\n")
    fh.write(subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
             .decode())

# 2) CUDA + GPU info (if torch installed)
try:
    import torch, platform, re
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
    cu  = torch.version.cuda or "n/a"
    fh  = OUT["cuda_info"].open("w")
    fh.write(f"Device          : {gpu}\n")
    fh.write(f"torch.__version : {torch.__version__}\n")
    fh.write(f"CUDA toolkit    : {cu}\n")
    fh.close()
except ImportError:
    OUT["cuda_info"].write_text("torch not installed – skipped\n")

# 3) diff for nanoGPT customisations
BASE_NANOGPT_URL = "https://raw.githubusercontent.com/karpathy/nanoGPT/master"

def diff_one(rel_path):
    """Return unified diff vs upstream head (remote fetch)."""
    import urllib.request, pathlib
    local_file = ROOT / "nanoGPT" / rel_path
    if not local_file.exists():
        return None
    src = local_file.read_text().splitlines(keepends=True)
    try:
        remote = urllib.request.urlopen(f"{BASE_NANOGPT_URL}/{rel_path}").read()
        remote = remote.decode().splitlines(keepends=True)
    except Exception:
        remote = ["(could not fetch upstream)\n"]
    return "".join(difflib.unified_diff(remote, src,
                                        fromfile=f"upstream/{rel_path}",
                                        tofile=f"local/{rel_path}"))

changed = []
for p in (ROOT / "nanoGPT").rglob("*.py"):
    rel = p.relative_to(ROOT / "nanoGPT")
    diff = diff_one(str(rel))
    if diff and len(diff.splitlines()) > 5:   # crude “changed” heuristic
        changed.append(diff)

OUT["local_diff"].write_text("\n\n".join(changed) or "No significant diffs.\n")

print("✓  wrote:", *[f.name for f in OUT.values()])

# 4) simple manifest of cfg-dataset source files
CFG_MANIFEST = ROOT / "cfg_dataset_manifest.txt"
with CFG_MANIFEST.open("w") as fh:
    for p in (ROOT / "cfg-dataset").rglob("*.py"):
        rel = p.relative_to(ROOT)
        fh.write(f"{rel}\t{p.stat().st_size} bytes\n")
print("✓  wrote:", CFG_MANIFEST.name)

