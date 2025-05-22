#!/usr/bin/env python
"""cfg_to_text.py  (v2)

Convert CFG‑derivation shards (.jsonl or .jsonl.gz) into a flat text corpus
for nanoGPT *with the new 3‑field line format*:

    GrammarName │ FinalString │ step|curr|rule|result|idx <STEP> …

Optional flags:
  --include_meta   – also prepend <TYPE> tag (trace_type) after grammar.
  --val_frac 0.1  – validation split fraction (default 1 %).

The script autodetects gzip, preserves full step metadata, and writes
train / val files.

Example:
    python cfg_to_text.py \
        --input "dataset/*.jsonl.gz" \
        --train_out data/cfg/v2/train.txt \
        --val_frac 0.1
"""

import argparse, glob, gzip, json, pathlib, random, sys

STEP_SEP  = " <STEP> "
FIELD_SEP = "|"
META_SEP  = " │ "   # separates Grammar │ String │ Derivation
random.seed(42)


def step_line(step: dict) -> str:
    """Serialize one derivation step with all metadata fields."""
    return FIELD_SEP.join([
        str(step["step"]),
        step["current_state"],
        step["rule_applied"],
        step["result_state"],
        str(step["applied_at_index"]),
    ])


def rec_to_text(rec: dict, include_meta: bool) -> str:
    """Return one corpus line in the new 3‑field format."""
    grammar = rec["grammar_name"]
    final   = rec["final_string"]
    steps   = STEP_SEP.join(
        step_line(s) for s in rec["derivation"] if s["rule_applied"] != "<STOP>"
    )
    prefix  = grammar
    if include_meta:
        prefix = f"<GRAMMAR>{grammar}</GRAMMAR> <TYPE>{rec['trace_type']}</TYPE>"
    return META_SEP.join([prefix, final, steps])


def iter_records(paths):
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"[WARN] bad JSON in {path}: {e}\n")


def main():
    ap = argparse.ArgumentParser(description="CFG JSONL → flat text (3‑field format)")
    ap.add_argument("--input", required=True, nargs="+", help="glob(s) or file list")
    ap.add_argument("--train_out", required=True, help="train.txt path")
    ap.add_argument("--val_out", help="val.txt path (default sibling of train)")
    ap.add_argument("--val_frac", type=float, default=0.01, help="validation fraction")
    ap.add_argument("--include_meta", action="store_true", help="prepend tags")
    args = ap.parse_args()

    files = [f for pat in args.input for f in glob.glob(pat)]
    if not files:
        sys.exit("ERROR: no input files matched.")

    val_path = args.val_out or str(pathlib.Path(args.train_out).with_name("val.txt"))
    pathlib.Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_path).parent.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    with open(args.train_out, "w", encoding="utf-8") as f_tr, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for rec in iter_records(files):
            line = rec_to_text(rec, include_meta=args.include_meta)
            if random.random() < args.val_frac:
                f_val.write(line + "\n"); n_val += 1
            else:
                f_tr.write(line + "\n"); n_train += 1

    print(f"✓ {n_train} train  |  {n_val} val lines written")


if __name__ == "__main__":
    main()