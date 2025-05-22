#!/usr/bin/env python
"""
Convert CFG‑derivation shards to flat text, including full step metadata.

Each output line looks like:
  [<GRAMMAR>…</GRAMMAR> <TYPE>…</TYPE>] 
  1|{S}|S -> { S } S|{{S}S}S|0 <STEP> 2|{{S}S}S|S -> ε|{{}}S|2 …

Arguments
---------
--input         Glob or list of .jsonl/.jsonl.gz shards
--train_out     Path to write train.txt
--val_out       Path for val.txt (default: alongside train.txt as val.txt)
--val_frac      Fraction to send to val (default 0.01)
--include_meta  Prepend <GRAMMAR> and <TYPE> tags
"""

import argparse, glob, gzip, json, pathlib, random, sys

STEP_SEP  = " <STEP> "
FIELD_SEP = "|"
random.seed(42)

def step_line(s: dict) -> str:
    """Serialize one step with all metadata fields."""
    # order: step, current_state, rule_applied, result_state, applied_at_index
    return FIELD_SEP.join([
        str(s["step"]),
        s["current_state"],
        s["rule_applied"],
        s["result_state"],
        str(s["applied_at_index"]),
    ])

def rec_to_text(rec: dict, include_meta: bool) -> str:
    """Flatten a JSON record into one training line."""
    parts = []
    if include_meta:
        parts.append(f"<GRAMMAR>{rec['grammar_name']}</GRAMMAR>")
        parts.append(f"<TYPE>{rec['trace_type']}</TYPE>")
    for s in rec["derivation"]:
        if s["rule_applied"] == "<STOP>":
            break
        parts.append(step_line(s))
    return STEP_SEP.join(parts)

def iter_records(paths):
    """Yield each JSON record from a list of JSONL or JSONL.GZ files."""
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"[WARN] bad JSON in {p}: {e}\n")

def main():
    p = argparse.ArgumentParser(description="CFG JSONL → flat text (full metadata)")
    p.add_argument("--input", required=True, nargs="+",
                   help="Glob(s) or file list of .jsonl(.gz) shards")
    p.add_argument("--train_out", required=True, help="Output train.txt")
    p.add_argument("--val_out",   help="Output val.txt (default: same folder as train)")
    p.add_argument("--val_frac",  type=float, default=0.01, help="Fraction to val set")
    p.add_argument("--include_meta", action="store_true",
                   help="Prepend <GRAMMAR> and <TYPE> tags")
    args = p.parse_args()

    # expand globs
    files = []
    for pat in args.input:
        files.extend(glob.glob(pat))
    if not files:
        sys.exit("ERROR: no input files matched.")

    # determine validation path
    val_path = args.val_out or str(pathlib.Path(args.train_out).with_name("val.txt"))
    pathlib.Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_path).parent.mkdir(parents=True, exist_ok=True)

    n_tr = n_vl = 0
    with open(args.train_out, "w", encoding="utf-8") as f_tr, \
         open(val_path,    "w", encoding="utf-8") as f_vl:

        for rec in iter_records(files):
            line = rec_to_text(rec, include_meta=args.include_meta)
            if not line:
                continue
            if random.random() < args.val_frac:
                f_vl.write(line + "\n"); n_vl += 1
            else:
                f_tr.write(line + "\n"); n_tr += 1

    print(f"✔ {n_tr} lines written to {args.train_out}")
    print(f"✔ {n_vl} lines written to {val_path}")

if __name__ == "__main__":
    main()
