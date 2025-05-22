#!/usr/bin/env python
"""
build_dataset_parallel.py   –   Parallel, crash-safe CFG dataset builder
-----------------------------------------------------------------------

"""
from __future__ import annotations
import argparse, gzip, itertools, json, os, random, shutil, sys
from collections import Counter, defaultdict, OrderedDict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

from grammar_catalog_v2 import GRAMMARS
from derivation_engine import (
    gen_derivations,
    inject_partial,
    inject_single_error,
    trace_to_tree,
)

# ───── helpers ───────────────────────────────────────────────────────
def _safe_randint(lo: int, hi: int) -> int:
    """randint with graceful fallback if range collapses."""
    return lo if hi < lo else random.randint(lo, hi)

def length_bucket(n: int, s_thr: int, m_thr: int) -> str:
    return "short" if n <= s_thr else "medium" if n <= m_thr else "long"

class Reservoir:
    __slots__ = ("cap", "buf", "seen")
    def __init__(self, cap): self.cap, self.buf, self.seen = cap, [], 0
    def push(self, item):
        self.seen += 1
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            j = random.randrange(self.seen)
            if j < self.cap: self.buf[j] = item

def write_jsonl(path: Path, recs, gzip_flag=False):
    open_fn = gzip.open if gzip_flag else open
    with open_fn(path, "wt", encoding="utf-8") as fh:
        for r in recs:
            json.dump(r, fh, separators=(",", ":")); fh.write("\n")

def _grammar_1line(cfg) -> str:
    return "; ".join(f"{p.lhs()} -> {' '.join(map(str,p.rhs())) or 'ε'}"
                     for p in cfg.productions())

def _ordered_record(cfg, final_string, derivation,
                    *, length_val, gname, strategy, trace_type):
    rec = OrderedDict()
    rec["grammar"]      = _grammar_1line(cfg)
    rec["final_string"] = final_string
    rec["derivation"]   = derivation
    rec["meta"] = {
        "length":       length_val,
        "grammar_name": gname,
        "strategy":     strategy,
        "trace_type":   trace_type,
    }
    return rec

# ───── parallelize -------------------------------------------------------------
def _init_worker(seed): random.seed(seed + os.getpid())

def build_grammar(gname: str, *, opts: Dict) -> List[Path]:
    grammar = GRAMMARS[gname]
    out_dir: Path  = opts["out_dir"]
    cap           = opts["bucket_cap"]
    s_thr, m_thr  = opts["len_thr"]
    max_depth     = opts["max_depth"]
    gzip_flag     = opts["gzip"]
    contrast_p    = opts["contrast_ratio"]

    shard_paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for strategy in ("leftmost", "rightmost", "random"):
        shard = out_dir / f"{gname}__{strategy}.jsonl{'.gz' if gzip_flag else ''}"
        if shard.exists():            # shard already done in previous run
            shard_paths.append(shard); continue

        capacity = {
            "correct":      int(cap * (1-contrast_p)),
            "partial":      int(cap * contrast_p / 4),
            "single_error": int(cap * contrast_p / 4),
            "off_by_index": int(cap * contrast_p / 4),
            "hallucinated": int(cap * contrast_p / 4),
        }
        reservoirs = defaultdict(lambda: Reservoir(capacity[key[0]]))

        for rec in gen_derivations(grammar, strategy, max_depth=max_depth):
            trace        = rec["derivation"]
            final_string = rec["final_string"]
            length_val   = rec["length"]
            trace_type   = "ill_formed" if gname.startswith("unbal_") else "correct"

            # ─── contrastive (not-correct examples) ──────────────────────────────────
            if random.random() < contrast_p:
                kind       = random.choice(
                    ["partial", "single_error", "off_by_index", "hallucinated"]
                )
                trace_type = kind
                if kind == "partial":
                    if len(trace) > 1:
                        k = _safe_randint(1, len(trace)-1)
                        trace = inject_partial(trace, k)
                    else:               # fallback to make it still contrastive
                        kind = trace_type = "single_error"

                if kind in ("single_error", "off_by_index", "hallucinated"):
                    idx = _safe_randint(1, len(trace)-1)  # 1 if len==1
                    mode = ("wrong_rule"   if kind=="single_error" else
                            "off_by_index" if kind=="off_by_index" else
                            "hallucinated")
                    trace = inject_single_error(grammar, trace, idx, mode)

                final_string = trace[-1]["result_state"]
                length_val   = len(final_string)

            # ─── optional parse tree ────────────────────────────────────
            parse_tree = None
            if trace_type in ("correct", "partial"):
                usable = trace[:-1] if trace and trace[-1]["rule_applied"]=="<STOP>" else trace
                try:   parse_tree = trace_to_tree(usable)
                except AssertionError: pass

            # ─── ordered record out ────────────────────
            rec_out = _ordered_record(grammar, final_string, trace,
                                      length_val=length_val,
                                      gname=gname, strategy=strategy,
                                      trace_type=trace_type)
            if parse_tree: rec_out["parse_tree"] = parse_tree

            key = (trace_type, length_bucket(length_val, s_thr, m_thr))
            reservoirs[key].push(rec_out)

        # write shard
        bucket_records = itertools.chain.from_iterable(r.buf for r in reservoirs.values())
        write_jsonl(shard, bucket_records, gzip_flag)
        shard_paths.append(shard)

    return shard_paths

def _worker(args): g, opts = args; return build_grammar(g, opts=opts)

# ───── cli-args -------------------------------------------------------
def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, cpu_count()))
    ap.add_argument("--output-dir", type=Path, default=Path("dataset/v2"))
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--bucket-cap", type=int, default=3000)
    ap.add_argument("--len-thr", type=int, nargs=2, default=(15, 40))
    ap.add_argument("--contrast-ratio", type=float, default=0.30)
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--grammars", nargs="*", default=sorted(GRAMMARS))
    opts = vars(ap.parse_args(argv))

    out: Path = opts["output_dir"]; out.mkdir(parents=True, exist_ok=True)

    frozen = dict(out_dir=out,
                  bucket_cap=opts["bucket_cap"],
                  len_thr=tuple(opts["len_thr"]),
                  max_depth=opts["max_depth"],
                  gzip=opts["gzip"],
                  contrast_ratio=opts["contrast_ratio"])

    print(f"▶ Building into {out}  ({len(opts['grammars'])} grammars) …")

    with Pool(opts["workers"], initializer=_init_worker, initargs=(42,)) as pool:
        gargs = [(g, frozen) for g in opts["grammars"]]
        shard_lists = pool.map(_worker, gargs, chunksize=1)

    # merge shards -> final file
    final = out / ("cfg_derivation_dataset.jsonl" + (".gz" if opts["gzip"] else ""))
    opener = gzip.open if opts["gzip"] else open
    with opener(final, "wt") as out_f:
        for shard in sorted(itertools.chain.from_iterable(shard_lists)):
            with opener(shard, "rt") as fh: shutil.copyfileobj(fh, out_f)
    print("✔ merged →", final.name)

if __name__ == "__main__":
    import multiprocessing as mp
    try: mp.set_start_method("fork")   # Linux
    except RuntimeError: mp.set_start_method("spawn")
    main()
