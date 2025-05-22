import json, gzip, glob, pandas as pd
from tabulate import tabulate        

DATA_DIR = "dataset/dyck_1"              
SHARDS    = glob.glob(f"{DATA_DIR}/*__*.jsonl.gz")
assert SHARDS, "No shards found -- check DATA_DIR"

# ── 1.  Loading each line as a list of dicts ─────────────────────────────
records = []
for fn in SHARDS:
    with gzip.open(fn, "rt", encoding="utf-8") as fh:
        for ln in fh:
            rec  = json.loads(ln)
            m    = rec["meta"]
            records.append({
                "grammar":    m["grammar_name"],
                "strategy":   m["strategy"],
                "trace_type": m["trace_type"],
                "length":     m["length"],
                "has_tree":   "parse_tree" in rec,
            })

df = pd.DataFrame(records)

# ── 2.  Pivot into per-grammar summary table ─────────────────────────────
summary = (df.pivot_table(index="grammar",
                          columns="trace_type",
                          aggfunc="size",
                          fill_value=0)
             .assign(total=lambda x: x.sum(axis=1),
                     pct_correct=lambda x: x["correct"] / x["total"],
                     pct_errors=lambda x: (x["partial"]
                                           + x["single_error"]
                                           + x["off_by_index"]
                                           + x["hallucinated"]) / x["total"]))

length_stats = df.groupby("grammar")["length"].agg(mean="mean", max="max")
tree_frac    = df.groupby("grammar")["has_tree"].mean().rename("tree_fraction")

final = (summary.join(length_stats).join(tree_frac)
         .sort_values("total", ascending=False))

# ── 3.  Pretty-print as Markdown ──────────────────────
print("\n### Per-grammar summary\n")
print(tabulate(final, headers="keys", tablefmt="github"))

# ── 4.  Save to CSV (optional) ────────────────────────────────
#final.to_csv("per_grammar_summary.csv")
#print("\nFull table saved to per_grammar_summary.csv")
