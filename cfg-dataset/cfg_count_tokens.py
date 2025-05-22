#!/usr/bin/env python
# count_tokens_and_chars.py
import gzip, glob, json, tiktoken

enc = tiktoken.get_encoding("gpt2")

def bpe_len(txt: str) -> int:
    return len(enc.encode(txt))

total_tokens = total_chars = total_lines = 0

for path in glob.glob("dataset/dyck_1/cfg_derivation_dataset.jsonl.gz"):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            total_lines   += 1
            total_chars   += len(ln)             # char-level (unicode codepoints)
            total_tokens  += bpe_len(ln)         # GPT-2 BPE tokens

# ── report ──────────────────────────────────────────────
print(f"Total samples            : {total_lines:,}")
print(f"Total GPT-2 tokens       : {total_tokens:,}")
print(f"Average tokens / sample  : {total_tokens/total_lines:,.1f}")
print()
print(f"Total characters         : {total_chars:,}")
print(f"Average chars / sample   : {total_chars/total_lines:,.1f}")
