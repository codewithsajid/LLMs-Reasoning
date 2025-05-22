#!/usr/bin/env python
"""
Generic plain‑text → binary dataset converter for nanoGPT.

Usage
-----
python data/prepare.py \
    --input data/cfg/v1/train.txt \
    --tokenizer char \
    --val_frac 0.01 \
    --out_dir data/cfg/v1

Arguments
---------
--input        Path to *train* text file (val path auto‑derived)
--val_path     Explicit val text file (optional; else input dir / val.txt)
--tokenizer    char | gpt2  (default char)
--val_frac     If val file missing, split this fraction from train (default 0.01)
--out_dir      Directory to write train.bin, val.bin, meta.json
"""

import argparse, json, os, mmap, numpy as np, random, sys, tiktoken
from pathlib import Path
from tqdm import tqdm

DTYPE = np.uint16          # good up to 65535 tokens
CHUNK_SIZE = 1024 * 1024   # 1 MB read blocks


def build_char_vocab(text_file):
    """Return sorted unique bytes (chars) and encoder/decoder dicts."""
    with open(text_file, "r", encoding="utf-8") as f:
        vocab = sorted(set(f.read()))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return vocab, stoi, itos


def encode_file(path, encoder, tokenizer_type):
    """Generator that yields encoded numpy arrays for each CHUNK_SIZE block."""
    if tokenizer_type == "char":
        stoi = encoder
        with open(path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield np.array([stoi[c] for c in chunk], dtype=DTYPE)
    else:  # gpt2 BPE
        enc = encoder
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ids = enc.encode_ordinary(line) + [enc.eot_token]
                yield np.array(ids, dtype=DTYPE)


def write_bin(source_txt, out_bin, encoder, tok_type):
    """Stream‑encode source_txt into out_bin file."""
    total_len = 0
    # first pass to size
    for arr in encode_file(source_txt, encoder, tok_type):
        total_len += len(arr)
    arr_bin = np.memmap(out_bin, dtype=DTYPE, mode="w+", shape=(total_len,))
    idx = 0
    for arr in encode_file(source_txt, encoder, tok_type):
        arr_bin[idx : idx + len(arr)] = arr
        idx += len(arr)
    arr_bin.flush()
    return total_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="train.txt")
    ap.add_argument("--val_path", help="val.txt (optional)")
    ap.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--out_dir", required=False)
    args = ap.parse_args()

    train_txt = Path(args.input)
    val_txt   = Path(args.val_path) if args.val_path else train_txt.with_name("val.txt")
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # if val.txt doesn't exist, split from train
    if not val_txt.exists():
        print(f"[info] {val_txt} missing; splitting {args.val_frac*100:.1f}% from train")
        lines = open(train_txt, "r", encoding="utf-8").read().splitlines()
        random.seed(42); random.shuffle(lines)
        pivot = int(len(lines) * (1 - args.val_frac))
        with open(train_txt, "w", encoding="utf-8") as f_tr:
            f_tr.write("\n".join(lines[:pivot]))
        with open(val_txt, "w", encoding="utf-8") as f_val:
            f_val.write("\n".join(lines[pivot:]))

    if args.tokenizer == "char":
        vocab, stoi, itos = build_char_vocab(train_txt)
        encoder = stoi
        vocab_size = len(vocab)
    else:  # gpt2 BPE
        enc = tiktoken.get_encoding("gpt2")
        encoder = enc
        vocab_size = enc.n_vocab

    print(f"Encoding train.txt → train.bin  (vocab={vocab_size})")
    train_tokens = write_bin(train_txt, out_dir / "train.bin", encoder, args.tokenizer)

    print("Encoding val.txt   → val.bin")
    val_tokens   = write_bin(val_txt,   out_dir / "val.bin",   encoder, args.tokenizer)

    # meta
    meta = dict(
        tokenizer=args.tokenizer,
        vocab_size=int(vocab_size),
        train_tokens=int(train_tokens),
        val_tokens=int(val_tokens),
    )
    if args.tokenizer == "char":
        meta["itos"] = {i: ch for ch, i in encoder.items()}
    json.dump(meta, open(out_dir / "meta.json", "w"))

    print("✓ Done.  Stats:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
