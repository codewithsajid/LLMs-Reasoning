"""
CFG catalog v2 — programmatic, parametric, and append v1
=============================================================
Highlights
* Single-pair Dyck-1 grammars in both LEFT- and RIGHT-recursive forms, means *{S} and {S}* forms
* Unbalanced variants (exactly one extra open or close bracket)
* Multi-type Dyck-k grammar (k = 3 by default)
* HTML-style tag pairing and escaped-delimiter examples
* Fan-out example ( S → “{” S S “}” S )
* All v1 grammars also imported

Public symbol
-------------
GRAMMARS : dict[str, nltk.grammar.CFG]
"""

from __future__ import annotations
from typing import Dict, Tuple, List

from nltk.grammar import CFG

from grammar_helper import (
    make_dyck1,
    make_dyck_k,
)

# ───────────────────────────── helpers ────────────────────────────────────
def make_unbalanced(
    name: str,
    open_tok: str,
    close_tok: str,
    *,
    extra: str = "open",
) -> Tuple[str, CFG]:
    """
    Return a grammar that leaves exactly **one** bracket 
    (either an unmatched opening or closing token).
    """
    if extra not in {"open", "close"}:
        raise ValueError("extra must be 'open' or 'close'")

    if extra == "open":
        # Either one leading '(' or a normal balanced pair with an extra opener inside
        bnf = f'S -> "{open_tok}" S | "{open_tok}" S "{close_tok}" | ""'
    else:
        # Either one trailing ')' or a balanced pair with an unmatched closer inside
        bnf = f'S -> S "{close_tok}" | "{open_tok}" S "{close_tok}" | ""'
    return name, CFG.fromstring(bnf)


# ────────────────────────── catalogue build ───────────────────────────────
GRAMMARS: Dict[str, CFG] = {}

# 1.  Balanced single-pair families (six token pairs × two recursion styles)
PAIRS: List[Tuple[str, str, str]] = [
    ("brace",   "{",  "}"),
    ("paren",   "(",  ")"),
    ("square",  "[",  "]"),
    ("angle",   "<",  ">"),
    ("unicode", "｛", "｝"),
    ("double",  "{{", "}}"),           # multi-token delimiter
]

for tag, op, cl in PAIRS:
    # left-recursive and right-recursive Dyck-1
    GRAMMARS[f"{tag}_left"]  = make_dyck1(op, cl, recursion="left")
    GRAMMARS[f"{tag}_right"] = make_dyck1(op, cl, recursion="right")

# 2.  Unbalanced variants (exactly one extra opener/closer)
for tag, op, cl in PAIRS:
    n, g = make_unbalanced(f"unbal_{tag}_open",  op, cl, extra="open")
    GRAMMARS[n] = g
    n, g = make_unbalanced(f"unbal_{tag}_close", op, cl, extra="close")
    GRAMMARS[n] = g

# 3.  Dyck-3 heterogeneous ( three bracket types, ambiguous concatenation )
GRAMMARS["dyck3_mixed"] = make_dyck_k([("{", "}"), ("(", ")"), ("[", "]")],
                                      recursion="mixed")

# 4.  HTML-style coloured tags (different open/close strings)
GRAMMARS["html_coloured_tags"] = CFG.fromstring(r"""
    S -> "<div>"  S "</div>"  S
    S -> "<span>" S "</span>" S
    S -> ""                       
""")

# 5.  Escaped-delimiter grammar
GRAMMARS["brace_escaped_literal"] = CFG.fromstring(r"""
    S -> "\{" S "\}" S
    S -> "{"  S "}"  S
    S -> ""                  
""")

# 6.  Fan-out example (two nested S inside each pair)
GRAMMARS["brace_fanout2"] = make_dyck1("{", "}", recursion="right", fan_out=2)

# 7.  Import *all* v1 grammars so old names remain valid
try:
    from grammar_catalog import GRAMMARS as _V1
    overlap = set(GRAMMARS).intersection(_V1)
    if overlap:
        raise ValueError(f"Name clash between v2 and v1 grammars: {sorted(overlap)[:5]}")
    GRAMMARS.update(_V1)
except ImportError:
    # v1 catalogue not on PYTHONPATH – no problem (just construct dyck-k)
    pass
