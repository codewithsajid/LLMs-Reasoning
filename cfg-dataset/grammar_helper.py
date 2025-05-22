#!/usr/bin/env python3
"""
grammar_helper.py  –  Utility builders for bracket (Dyck) grammars
==================================================================

--------------
make_token_pair(open_tok, close_tok)
make_dyck1(open_tok='{', close_tok='}', recursion='right', fan_out=1)
make_dyck_k(pairs, recursion='right', fan_out=1)

Dyck-k is the language of *balanced* strings over k distinct bracket types.
"""

from __future__ import annotations
from itertools import repeat
from typing import List, Tuple

from nltk.grammar import CFG, Nonterminal, Production

# ───────────────────────── helpers ──────────────────────────────
def _to_sym(tok: str):
    """Return Nonterminal if the token name is uppercase, else leave as terminal."""
    return Nonterminal(tok) if tok[:1].isupper() else tok


def make_token_pair(open_tok: str, close_tok: str) -> Tuple[str, str]:
    """Validate and return a (open, close) token pair."""
    if not open_tok or not close_tok or open_tok == close_tok:
        raise ValueError("Bracket tokens must be two *different* non-empty strings")
    return open_tok, close_tok


# ───────────────────────── single-type Dyck (Dyck-1) ──────────────────────
def _right_recursive(open_tok: str, close_tok: str, fan_out: int) -> List[Production]:
    """S → open S^fan_out close S | ε"""
    S = Nonterminal("S")
    rhs = [open_tok] + list(repeat(S, fan_out)) + [close_tok, S]
    return [Production(S, rhs), Production(S, [])]


def _left_recursive(open_tok: str, close_tok: str, fan_out: int) -> List[Production]:
    """S → S open S^fan_out close | ε"""
    S = Nonterminal("S")
    rhs = [S, open_tok] + list(repeat(S, fan_out)) + [close_tok]
    return [Production(S, rhs), Production(S, [])]


def _mixed_recursive(open_tok: str, close_tok: str, fan_out: int) -> List[Production]:
    """Ambiguous: S → open S^fan_out close | S S | ε"""
    S = Nonterminal("S")
    rhs_balanced = [open_tok] + list(repeat(S, fan_out)) + [close_tok]
    return [
        Production(S, rhs_balanced),
        Production(S, [S, S]),
        Production(S, []),
    ]


def make_dyck1(
    open_tok: str = "{",
    close_tok: str = "}",
    *,
    recursion: str = "right",
    fan_out: int = 1,
) -> CFG:
    """Return an NLTK CFG for a single-bracket-type (Dyck-1) language."""
    if fan_out < 1:
        raise ValueError("fan_out must be ≥ 1")

    open_tok, close_tok = make_token_pair(open_tok, close_tok)

    if recursion == "right":
        prods = _right_recursive(open_tok, close_tok, fan_out)
    elif recursion == "left":
        prods = _left_recursive(open_tok, close_tok, fan_out)
    elif recursion == "mixed":
        prods = _mixed_recursive(open_tok, close_tok, fan_out)
    else:
        raise ValueError("recursion must be 'right', 'left', or 'mixed'")

    return CFG(Nonterminal("S"), prods)


# ───────────────────────── multi-type Dyck (Dyck-k) ───────────────────────
def make_dyck_k(
    pairs: List[Tuple[str, str]] | None = None,
    *,
    recursion: str = "right",
    fan_out: int = 1,
) -> CFG:
    """
    Build a balanced-bracket grammar for **k** distinct bracket types.

    Parameters
    ----------
    pairs : list[(open, close)]
        List of token pairs.  If None, uses the three classic ASCII pairs.
    recursion : {'right','left','mixed'}
        Same semantics as in `make_dyck1`.
    fan_out : int ≥ 1
        Number of nested S’s inside each matching pair (≥ 1).

    Example
    -------
    >>> g = make_dyck_k([("(", ")"), ("{", "}")], recursion="mixed")
    """
    if fan_out < 1:
        raise ValueError("fan_out must be ≥ 1")

    if pairs is None:
        pairs = [("(", ")"), ("{", "}"), ("[", "]")]

    S = Nonterminal("S")
    prods: List[Production] = [Production(S, [])]          # S -> ε

    for open_tok, close_tok in pairs:
        open_tok, close_tok = make_token_pair(open_tok, close_tok)

        if recursion == "right":
            rhs = [open_tok] + list(repeat(S, fan_out)) + [close_tok, S]
            prods.append(Production(S, rhs))
        elif recursion == "left":
            rhs = [S, open_tok] + list(repeat(S, fan_out)) + [close_tok]
            prods.append(Production(S, rhs))
        elif recursion == "mixed":
            # one fully balanced rule plus the S S (concatenation) rule
            rhs_bal = [open_tok] + list(repeat(S, fan_out)) + [close_tok]
            prods.append(Production(S, rhs_bal))
        else:
            raise ValueError("recursion must be 'right', 'left', or 'mixed'")

    if recursion == "mixed":
        prods.append(Production(S, [S, S]))

    return CFG(S, prods)


# what gets imported by `from grammar_helper import *`
__all__ = [
    "make_token_pair",
    "make_dyck1",
    "make_dyck_k",
]
