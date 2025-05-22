#!/usr/bin/env python3
"""
derivation_engine.py  –  Trace generator, contrastive-trace injectors (for uncorrect types), parse-tree builder
========================================================================================
----------
gen_derivations(grammar: CFG,
                strategy: {"leftmost","rightmost","random"} = "leftmost",
                max_depth: int = 30,
                seed: int | None = None)
    -> iterator[TraceRecord]

inject_partial(trace: list[TraceStep], k: int) -> list[TraceStep]
inject_single_error(grammar: CFG, trace: list[TraceStep],
                    idx: int, mode: {"wrong_rule","off_by_index","hallucinated"})
                    -> list[TraceStep]

trace_to_tree(trace: list[TraceStep]) -> str
"""

from __future__ import annotations
import random
from typing import List, Dict, Iterator, TypedDict, Union
from nltk.grammar import Nonterminal, Production, CFG
from nltk.tree import Tree

# ────────────────────────────── types ─────────────────────────────────────
Symbol = Union[str, Nonterminal]


class TraceStep(TypedDict):
    step: int
    current_state: str
    rule_applied: str
    result_state: str
    applied_at_index: int


class TraceRecord(TypedDict):
    final_string: str
    length: int
    derivation: List[TraceStep]


# ────────────────────── helper (token → printable str) ────────────────────
def _tok_str(seq: List[Symbol]) -> str:
    """Render a list of terminals / non-terminals into a plain string."""
    return "".join(str(x) for x in seq)


# ─────────────────── correct-derivation generator ─────────────────────────
def gen_derivations(
    grammar: CFG,
    strategy: str = "leftmost",
    max_depth: int = 30,
    *,
    seed: int | None = None,
) -> Iterator[TraceRecord]:
    """
    Yield **unique** derivations (traces) up to ``max_depth`` rule applications.

    Parameters
    ----------
    grammar : CFG
        Any NLTK grammar. Its start symbol is taken from ``grammar.start()``.
    strategy : str
        'leftmost', 'rightmost', or 'random' choice of non-terminal at each step.
    max_depth : int
        Maximum number of rule applications in a trace (root included).
    seed : int | None
        Optional RNG seed for reproducibility across runs.

    Yields
    ------
    TraceRecord
        A dict with keys ``final_string``, ``length``, and ``derivation``.
    """
    if seed is not None:
        random.seed(seed)

    seen: set[str] = set()            # avoid yielding duplicate strings
    start_sym = grammar.start()

    # iterative deepening DFS keeps memory usage bounded
    for depth_limit in range(1, max_depth + 1):
        stack: List[tuple[List[Symbol], List[TraceStep], int]] = [
            ([start_sym], [], 1)
        ]  # (frontier, trace_so_far, next_step_id)

        while stack:
            frontier, trace, step_id = stack.pop()

            # terminal state?
            if not any(isinstance(s, Nonterminal) for s in frontier):
                term_string = _tok_str(frontier)
                if term_string not in seen:
                    seen.add(term_string)
                    yield {
                        "final_string": term_string,
                        "length": len(frontier),
                        "derivation": trace,
                    }
                continue

            if len(trace) >= depth_limit:
                continue  # reached this iteration’s cutoff

            # choose NT index per strategy
            nt_indices = [i for i, s in enumerate(frontier) if isinstance(s, Nonterminal)]
            if strategy == "leftmost":
                idx = nt_indices[0]
            elif strategy == "rightmost":
                idx = nt_indices[-1]
            else:  # random
                idx = random.choice(nt_indices)

            # expand productions – shuffled each iteration for variety
            prods = list(grammar.productions(lhs=frontier[idx]))
            random.shuffle(prods)

            for prod in prods:
                next_frontier = (
                    frontier[:idx] + list(prod.rhs()) + frontier[idx + 1 :]
                )
                rule_str = f"{prod.lhs()} -> {' '.join(map(str, prod.rhs())) or 'ε'}"
                trace_step: TraceStep = {
                    "step": step_id,
                    "current_state": _tok_str(frontier),
                    "rule_applied": rule_str,
                    "result_state": _tok_str(next_frontier),
                    "applied_at_index": idx,
                }
                stack.append((next_frontier, trace + [trace_step], step_id + 1))


# ───────────────────── contrastive-trace injectors (uncorrect types) ────────────────────────
def inject_partial(trace: List[TraceStep], k: int) -> List[TraceStep]:
    """
    Truncate a trace after ``k`` steps and append a sentinel <STOP> step.
    """
    assert 1 <= k < len(trace), "k must be inside the trace (1-based indexing)"
    last = trace[k - 1]["result_state"]
    return trace[:k] + [
        {
            "step": k + 1,
            "current_state": last,
            "rule_applied": "<STOP>",
            "result_state": last,
            "applied_at_index": -1,
        }
    ]


def inject_single_error(
    grammar: CFG,
    trace: List[TraceStep],
    idx: int,
    mode: str = "wrong_rule",
) -> List[TraceStep]:
    """
    Mutate *exactly one* step inside ``trace`` to create an ill-formed variant.

    Parameters
    ----------
    grammar : CFG
        Needed when ``mode == "wrong_rule"`` to sample an alternative production.
    trace : List[TraceStep]
        Original (correct) trace.
    idx : int
        1-based position of the step to corrupt.
    mode : str
        'wrong_rule'      – replace the production with a random alternative  
        'off_by_index'    – increment applied_at_index (if still valid)  
        'hallucinated'    – replace the rule string with garbage
    """
    idx = max(1, min(idx, len(trace)))  # clamp → never out of range
    new: List[TraceStep] = []

    for i, rec in enumerate(trace, 1):
        if i != idx:
            new.append(rec)
            continue

        bogus = rec.copy()
        lhs = bogus["rule_applied"].split("->", 1)[0].strip()

        if mode == "wrong_rule":
            alt = random.choice(grammar.productions(lhs=Nonterminal(lhs)))
            bogus["rule_applied"] = (
                f"{alt.lhs()} -> {' '.join(map(str, alt.rhs())) or 'ε'}"
            )

        elif mode == "off_by_index":
            if bogus["applied_at_index"] >= 0:
                bogus["applied_at_index"] += 1

        elif mode == "hallucinated":
            bogus["rule_applied"] = f"{lhs} -> $$$"

        new.append(bogus)

    return new


# ───────────────────── parse-tree reconstruction (optional for now) ──────────────────────────
def trace_to_tree(trace: List[TraceStep]) -> str:
    """
    Rebuild an NLTK ``Tree`` from a (possibly partial) trace.

    Returns
    -------
    str
        A flat-printed bracketed representation (one line).
    Raises
    ------
    AssertionError
        If reconstruction fails (e.g. trace structurally inconsistent).
    """
    if not trace:
        raise AssertionError("empty trace")

    # The initial symbol is always grammar.start(), but traces may begin later
    start_sym = trace[0]["rule_applied"].split("->")[0].strip()
    frontier: List[Union[Symbol, Tree]] = [Nonterminal(start_sym)]

    for rec in trace:
        if rec["rule_applied"] == "<STOP>":
            break

        idx = rec["applied_at_index"]
        if idx >= len(frontier):
            raise AssertionError("applied_at_index exceeds frontier length")

        lhs = rec["rule_applied"].split("->")[0].strip()
        rhs = rec["rule_applied"].split("->", 1)[1].strip()

        # verify front symbol matches LHS unless the trace was corrupted
        if isinstance(frontier[idx], Nonterminal) and str(frontier[idx]) != lhs:
            raise AssertionError("frontier/LHS mismatch during reconstruction")

        rhs_syms = [] if rhs == "ε" else rhs.split()
        children: List[Symbol] = [
            Nonterminal(tok) if tok[:1].isupper() else tok for tok in rhs_syms
        ]
        node = Tree(lhs, children)

        # splice the new node into the front, then *flatten* the front
        frontier = frontier[:idx] + [node] + frontier[idx + 1 :]
        frontier = [
            kid for sym in frontier for kid in (sym if isinstance(sym, Tree) else [sym])
        ]

    # first real Tree in the frontier is our root
    for itm in frontier:
        if isinstance(itm, Tree):
            return itm.pformat(margin=99999)

    raise AssertionError("parse reconstruction failed")
