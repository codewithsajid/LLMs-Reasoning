"""
Comprehensive CFG catalog (NLTK‑compatible):
  • 27 brace‑centric grammars
  • 13 additional challenge grammars
"""
from __future__ import annotations            
from typing import Dict                       
from nltk.grammar import CFG

GRAMMARS: Dict[str, CFG] = {
    # ────────── brace‑centric grammars ──────────

    "braces_simple": CFG.fromstring(r"""
        S -> "{" S "}" S
        S -> ""
    """),

    "braces_seq": CFG.fromstring(r"""
        S -> "{" "}" S
        S -> ""
    """),

    "braces_ambiguous": CFG.fromstring(r"""
        S -> "{" S "}"
        S -> S S
        S -> ""
    """),

    "nested_two_levels": CFG.fromstring(r"""
        S -> "{" "}" S
        S -> "{" "{" "}" "}" S
        S -> ""
    """),

    "nested_three_levels": CFG.fromstring(r"""
        S -> "{" "}" S
        S -> "{" "{" "}" "}" S
        S -> "{" "{" "{" "}" "}" "}" S
        S -> ""
    """),

    "nested_four_levels": CFG.fromstring(r"""
        S -> "{" "}" S
        S -> "{" "{" "}" "}" S
        S -> "{" "{" "{" "}" "}" "}" S
        S -> "{" "{" "{" "{" "}" "}" "}" "}" S
        S -> ""
    """),

    "nested_five_levels": CFG.fromstring(r"""
        S -> "{" "}" S
        S -> "{" "{" "}" "}" S
        S -> "{" "{" "{" "}" "}" "}" S
        S -> "{" "{" "{" "{" "}" "}" "}" "}" S
        S -> "{" "{" "{" "{" "{" "}" "}" "}" "}" "}" S
        S -> ""
    """),

    "braces_tree": CFG.fromstring(r"""
        S -> "{" Children "}"
        Children -> S Children
        Children -> ""
    """),

    "braces_with_content": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" C "}"
        C -> "x" C
        C -> "y" C
        C -> ""
    """),

    "braces_with_digits": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" D "}"
        D -> "0" D
        D -> "1" D
        D -> ""
    """),

    "braces_with_digits_full": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" Num "}"
        Num -> Dig Num
        Num -> Dig
        Dig -> "0"
        Dig -> "1"
        Dig -> "2"
        Dig -> "3"
        Dig -> "4"
        Dig -> "5"
        Dig -> "6"
        Dig -> "7"
        Dig -> "8"
        Dig -> "9"
    """),

    "braces_with_labels": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" Label "}"
        Label -> "id"
        Label -> "name"
        Label -> "key"
    """),

    "mixed_braces_paren": CFG.fromstring(r"""
        S -> "{" S "}"
        S -> "(" S ")"
        S -> S S
        S -> ""
    """),

    "mixed_all_brackets": CFG.fromstring(r"""
        S -> "[" S "]"
        S -> "(" S ")"
        S -> "{" S "}"
        S -> S S
        S -> ""
    """),

    "braces_list_semicolon": CFG.fromstring(r"""
        S -> Block ";" S
        S -> Block
        Block -> "{" S "}"
    """),

    "braces_list_comma": CFG.fromstring(r"""
        S -> Block "," S
        S -> Block
        Block -> "{" S "}"
    """),

    "braces_json_simple": CFG.fromstring(r"""
        S -> "{" Members "}"
        Members -> Pair
        Members -> Pair "," Members
        Pair -> Key ":" Value
        Key -> "id"
        Key -> "name"
        Value -> "foo"
        Value -> "bar"
    """),

    "braces_nested_json": CFG.fromstring(r"""
        S -> "{" Members "}"
        Members -> Pair
        Members -> Pair "," Members
        Members -> ""
        Pair -> Key ":" Val
        Val -> Value
        Val -> S
        Key -> "id"
        Key -> "name"
        Key -> "flag"
        Value -> "foo"
        Value -> "bar"
        Value -> "baz"
    """),

    "braces_code_block": CFG.fromstring(r"""
        S -> FunctionDecl
        FunctionDecl -> "function" "main" "(" ")" Block
        Block -> "{" Body "}"
        Body -> Stmt Body
        Body -> ""
        Stmt -> "return" Expr ";"
        Expr -> Num
        Expr -> Var
        Num -> "0"
        Num -> "1"
        Var -> "x"
        Var -> "y"
    """),

    "braces_with_comments": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" Comment "}"
        Comment -> "/*" Txt "*/"
        Txt -> "a" Txt
        Txt -> "b" Txt
        Txt -> ""
    """),

    "braces_template_tags": CFG.fromstring(r"""
        S -> Tmpl S
        S -> ""
        Tmpl -> "{{" Var "}}"
        Var -> "x"
        Var -> "y"
        Var -> "z"
    """),

    "braces_alternating_pairs": CFG.fromstring(r"""
        S -> "{" Alt "}" S
        S -> ""
        Alt -> "(" ")" "{" "}" Alt
        Alt -> ""
    """),

    "braces_if_else_block": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" Stmts "}"
        Stmts -> If Stmts
        Stmts -> ""
        If -> "if" "(" "cond" ")" Block ElseOpt
        ElseOpt -> "else" Block
        ElseOpt -> ""
    """),

    "braces_with_html": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" H "}"
        H -> "<b>" S "</b>"
        H -> "<i>" S "</i>"
    """),

    "braces_paren_then_brace": CFG.fromstring(r"""
        S -> "{" S "}"
        S -> "{" "(" S ")" "}"
        S -> S S
        S -> ""
    """),

    "braces_letters": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" Letters "}"
        Letters -> L Letters
        Letters -> L
        L -> "a"
        L -> "b"
        L -> "c"
    """),

    "braces_kv_numeric": CFG.fromstring(r"""
        S -> Block S
        S -> ""
        Block -> "{" KVList "}"
        KVList -> KV
        KVList -> KV "," KVList
        KV -> Key ":" Num
        Key -> "id"
        Key -> "age"
        Key -> "score"
        Num -> "0"
        Num -> "1"
        Num -> "2"
        Num -> "3"
    """),

    # ────────── additional challenge grammars ──────────

    "arith_ambiguous": CFG.fromstring(r"""
        E -> E "+" E
        E -> E "*" E
        E -> "(" E ")"
        E -> "n"
    """),

    "arith_unambiguous": CFG.fromstring(r"""
        E -> E "+" T
        E -> T
        T -> T "*" F
        T -> F
        F -> "(" E ")"
        F -> "n"
    """),

    "dangling_else": CFG.fromstring(r"""
        S -> "if" C "then" S
        S -> "if" C "then" S "else" S
        S -> "stmt"
        C -> "cond"
    """),

    "html_tags": CFG.fromstring(r"""
        S -> "<b>" S "</b>"
        S -> "<i>" S "</i>"
        S -> S S
        S -> ""
    """),

    "ambiguous_palindrome": CFG.fromstring(r"""
        S -> "a" S
        S -> S "a"
        S -> "b" S
        S -> S "b"
        S -> ""
    """),

    "palindrome_unambiguous": CFG.fromstring(r"""
        S -> "a" S "a"
        S -> "b" S "b"
        S -> "a"
        S -> "b"
        S -> ""
    """),

    "copy_ab": CFG.fromstring(r"""
        S -> "a" S "b"
        S -> ""
    """),

    "copy_ab_ba_amb": CFG.fromstring(r"""
        S -> "a" S "b"
        S -> "b" S "a"
        S -> ""
    """),

    "comma_list": CFG.fromstring(r"""
        L -> I
        L -> I "," L
        I -> "a"
        I -> "b"
    """),

    "binary_numbers": CFG.fromstring(r"""
        B -> "0" B
        B -> "1" B
        B -> ""
    """),

    "xml_like": CFG.fromstring(r"""
        S -> "<tag>" S "</tag>"
        S -> S S
        S -> ""
    """),

    "xml_with_attrs": CFG.fromstring(r"""
        S -> "<tag" A ">" S "</tag>"
        S -> ""
        A -> Att
        A -> Att A
        Att -> "id" "=" V
        Att -> "class" "=" V
        V -> "0"
        V -> "1"
        V -> "x"
    """),

    "css_simple": CFG.fromstring(r"""
        S -> Sel "{" Rules "}" S
        S -> ""
        Sel -> "div"
        Sel -> "span"
        Sel -> "#header"
        Sel -> ".active"
        Rules -> R
        Rules -> R Rules
        R -> Prop ":" Val ";"
        Prop -> "color"
        Prop -> "font-size"
        Val -> "red"
        Val -> "12px"
    """),
}
