"""A small R *expression* parser + evaluator for the R dialect.

Hand-rolled (no dependency, auditable, **default-deny**): only the tokens,
operators and functions implemented here are recognised — anything else raises.
Expressions are evaluated against the ``SafeColumn`` algebra the STRICT facade
already exposes, so an R expression like ``ifelse(log(salary) > 10, 'hi', 'lo')``
becomes safe derived columns/masks that can only exit via a suppressed aggregate.
R is never executed; this only *parses* text and drives the facade.

Grammar (precedence low→high): ``| || ; & && ; ! (prefix) ; == != < <= > >= %in% ;
+ - ; * / ; %% ; unary - + ; ^ (right) ; atoms`` where an atom is a number,
string, ``TRUE``/``FALSE``/``NA``, a bare name (→ column), ``frame$col``, a
parenthesised expression, ``c(...)``, or a whitelisted ``fn(...)`` call.
"""

from __future__ import annotations

import re

import numpy as np

from .errors import DisclosureError, ValidationError

_TOKEN_RE = re.compile(r"""
      (?P<WS>\s+)
    | (?P<NUM>\d+\.\d+|\.\d+|\d+)
    | (?P<STR>"[^"]*"|'[^']*')
    | (?P<INOP>%in%)
    | (?P<MOD>%%)
    | (?P<OP><=|>=|==|!=|&&|\|\||[-+*/^<>!&|$(),~])
    | (?P<NAME>[A-Za-z_.][A-Za-z0-9_.]*)
""", re.X)

# binary left-binding powers; higher binds tighter. '^' is right-associative.
# '~' (formula, lowest) is only meaningful inside case_when; elsewhere it errors.
_LBP = {
    "~": 5,
    "||": 10, "|": 10, "&&": 20, "&": 20,
    "==": 30, "!=": 30, "<": 30, "<=": 30, ">": 30, ">=": 30, "%in%": 30,
    "+": 40, "-": 40, "*": 50, "/": 50, "%%": 60, "^": 70,
}
_RIGHT = {"^"}
_KEYWORDS = {"TRUE": True, "T": True, "FALSE": False, "F": False}


def _tokenize(s: str):
    toks, i = [], 0
    while i < len(s):
        m = _TOKEN_RE.match(s, i)
        if not m:
            raise ValidationError(f"unexpected character in R expression: {s[i]!r}",
                                  kind="syntax")
        i = m.end()
        if m.lastgroup == "WS":
            continue
        toks.append((m.lastgroup, m.group()))
    toks.append(("EOF", ""))
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks, self.pos = toks, 0

    def _peek(self):
        return self.toks[self.pos]

    def _next(self):
        t = self.toks[self.pos]; self.pos += 1; return t

    def _expect(self, text):
        k, v = self._next()
        if v != text:
            raise ValidationError(f"expected {text!r} in R expression, got {v!r}",
                                  kind="syntax")

    def parse(self):
        node = self._expr(0)
        if self._peek()[0] != "EOF":
            raise ValidationError(f"trailing tokens in R expression: {self._peek()[1]!r}",
                                  kind="syntax")
        return node

    def _expr(self, min_bp):
        left = self._prefix()
        while True:
            op = self._peek()[1]
            lbp = _LBP.get(op, 0)
            if lbp <= min_bp:
                break
            self._next()
            right = self._expr(lbp - 1 if op in _RIGHT else lbp)
            left = ("binary", op, left, right)
        return left

    def _prefix(self):
        kind, val = self._next()
        if kind == "NUM":
            return ("num", float(val) if ("." in val) else int(val))
        if kind == "STR":
            return ("str", val[1:-1])
        if val == "(":
            node = self._expr(0); self._expect(")"); return node
        if val == "-":
            return ("unary", "-", self._expr(65))
        if val == "+":
            return self._expr(65)
        if val == "!":
            return ("unary", "!", self._expr(15))
        if kind == "NAME":
            if val in _KEYWORDS:
                return ("bool", _KEYWORDS[val])
            if val in ("NA", "NULL", "NaN"):
                return ("na",)
            if self._peek()[1] == "(":
                self._next()
                args = self._args()
                return ("call", val, args)
            if self._peek()[1] == "$":            # frame$col
                self._next()
                ck, cv = self._next()
                if ck != "NAME":
                    raise ValidationError("expected a column name after '$'", kind="syntax")
                return ("dollar", val, cv)
            return ("name", val)
        raise ValidationError(f"unexpected token in R expression: {val!r}", kind="syntax")

    def _args(self):
        args = []
        if self._peek()[1] == ")":
            self._next(); return args
        while True:
            args.append(self._expr(0))
            v = self._next()[1]
            if v == ")":
                return args
            if v != ",":
                raise ValidationError(f"expected ',' or ')' in call, got {v!r}",
                                      kind="syntax")


def parse(expr: str):
    """Parse an R expression string into an AST (a tuple tree)."""
    return _Parser(_tokenize(expr)).parse()


# ── evaluation against the SafeColumn algebra ────────────────────────────────
_BINOPS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / b,
    "^": lambda a, b: a ** b, "%%": lambda a, b: a % b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "&": lambda a, b: a & b, "&&": lambda a, b: a & b,
    "|": lambda a, b: a | b, "||": lambda a, b: a | b,
}


def _from_safecol(x):
    from .safeframe import SafeColumn
    return isinstance(x, SafeColumn)


def eval_expr(node, sf):
    """Evaluate a parsed R expression against ``sf`` (a ``SafeFrame``). Bare names
    resolve to columns (dplyr NSE); the result is a ``SafeColumn`` (derived column
    or boolean mask) or a Python scalar."""
    from .namespaces import SafeNp

    def ev(n):
        k = n[0]
        if k == "num" or k == "str" or k == "bool":
            return n[1]
        if k == "na":
            return np.nan
        if k == "name":
            return sf[n[1]]                 # SafeColumn (raises on unknown column)
        if k == "dollar":
            return sf[n[2]]
        if k == "unary":
            v = ev(n[2])
            if n[1] == "-":
                return -v
            return (~v) if _from_safecol(v) else (not v)
        if k == "binary":
            op = n[1]
            if op == "~":
                raise DisclosureError("'~' is only valid inside case_when(...)")
            a, b = ev(n[2]), ev(n[3])
            if op == "%in%":
                if not _from_safecol(a):
                    raise DisclosureError("%in% needs a column on the left")
                return a.isin(b if isinstance(b, (list, tuple)) else [b])
            if op not in _BINOPS:
                raise DisclosureError(f"operator {op!r} is not allowed here")
            return _BINOPS[op](a, b)
        if k == "call":
            if n[1] == "case_when":
                return _case_when(n[2], ev)
            return _call(n[1], [ev(a) for a in n[2]], SafeNp())
        raise ValidationError("unsupported R expression element", kind="syntax")

    return ev(node)


def _case_when(arg_nodes, ev):
    """``case_when(cond1 ~ v1, cond2 ~ v2, TRUE ~ default)`` — a first-match
    vectorised recode. Built inside-out (last clause first) so earlier clauses
    take priority; an unmatched row is NA when there is no ``TRUE ~`` default."""
    from .namespaces import SafeNp
    npf = SafeNp()
    result = np.nan
    for node in reversed(arg_nodes):
        if node[0] != "binary" or node[1] != "~":
            raise ValidationError(
                "case_when needs 'condition ~ value' clauses", kind="syntax")
        cond_node, val_node = node[2], node[3]
        val = ev(val_node)
        if cond_node == ("bool", True):        # TRUE ~ default
            result = val
        else:
            result = npf.where(ev(cond_node), val, result)
    return result


def _call(name, args, npf):
    """Dispatch a whitelisted R function to the safe facade. Unknown functions are
    refused (default-deny) — this is the security surface for R expressions."""
    if name == "c":
        return list(args)
    if name in ("log", "log2", "log10", "exp", "sqrt", "abs", "sign",
                "floor", "ceiling"):
        fn = {"ceiling": "ceil"}.get(name, name)
        return getattr(npf, fn)(args[0])
    if name == "round":
        return npf.round(args[0], int(args[1]) if len(args) > 1 else 0)
    if name in ("ifelse", "if_else"):
        if len(args) != 3:
            raise ValidationError("ifelse needs (cond, yes, no)", kind="syntax")
        return npf.where(args[0], args[1], args[2])
    if name in ("is.na", "is_na"):
        return args[0].isna()
    if name == "as.numeric":
        return args[0].astype("float64")
    if name == "as.integer":
        return args[0].astype("int64")
    if name == "as.character":
        return args[0].astype("str")
    if name == "as.logical":
        return args[0].astype("bool")
    if name == "toupper":
        return args[0].str.upper()
    if name == "tolower":
        return args[0].str.lower()
    if name == "nchar":
        return args[0].str.len()
    if name == "substr":                    # R: 1-indexed, inclusive
        return args[0].str.slice(int(args[1]) - 1, int(args[2]))
    raise DisclosureError(
        f"R function '{name}' is not available in safepy's R dialect")
