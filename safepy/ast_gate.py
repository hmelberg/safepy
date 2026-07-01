"""The static gate: decide, *before running anything*, whether a snippet of user
Python is allowed into the sandbox.

Design stance (see DESIGN.md):

* The gate's job is to kill **code-execution escapes** and **direct row-dumps**,
  not to prove a result is an aggregate. Statistical disclosure is the job of
  the output mediator + ``protect.suppress`` downstream. Two cheap layers beat
  one clever one.
* **Default-deny on node *types*.** Every AST node type must be on
  ``_ALLOWED_NODES`` or the snippet is rejected. New syntax can't sneak in.
* **Default-deny on bare function calls**, allow-listed builtins only.
* **Deny-list on attribute/method names** for the long tail of pandas/polars
  verbs that dump rows or take arbitrary callables. This list is the security
  surface; every entry is a decision, and it is meant to grow.
* **Structural rule:** a snippet is a flat sequence of simple assignments
  followed by one final expression. No loops, defs, lambdas, comprehensions,
  imports, or control flow. This is what makes the tree small enough to reason
  about, and removes most escape vectors for free.

The node-walk discipline mirrors m2py's ``m2py_runtime/exprcompile.py``
(node-by-node against a whitelist, raise on anything outside it).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .errors import ValidationError


# --- node types the gate accepts anywhere -----------------------------------
# Anything not in this set is rejected by _Validator.visit. Note the absence of:
# Lambda, comprehensions, FunctionDef/ClassDef, For/While/With/Try, Import,
# Global/Nonlocal, Delete, Await/Yield, JoinedStr (f-strings), NamedExpr, Slice.
_ALLOWED_NODES: frozenset = frozenset({
    ast.Module, ast.Expr, ast.Assign,
    ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Attribute, ast.Subscript, ast.Call, ast.keyword,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Import, ast.ImportFrom, ast.alias,
})

# Modules that may be imported (only when allow_imports; they resolve to safe
# facades in the runtime, never the real modules).
_IMPORT_WHITELIST: frozenset = frozenset({"lifelines", "numpy", "pandas", "pyfixest"})

# --- bare-name calls allowed (everything else must be lib.method(...)) -------
_SAFE_BUILTINS: frozenset = frozenset({
    "len", "round", "abs", "int", "float", "str", "bool",
})

# --- attribute / method names that are never allowed ------------------------
# Grouped by why. Membership is checked on the *attribute name only* — we do not
# try to know the receiver's type statically.
_DENIED_METHODS: frozenset = frozenset({
    # --- direct row materialisation / export ---
    "head", "tail", "sample", "iloc", "loc", "at", "iat", "xs", "lookup",
    "iterrows", "itertuples", "items", "iteritems", "get", "pop", "squeeze",
    "to_csv", "to_dict", "to_json", "to_numpy", "to_records", "to_list",
    "tolist", "to_clipboard", "to_pickle", "to_parquet", "to_feather",
    "to_excel", "to_string", "to_markdown", "to_latex", "to_html",
    "values", "array", "item", "info", "memory_usage", "glimpse", "row",
    "rows", "get_column", "write_csv", "write_parquet",
    # --- positional / row-identifying reducers: return individual rows ---
    # (max/min/quantile/describe are NOT here: SafeColumn provides safe,
    # order-statistic-checked versions; on a raw frame/OPEN the mediator still
    # refuses the bare scalar/frame they produce.)
    "idxmax", "idxmin", "argmax", "argmin",
    "nlargest", "nsmallest", "first", "last", "nth", "mode", "rank",
    # --- accept arbitrary callables / mini-language code ---
    "apply", "applymap", "map", "transform", "pipe", "agg", "aggregate",
    "query", "eval", "rolling", "expanding", "ewm",
    # --- rendering surface that embeds raw arrays ---
    # 'plot'/'hist'/'boxplot' are intentionally NOT here: plotting is enforced by
    # type instead. It exists only on aggregate results (Released.plot) and, for
    # hist, on a column where it is redirected to a suppressed binned frequency.
    # On a raw SafeFrame there is no .plot, so df.plot() is refused by the facade.
    "style",
})

# --- bare names / calls that are hard bans (sandbox escapes) -----------------
_DENIED_NAMES: frozenset = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "hasattr", "vars", "globals", "locals",
    "dir", "type", "object", "super", "memoryview", "breakpoint", "exit", "quit",
    "help", "id", "format",
})


@dataclass
class GateResult:
    ok: bool
    names_assigned: list[str]
    calls: list[str]            # attribute/method names invoked, for the audit
    error: ValidationError | None = None


class _Stop(Exception):
    pass


class _Validator(ast.NodeVisitor):
    def __init__(self, allowed_names: frozenset[str], allow_imports: bool = False):
        self.allowed_names = allowed_names
        self.allow_imports = allow_imports
        self.assigned: list[str] = []
        self.imported: set[str] = set()
        self.calls: list[str] = []
        self.error: ValidationError | None = None

    # -- helpers --
    def _fail(self, node, kind, msg, token=None):
        self.error = ValidationError(
            msg, kind=kind, line=getattr(node, "lineno", None), token=token)
        raise _Stop

    # -- the default-deny gate on node types --
    def visit(self, node):
        if type(node) not in _ALLOWED_NODES:
            self._fail(node, "syntax",
                       f"{type(node).__name__} is not allowed in safepy")
        return super().visit(node)  # dispatches to visit_X then generic_visit

    # -- structural rule: assignments then a final expression --
    def visit_Module(self, node: ast.Module):
        if not node.body:
            self._fail(node, "empty", "empty program")
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self._check_assign(stmt)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                pass  # validated in visit_Import / visit_ImportFrom
            elif isinstance(stmt, ast.Expr):
                # An intermediate bare expression (e.g. `cph.fit(...)`) has its
                # value discarded, like normal Python; only the final expression
                # becomes the mediated result. Harmless for disclosure.
                pass
            else:
                self._fail(stmt, "structure",
                           f"top-level {type(stmt).__name__} is not allowed; "
                           "use simple assignments and expressions")
        # A script may end on an assignment (build datasets -> catalog only) or on
        # one/several bare expressions (each a released result). No end-in-expr rule.
        self.generic_visit(node)

    def _check_assign(self, stmt: ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            self._fail(stmt, "structure",
                       "assignment must bind a single plain name")
        name = stmt.targets[0].id
        if name.startswith("_"):
            self._fail(stmt, "name", f"names may not start with '_': {name}", token=name)
        self.assigned.append(name)

    # -- expression-level checks --
    def visit_Name(self, node: ast.Name):
        if node.id.startswith("_"):
            self._fail(node, "name",
                       f"dunder/private names are not allowed: {node.id}", token=node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr = node.attr
        if attr.startswith("_"):
            self._fail(node, "attribute",
                       f"access to dunder/private attribute '{attr}' is not allowed",
                       token=attr)
        if attr in _DENIED_METHODS:
            self._fail(node, "attribute",
                       f"'{attr}' is not allowed: it can reveal individual rows "
                       "or run arbitrary code", token=attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _DENIED_NAMES:
                self._fail(node, "call",
                           f"calling '{func.id}' is not allowed", token=func.id)
            if (func.id not in _SAFE_BUILTINS and func.id not in self.allowed_names
                    and func.id not in self.imported):
                self._fail(node, "call",
                           f"unknown function '{func.id}'; only library handles "
                           f"and {sorted(_SAFE_BUILTINS)} may be called by name",
                           token=func.id)
        elif isinstance(func, ast.Attribute):
            self.calls.append(func.attr)  # _DENIED_METHODS already enforced in visit_Attribute
        else:
            self._fail(node, "call", "only name(...) and obj.method(...) calls are allowed")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        if not self.allow_imports:
            self._fail(node, "import", "imports are not allowed here")
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in _IMPORT_WHITELIST:
                self._fail(node, "import",
                           f"module '{alias.name}' is not available in safepy",
                           token=alias.name)
            self.imported.add(alias.asname or root)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not self.allow_imports:
            self._fail(node, "import", "imports are not allowed here")
        root = (node.module or "").split(".")[0]
        if root not in _IMPORT_WHITELIST:
            self._fail(node, "import",
                       f"module '{node.module}' is not available in safepy",
                       token=node.module)
        for alias in node.names:
            self.imported.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Allow column selection: df["col"], df[["a","b"]], df[mask-expression].
        # Forbid positional/row indexing: df[0], df[1:5].
        s = node.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, (int, bool)):
            self._fail(node, "subscript",
                       "positional indexing (df[<int>]) is not allowed; select by "
                       "column name or boolean mask")
        # ast.Slice is not in _ALLOWED_NODES, so df[1:5] is already rejected.
        self.generic_visit(node)


def validate(code: str, *, allowed_names: frozenset[str],
             allow_imports: bool = False) -> GateResult:
    """Static-check ``code``. Returns a GateResult; never executes anything."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return GateResult(False, [], [],
                          ValidationError(f"could not parse: {exc.msg}",
                                          kind="parse", line=exc.lineno))
    v = _Validator(allowed_names, allow_imports=allow_imports)
    try:
        v.visit(tree)
    except _Stop:
        return GateResult(False, v.assigned, v.calls, v.error)
    return GateResult(True, v.assigned, v.calls, None)
