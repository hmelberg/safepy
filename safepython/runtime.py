"""Restricted execution of already-gated code.

Preconditions: ``code`` has passed :func:`safepython.ast_gate.validate`. The
gate guarantees the structural shape (simple assignments + final expression) and
that no banned node/verb/name is present, so this module only has to:

* build a namespace with ``__builtins__`` stripped to a tiny safe set,
* bind the library handles and the private data sources,
* exec the assignment prefix, eval the final expression, and
* sanitise any exception so it cannot carry a data value to the user.

This is defence-in-depth, not the primary guard. We never rely on
``__builtins__`` stripping alone to contain untrusted Python — the gate is what
makes the input trustworthy; this just narrows the blast radius further.
"""

from __future__ import annotations

import ast

from .errors import SandboxError

# A minimal, audited builtin surface. Mirrors ast_gate._SAFE_BUILTINS plus the
# constants/exceptions ordinary expressions need. No import, eval, open, getattr.
import builtins as _b

_SAFE_BUILTINS = {
    name: getattr(_b, name)
    for name in ("len", "round", "abs", "int", "float", "str", "bool",
                 "True", "False", "None")
    if hasattr(_b, name)
}


def execute(code: str, namespace: dict):
    """Run gated ``code`` in ``namespace`` and return the final expression's value.

    ``namespace`` should contain the library handles (``pd``, ``np``, ...) and
    the data sources. ``__builtins__`` is overwritten here.
    """
    ns = dict(namespace)
    ns["__builtins__"] = _SAFE_BUILTINS

    tree = ast.parse(code, mode="exec")
    *prefix, last = tree.body  # gate guarantees last is an ast.Expr

    try:
        if prefix:
            exec(compile(ast.Module(body=prefix, type_ignores=[]), "<safepython>", "exec"), ns)
        result = eval(compile(ast.Expression(body=last.value), "<safepython>", "eval"), ns)
    except SandboxError:
        raise
    except BaseException as exc:  # noqa: BLE001 - deliberately broad; message is dropped
        # Do NOT include str(exc): it may contain a data value (KeyError on a
        # name, an assertion message, a formatted row). Keep only the type for
        # the audit; show the user nothing data-bearing.
        raise SandboxError(
            f"your code raised {type(exc).__name__} during execution"
        ) from None
    return result
