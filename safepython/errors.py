"""Exceptions for safepython.

Two rules govern every message here:

1. A message must NEVER embed a data value. Exception text is an exfiltration
   channel (``KeyError: 'Ola Nordmann'`` leaks an individual). The sandbox
   sanitises any exception coming out of user code into a generic message and
   keeps the original only in the server-side audit log.
2. Validation messages may name *code* (a verb, a node type, a column name the
   user typed) but never *data*.
"""

from __future__ import annotations


class SafePythonError(Exception):
    """Base class for everything safepython raises."""


class ValidationError(SafePythonError):
    """User code failed the static AST gate. Safe to show the user verbatim."""

    def __init__(self, message: str, *, kind: str = "validation",
                 line: int | None = None, token: str | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.line = line
        self.token = token

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message,
                "line": self.line, "token": self.token}


class SandboxError(SafePythonError):
    """User code passed the gate but raised at runtime. The message shown to the
    user is generic; the original lives in the audit record only."""


class DisclosureError(SafePythonError):
    """The output mediator refused to release a result (could not prove it is an
    aggregate, or no group counts available to apply suppression)."""
