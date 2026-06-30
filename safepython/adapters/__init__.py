"""Adapter registry. Importing this package registers the built-in adapters."""

from . import base  # noqa: F401
from . import pandas_adapter  # noqa: F401  (registers PandasAdapter on import)
from . import safeframe_adapter  # noqa: F401  (registers SafeFrameAdapter on import)

find = base.find
register = base.register
