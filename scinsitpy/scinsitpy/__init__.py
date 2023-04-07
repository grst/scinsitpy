from importlib.metadata import version

from . import pl, pp, tl, io, util

__all__ = ["pl", "pp", "tl", "io", "util"]

__version__ = version("scinsitpy-0.0.1")
