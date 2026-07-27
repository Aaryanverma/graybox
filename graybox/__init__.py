# graybox/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("graybox")
except PackageNotFoundError:
    __version__ = "0.0.0"