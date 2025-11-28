# Make `public` a Python package so `from public.usage import USAGE` works
# (This file can be empty, but having it avoids ModuleNotFoundError in some runtimes.)
__all__ = ["usage"]
