"""Extras handling."""
# extras requirements
EXTRAS_ENABLED: bool
try:
    import cog  # noqa: F401
    import pag  # noqa: F401

    EXTRAS_ENABLED = True
except ImportError:
    EXTRAS_ENABLED = False
