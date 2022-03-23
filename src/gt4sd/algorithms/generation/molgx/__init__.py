"""MolGX initialization."""
import logging

from .core import MolGX, MolGXQM9Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "MolGX",
    "MolGXQM9Generator",
]
