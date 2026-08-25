"""Canonical MiniPushT environment.

Imported by Chapter 1 and referenced in later chapters.
The full implementation lives in chapters/01_tiny_vla/mini_pusht.py
and is re-exported here for convenience.
"""
import sys
import os

# Add chapter directory so we can import the canonical implementation
_chapter_dir = os.path.join(os.path.dirname(__file__), "..", "chapters", "01_tiny_vla")
sys.path.insert(0, os.path.abspath(_chapter_dir))

from mini_pusht import MiniPushT  # noqa: F401

__all__ = ["MiniPushT"]
