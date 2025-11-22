"""
Site customization to patch Python 3.13 typing module compatibility.
This file is automatically imported by Python and patches the typing module
before any other imports occur.
"""

import typing

# Patch for Python 3.13 compatibility
if not hasattr(typing, '_ClassVar'):
    typing._ClassVar = typing.ClassVar
    print("Applied Python 3.13 typing._ClassVar compatibility patch")
