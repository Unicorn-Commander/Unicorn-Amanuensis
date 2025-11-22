#!/usr/bin/env python3
"""
Wrapper script to run aiecc.py with Python 3.13 compatibility patch.
"""

import sys
import os

# Patch dataclasses to work with Python 3.13
import dataclasses

# Save original _is_classvar function
_original_is_classvar = dataclasses._is_classvar

def _patched_is_classvar(a_type, typing):
    """Patched version that handles both typing.ClassVar and typing._ClassVar"""
    # Check for ClassVar using hasattr to avoid AttributeError
    if hasattr(typing, '_ClassVar'):
        return type(a_type) is typing._ClassVar
    else:
        # Python 3.13+ - _ClassVar doesn't exist, use ClassVar directly
        return (type(a_type) is type(typing.ClassVar) or
                (hasattr(a_type, '__origin__') and
                 a_type.__origin__ is typing.ClassVar))

# Apply the patch
dataclasses._is_classvar = _patched_is_classvar

# Now run aiecc.py as subprocess
import subprocess

args = [
    '/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py',
    '--alloc-scheme=basic-sequential',
    '--aie-generate-xclbin',
    '--no-xchesscc',
    '--no-xbridge',
    'test_simple_ln.mlir'
]

result = subprocess.run(args, capture_output=False, text=True)
sys.exit(result.returncode)
