#!/usr/bin/env python3
"""
Workaround script to compile XCLBIN with Python 3.13 compatibility patch.
This script patches the typing module before importing aiecc to work around
the _ClassVar deprecation in Python 3.13.
"""

import sys
import typing

# Patch Python 3.13 compatibility issue
if not hasattr(typing, '_ClassVar'):
    typing._ClassVar = typing.ClassVar

# Now import and run aiecc
sys.path.insert(0, '/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages')

from aie.compiler.aiecc.main import main

if __name__ == '__main__':
    # Run aiecc with command line arguments
    sys.argv = [
        'aiecc.py',
        '--alloc-scheme=basic-sequential',
        '--aie-generate-xclbin',
        '--no-xchesscc',
        '--no-xbridge',
        'test_simple_ln.mlir'
    ]
    main()
