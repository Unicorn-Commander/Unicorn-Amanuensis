#!/usr/bin/env python3
"""
Direct compilation of 32x32 matmul kernel using MLIR AIE tools
"""

import subprocess
import os
import sys
from pathlib import Path

# Set up paths
kernel_dir = Path(__file__).parent
build_dir = kernel_dir / "build_matmul_32x32"
build_dir.mkdir(exist_ok=True)

mlir_file = kernel_dir / "matmul_32x32.mlir"
c_file = kernel_dir / "matmul_int8_32x32.c"

# Peano compiler location
peano_dir = Path("/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie")
peano_bin = peano_dir / "bin"

if not peano_bin.exists():
    print(f"❌ ERROR: Peano compiler not found at {peano_bin}")
    sys.exit(1)

print("="*60)
print("Compiling 32x32 Matmul Kernel")
print("="*60)
print()

os.chdir(build_dir)

# Step 1: Compile C kernel
print("Step 1: Compiling C kernel...")
cmd = [
    str(peano_bin / "clang"),
    "--target=aie2-none-unknown-elf",
    "-c",
    str(c_file),
    "-o",
    "matmul_32x32.o"
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ C compilation failed:")
    print(result.stderr)
    sys.exit(1)
print("✅ C kernel compiled: matmul_32x32.o")
print()

# Step 2: Use aiecc.py from venv
print("Step 2: Compiling MLIR to XCLBIN...")
aiecc = Path("/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py")

cmd = [
    "python3",
    str(aiecc),
    f"--sysroot={peano_dir}/sysroot",
    "--host-target=x86_64-amd-linux-gnu",
    str(mlir_file),
    f"-I{peano_dir}/aie_kernels/aie2/include",
    "-o", "matmul_32x32.xclbin",
    "--xclbin-kernel-name=MLIR_AIE",
    f"--peano-install-dir={peano_dir}"
]

print(f"Running: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=False, text=True)
if result.returncode != 0:
    print(f"❌ MLIR compilation failed with return code {result.returncode}")
    sys.exit(1)

print()
print("="*60)
print("✅ Compilation Complete!")
print("="*60)
print()

# Check outputs
xclbin_path = build_dir / "matmul_32x32.xclbin"
seq_path = build_dir / "main_sequence.bin"

if xclbin_path.exists():
    size_kb = xclbin_path.stat().st_size / 1024
    print(f"✅ XCLBIN created: {xclbin_path} ({size_kb:.1f} KB)")
else:
    print(f"❌ XCLBIN not found: {xclbin_path}")

if seq_path.exists():
    size_b = seq_path.stat().st_size
    print(f"✅ Sequence created: {seq_path} ({size_b} bytes)")
else:
    print(f"❌ Sequence not found: {seq_path}")

print()
print("Next step: Test with python3 test_matmul_32x32.py")
