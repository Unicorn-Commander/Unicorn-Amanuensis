#!/usr/bin/env python3
"""
Generate MLIR-AIE2 Kernels using Python Bindings
Generates valid MLIR-AIE2 code for AMD Phoenix NPU
"""

import sys
sys.path.insert(0, "/home/ucadmin/mlir-aie-prebuilt/mlir_aie/python")

from aie.dialects import aie, aiex, scf, arith, memref, func
from aie.dialects.aie import *
from aie.ir import *
from aie.passmanager import *

def generate_whisper_npu_kernel():
    """Generate MLIR-AIE2 kernel for WhisperX acceleration"""

    with Context() as ctx, Location.unknown():
        # Load AIE dialect
        aie.register_dialect(ctx)
        aiex.register_dialect(ctx)

        # Create module
        module = Module.create()

        with InsertionPoint(module.body):
            # Create AIE device for npu1_4col (Phoenix NPU)
            @device(AIEDevice.npu1_4col)
            def device_body():
                # Define compute tiles (row 2-5 have memory in AIE2)
                tile_0_2 = tile(0, 2)
                tile_1_2 = tile(1, 2)
                tile_2_2 = tile(2, 2)
                tile_3_2 = tile(3, 2)

                # Define shim tiles for DMA (row 0)
                tile_0_0 = tile(0, 0)
                tile_1_0 = tile(1, 0)

                # Create buffers in compute tiles
                with InsertionPoint.at_block_begin(tile_0_2.body):
                    buf_a = buffer(tile_0_2, [1024], T.i32(), name="buf_a")

                with InsertionPoint.at_block_begin(tile_1_2.body):
                    buf_b = buffer(tile_1_2, [1024], T.i32(), name="buf_b")

                with InsertionPoint.at_block_begin(tile_2_2.body):
                    buf_c = buffer(tile_2_2, [1024], T.i32(), name="buf_c")

                # Create locks for synchronization
                lock_0 = lock(tile_0_2, lock_id=0, init=1)
                lock_1 = lock(tile_1_2, lock_id=0, init=0)
                lock_2 = lock(tile_2_2, lock_id=0, init=0)

                # Core computation
                @core(tile_0_2)
                def core_body():
                    use_lock(lock_0, LockAction.Acquire, 1)

                    # Simple computation
                    for_op = scf.ForOp(
                        arith.ConstantOp(IndexType.get(), 0),
                        arith.ConstantOp(IndexType.get(), 1024),
                        arith.ConstantOp(IndexType.get(), 1)
                    )

                    with InsertionPoint(for_op.body):
                        val = arith.ConstantOp(IntegerType.get_signless(32), 42)
                        memref.StoreOp(val, buf_a, [for_op.induction_variable])
                        scf.YieldOp([])

                    use_lock(lock_0, LockAction.Release, 0)
                    end()

                # Flow between tiles
                flow(tile_0_2, WireBundle.Core, 0, tile_1_2, WireBundle.Core, 0)
                flow(tile_1_2, WireBundle.Core, 0, tile_2_2, WireBundle.Core, 0)

        return module

def generate_simple_matmul():
    """Generate simple matrix multiply kernel"""

    with Context() as ctx, Location.unknown():
        # Register dialects
        aie.register_dialect(ctx)

        module = Module.create()

        with InsertionPoint(module.body):
            # Simple function for matrix multiply
            i32 = IntegerType.get_signless(32)
            memref_type = MemRefType.get([16, 16], i32)

            @func.FuncOp.from_py_func(memref_type, memref_type, memref_type)
            def matmul(A, B, C):
                c0 = arith.ConstantOp(IndexType.get(), 0)
                c16 = arith.ConstantOp(IndexType.get(), 16)
                c1 = arith.ConstantOp(IndexType.get(), 1)

                # Triple nested loop
                for_i = scf.ForOp(c0, c16, c1)
                with InsertionPoint(for_i.body):
                    i = for_i.induction_variable

                    for_j = scf.ForOp(c0, c16, c1)
                    with InsertionPoint(for_j.body):
                        j = for_j.induction_variable

                        acc = arith.ConstantOp(i32, 0)

                        for_k = scf.ForOp(c0, c16, c1, [acc])
                        with InsertionPoint(for_k.body):
                            k = for_k.induction_variable
                            acc_in = for_k.inner_iter_args[0]

                            a_val = memref.LoadOp(A, [i, k])
                            b_val = memref.LoadOp(B, [k, j])
                            prod = arith.MulIOp(a_val, b_val)
                            acc_new = arith.AddIOp(acc_in, prod)

                            scf.YieldOp([acc_new])

                        memref.StoreOp(for_k.results[0], C, [i, j])
                        scf.YieldOp([])

                    scf.YieldOp([])

                func.ReturnOp([])

        return module

if __name__ == "__main__":
    print("Generating MLIR-AIE2 kernel for WhisperX NPU...")
    print("=" * 60)

    try:
        # Generate simple matmul first (easier to validate)
        module = generate_simple_matmul()
        mlir_code = str(module)

        print("Generated MLIR code:")
        print(mlir_code)

        # Save to file
        output_file = "whisper_npu_generated.mlir"
        with open(output_file, "w") as f:
            f.write(mlir_code)

        print(f"\nSaved to: {output_file}")

    except Exception as e:
        print(f"Error generating MLIR: {e}")
        import traceback
        traceback.print_exc()
