//===- test_nosqrt_ln.mlir ----------------------------------------*- MLIR -*-===//
//
// Minimal test: Single LayerNorm kernel (512 elements) - NO SQRT version
// Tests basic XCLBIN generation and NPU execution
//
//===----------------------------------------------------------------------===//

module @test_layernorm_512_nosqrt {
    aie.device(npu1) {
        // External kernel function
        func.func private @layernorm_512_nosqrt(memref<1024xi8>, memref<1024xi8>)

        // Tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC
        %tile02 = aie.tile(0, 2)  // Compute

        // ObjectFIFOs (512 bf16 = 1024 bytes)
        aie.objectfifo @of_input(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>

        // Core implementation
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewIn = aie.objectfifo.acquire @of_input(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<1024xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>

                func.call @layernorm_512_nosqrt(%elemIn, %elemOut) : (memref<1024xi8>, memref<1024xi8>) -> ()

                aie.objectfifo.release @of_input(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }
            aie.end
        } {link_with="layernorm_512_nosqrt.o"}

        // Runtime sequence
        aiex.runtime_sequence(
            %input : memref<1024xi8>,
            %output : memref<1024xi8>
        ) {
            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c1024 = arith.constant 1024 : i64

            aiex.npu.dma_memcpy_nd(%input[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1, %c1024]
                                          [%c0, %c0, %c0, %c1]) {
                metadata = @of_input,
                id = 1 : i64
            } : memref<1024xi8>

            aiex.npu.dma_memcpy_nd(%output[%c0, %c0, %c0, %c0]
                                          [%c1, %c1, %c1, %c1024]
                                          [%c0, %c0, %c0, %c1]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<1024xi8>

            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
