module @test_simple {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 0)

    func.func private @my_kernel(%l : index) {
      aie.use_lock(%l, "Acquire", 0)
      return
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.for %i = %c0 to %c10 step %c1 {
        func.call @my_kernel(%lock) : (index) -> ()
      }
      aie.end
    }
  }
}
