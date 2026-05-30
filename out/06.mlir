module @a_inference_add_fn_108__.1 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: memref<2xf32, strided<[?], offset: ?>>, %arg1: memref<2xf32, strided<[?], offset: ?>>) -> memref<2xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<2xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg1[%arg2] : memref<2xf32, strided<[?], offset: ?>>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%arg2] : memref<2xf32>
    }
    return %alloc : memref<2xf32>
  }
}

