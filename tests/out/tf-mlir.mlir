module @a_inference_add_fn_8__.9 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: memref<2xf32, strided<[?], offset: ?>>, %arg1: memref<2xf32, strided<[?], offset: ?>>) -> memref<2xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    linalg.add ins(%arg0, %arg1 : memref<2xf32, strided<[?], offset: ?>>, memref<2xf32, strided<[?], offset: ?>>) outs(%alloc : memref<2xf32>)
    %cast = memref.cast %alloc : memref<2xf32> to memref<2xf32, strided<[?], offset: ?>>
    return %alloc : memref<2xf32>
  }
}

