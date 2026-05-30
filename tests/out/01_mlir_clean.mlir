module @a_inference_add_fn_8__.9 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<2xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2xf32>
    %3 = stablehlo.reshape %2 : (tensor<2xf32>) -> tensor<2xf32>
    return %3 : tensor<2xf32>
  }
}

