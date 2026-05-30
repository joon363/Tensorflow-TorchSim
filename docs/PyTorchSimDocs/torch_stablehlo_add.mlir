module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32> {jax.result_info = "result[0]"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<2xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}