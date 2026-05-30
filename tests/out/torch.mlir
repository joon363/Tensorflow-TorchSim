module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32> {jax.result_info = "result[0]"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

