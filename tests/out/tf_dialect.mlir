module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 2288 : i32}} {
  func.func @__inference_add_fn_8(%arg0: tensor<2xf32> {tf._user_specified_name = "x"}, %arg1: tensor<2xf32> {tf._user_specified_name = "y"}) -> tensor<2xf32> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "x,y", outputs = "identity_RetVal"}} {
    %0 = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
