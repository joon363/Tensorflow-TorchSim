from mlir import ir
import tensorflow as tf

@tf.function
def matrix_multiply(x, y):
    return tf.matmul(x, y)

cf = matrix_multiply.get_concrete_function(
    tf.TensorSpec((128,128), tf.float32),
    tf.TensorSpec((128,128), tf.float32)
)

mlir_string = tf.mlir.experimental.convert_function(cf)

with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(mlir_string)

print(module)