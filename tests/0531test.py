from mlir import ir
import tensorflow as tf

# TensorFlow 함수 정의
@tf.function(jit_compile=True)
def matrix_multiply(x, y):
    return tf.matmul(x, y)

# ConcreteFunction 생성
x_spec = tf.TensorSpec(shape=(128, 128), dtype=tf.float32)
y_spec = tf.TensorSpec(shape=(128, 128), dtype=tf.float32)

concrete_fn = matrix_multiply.get_concrete_function(x_spec, y_spec)

# TensorFlow MLIR 생성
mlir_string = tf.mlir.experimental.convert_function(concrete_fn)

print("=== MLIR Preview ===")
print(mlir_string[:500])

# MLIR Python bindings로 파싱
with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(mlir_string)

    print("\n=== Parse Success ===")
    print("Type:", type(module))
    print("Top operation:", module.operation.name)

    print("\n=== Module ===")
    print(module)