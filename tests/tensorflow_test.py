from mlir import ir
import tensorflow as tf

# 1. Define your TensorFlow function
@tf.function(jit_compile=True)
def matrix_multiply(x, y):
    return tf.matmul(x, y)

x_spec = tf.TensorSpec(shape=(128, 128), dtype=tf.float32)
y_spec = tf.TensorSpec(shape=(128, 128), dtype=tf.float32)

# 2. Trace the Concrete Function
concrete_fn = matrix_multiply.get_concrete_function(x_spec, y_spec)

# 3. Extract the MLIR text (Bytes) from TensorFlow
try:
    mlir_bytes = concrete_fn.experimental_get_compiler_ir(stage="stablehlo")
except ValueError:
    # Fallback to HLO if StableHLO isn't supported in your TF version
    mlir_bytes = concrete_fn.experimental_get_compiler_ir(stage="hlo")

# Decode bytes to a standard string
mlir_string = mlir_bytes.decode("utf-8") if isinstance(mlir_bytes, bytes) else mlir_bytes

# 4. Parse the TensorFlow string into an MLIR Module Object
with ir.Context() as ctx:
    # This registers the OpenXLA dialects so it understands the TF output
    mlir_module_obj = ir.Module.parse(mlir_string)
    
    print("Success! Object type:")
    print(type(mlir_module_obj)) 
    # Output: <class 'jaxlib.mlir.ir.Module'>
    
    # Example programmatic interaction:
    print(f"Module Operation Name: {mlir_module_obj.operation.name}")