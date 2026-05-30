import torch
import os
import sys
import subprocess
import tensorflow as tf
from pathlib import Path

base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_dir)
MLIR_OPT = "/LLVM-22.1.0-rc1-Linux-X64/bin/mlir-opt"
MLIR_TRANSLATE = "/LLVM-22.1.0-rc1-Linux-X64/bin/mlir-translate"
MLIR_OPT_PYTORCHSIM = "/riscv-llvm/bin/mlir-opt"
STABLEHLO_OPT = "/workspace/stablehlo/build/bin/stablehlo-opt"
OUT_DIR = "/workspace/PyTorchSim/tests/TensorFlow/out"
os.makedirs(OUT_DIR, exist_ok=True)

# Default Config
os.environ['TOGSIM_CONFIG']=f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_functional_only.json"
os.environ['TORCHSIM_DUMP_LOG_PATH']=os.path.join(os.getcwd(), "togsim_results")
# Test Config
os.environ['TENSORFLOW_MLIR_DIRECT_TEST']="True"

# Helper
def run(cmd):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

# Tensorflow -> MLIR (Direct Conversion)
def tf_to_MLIR(fn, *args):
    # 1. stableHLO opt
    input_stablehlo = fn.experimental_get_compiler_ir(*args)(stage="stablehlo")
    input_stablehlo_path = Path(OUT_DIR) / "0_input_stablehlo.mlir"
    input_stablehlo_path.write_text(input_stablehlo)
    output_stablehlo_path = Path(OUT_DIR) / "1_stablehlo_clean.mlir"
    run([
        STABLEHLO_OPT,
        str(input_stablehlo_path),
        "--canonicalize",
        "--cse",
        "--stablehlo-legalize-to-linalg",
        "--linalg-specialize-generic-ops",  
        "-o", str(output_stablehlo_path),
    ])

    # 2. MLIR opt
    output_path = Path(OUT_DIR) / "tf-mlir.mlir"
    run([
        MLIR_OPT,
        str(output_stablehlo_path),
        "-one-shot-bufferize=\"bufferize-function-boundaries\"",
        "-convert-bufferization-to-memref",

        "-o", str(output_path),
    ])
    print(f"==== Final MLIR at {str(output_path)}====")
    print(output_path.read_text())


# TensorFlow Function to Test
@tf.function(jit_compile=True)
def add_fn(x, y, z):
    return x + y + z

x_dummy = tf.constant([0, 0], dtype=tf.float32)
y_dummy = tf.constant([0, 0], dtype=tf.float32)
z_dummy = tf.constant([0, 0], dtype=tf.float32)

# Do conversion
tf_to_MLIR(add_fn, x_dummy, y_dummy, z_dummy)

# Set PyTorchSim
from Scheduler.scheduler import PyTorchSimRunner
device = PyTorchSimRunner.setup_device().custom_device()

# Trigger PyTorchSim
def no_op(x, y):
    "PyTorchSim Will Use add_fn's MLIR, not this one"
    return x*y
x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
y = torch.tensor([3.0, 4.0], dtype=torch.float32, device=device)
opt_fn = torch.compile(dynamic=False)(no_op)
npu_out_test = opt_fn(x,y)
print(npu_out_test.cpu())