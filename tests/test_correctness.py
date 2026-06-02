import torch
import os
import sys
import subprocess
import tensorflow as tf
import numpy as np
import json
import shutil
import logging
from pathlib import Path

# Suppress verbose Python logging for PyTorchSim
logging.getLogger("pytorchsimfrontend").setLevel(logging.WARNING)

base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_dir)

MLIR_OPT = f"{base_dir}/Tensorflow/binaries/mlir-opt"
MLIR_TRANSLATE = f"{base_dir}/Tensorflow/binaries/mlir-translate"
STABLEHLO_OPT = f"{base_dir}/Tensorflow/binaries/stablehlo-opt"
OUT_DIR = f"{base_dir}/Tensorflow/tests/out"

# 1. Clean the out folder before running tests
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Default Config
os.environ['TOGSIM_CONFIG'] = f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_functional_only.yml"
os.environ['TORCHSIM_DUMP_LOG_PATH'] = os.path.join(os.getcwd(), "togsim_results")
os.environ['TENSORFLOW_MLIR_DIRECT_TEST'] = "True"

LOG_FILE_PATH = os.path.join(OUT_DIR, "compilation.log")

# Helper for run (logs to compilation.log and suppresses standard output)
def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f">> {' '.join(cmd)}\n")
        if result.stdout:
            f.write(result.stdout + "\n")
        if result.stderr:
            f.write(result.stderr + "\n")
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
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"==== Final MLIR at {str(output_path)} ====\n")
        f.write(output_path.read_text() + "\n")

# Generic Correctness Test Runner
def run_tf_test(name, tf_fn, inputs_tf, inputs_torch, dummy_fn, rtol=1e-4, atol=1e-4):
    try:
        # 1. Compile TF to MLIR
        tf_to_MLIR(tf_fn, *inputs_tf)
        
        # 2. Run TF ground truth
        tf_out = tf_fn(*inputs_tf)
        if isinstance(tf_out, tuple):
            tf_out = tf_out[0]
        tf_out_np = tf_out.numpy()
        
        # 3. Trigger PyTorchSim
        device = torch.device("npu:0")
        torch_inputs = [t.to(device) for t in inputs_torch]
        
        opt_fn = torch.compile(dynamic=False)(dummy_fn)
        npu_out = opt_fn(*torch_inputs)
        npu_out_np = npu_out.cpu().numpy()
        
        # Copy the PyTorch wrapper codegen path to out/
        wrapper_path = os.environ.get("TORCHSIM_LAST_COMPILED_MODULE")
        if wrapper_path and os.path.exists(wrapper_path):
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            shutil.copy(wrapper_path, os.path.join(OUT_DIR, f"pytorch_wrapper_codegen_{safe_name}.py"))
        
        # 4. Correctness Check
        if np.allclose(npu_out_np, tf_out_np, rtol=rtol, atol=atol):
            return True, None
        else:
            err_msg = f"Numerical mismatch.\nTF shape: {tf_out_np.shape}, NPU shape: {npu_out_np.shape}\nTF: {tf_out_np}\nNPU: {npu_out_np}"
            return False, err_msg
    except Exception as e:
        import traceback
        err_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return False, err_msg

# TF functions to test
@tf.function(jit_compile=True)
def add_fn(x, y, z):
    return x + y + z

@tf.function(jit_compile=True)
def mul_fn(x, y):
    return x * y

@tf.function(jit_compile=True)
def matmul_fn(x, y):
    return tf.matmul(x, y)

@tf.function(jit_compile=True)
def linear_bias_fn(x, w, b):
    return tf.matmul(x, w) + b

@tf.function(jit_compile=True)
def relu_fn(x):
    return tf.nn.relu(x)

@tf.function(jit_compile=True)
def conv2d_fn(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

if __name__ == "__main__":
    tests = []
    
    # 1. Vector Add 1D
    x_tf = tf.constant([1.0, 2.0], dtype=tf.float32)
    y_tf = tf.constant([3.0, 4.0], dtype=tf.float32)
    z_tf = tf.constant([5.0, 6.0], dtype=tf.float32)
    x_pt = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y_pt = torch.tensor([3.0, 4.0], dtype=torch.float32)
    z_pt = torch.tensor([5.0, 6.0], dtype=torch.float32)
    dummy_add = lambda x, y, z: x + y + z
    tests.append(("Vector Add 1D", add_fn, [x_tf, y_tf, z_tf], [x_pt, y_pt, z_pt], dummy_add))

    # 2. Matrix Add 2D
    x_tf_2d = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_tf_2d = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    z_tf_2d = tf.constant([[9.0, 10.0], [11.0, 12.0]], dtype=tf.float32)
    x_pt_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_pt_2d = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    z_pt_2d = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    tests.append(("Matrix Add 2D", add_fn, [x_tf_2d, y_tf_2d, z_tf_2d], [x_pt_2d, y_pt_2d, z_pt_2d], dummy_add))

    # 3. Elementwise multiplication
    dummy_mul = lambda x, y: x * y
    tests.append(("Vector Mul 2D", mul_fn, [x_tf_2d, y_tf_2d], [x_pt_2d, y_pt_2d], dummy_mul))

    # 4. Matmul 32x32
    a_tf = tf.random.normal([32, 32], seed=42)
    b_tf = tf.random.normal([32, 32], seed=43)
    a_pt = torch.tensor(a_tf.numpy())
    b_pt = torch.tensor(b_tf.numpy())
    dummy_matmul = lambda x, y: torch.matmul(x, y)
    tests.append(("Matmul 32x32", matmul_fn, [a_tf, b_tf], [a_pt, b_pt], dummy_matmul))

    # 5. Linear Bias 32x32
    bias_tf = tf.random.normal([32], seed=44)
    bias_pt = torch.tensor(bias_tf.numpy())
    dummy_linear_bias = lambda x, w, b: torch.matmul(x, w) + b
    tests.append(("Linear Bias 32x32", linear_bias_fn, [a_tf, b_tf, bias_tf], [a_pt, b_pt, bias_pt], dummy_linear_bias))

    # 6. ReLU 32x32
    dummy_relu = lambda x: torch.nn.functional.relu(x)
    tests.append(("ReLU 32x32", relu_fn, [a_tf], [a_pt], dummy_relu))

    # 7. Non-square Matmul (GEMM)
    gemm_a_tf = tf.random.normal([16, 32], seed=45)
    gemm_b_tf = tf.random.normal([32, 64], seed=46)
    gemm_a_pt = torch.tensor(gemm_a_tf.numpy())
    gemm_b_pt = torch.tensor(gemm_b_tf.numpy())
    tests.append(("GEMM (non-square Matmul 16x32x64)", matmul_fn, [gemm_a_tf, gemm_b_tf], [gemm_a_pt, gemm_b_pt], dummy_matmul))

    # 8. Conv2D (NHWC / HWCF)
    conv_x_tf = tf.random.normal([1, 14, 14, 8], seed=47)
    conv_w_tf = tf.random.normal([3, 3, 8, 16], seed=48)
    conv_x_pt = torch.tensor(conv_x_tf.numpy())
    conv_w_pt = torch.tensor(conv_w_tf.numpy())
    dummy_conv = lambda x, w: x.repeat(1, 1, 1, 2) + w[0, 0, 0, 0] * 0.0
    tests.append(("Conv2D 14x14x8 to 16", conv2d_fn, [conv_x_tf, conv_w_tf], [conv_x_pt, conv_w_pt], dummy_conv))

    results = []
    passed_count = 0
    
    # Run tests and output log
    for idx, (name, fn, tf_in, pt_in, dummy) in enumerate(tests):
        passed, err = run_tf_test(name, fn, tf_in, pt_in, dummy)
        if passed:
            passed_count += 1
            status = "PASS"
            print(f"Test [{idx+1}/{len(tests)}] {name}: PASSED")
        else:
            status = "FAIL"
            print(f"Test [{idx+1}/{len(tests)}] {name}: FAILED")
            print(f"Error info: {err}")
        results.append({
            "test_index": idx + 1,
            "name": name,
            "status": status,
            "error": err
        })

    summary = {
        "total_tests": len(tests),
        "passed_tests": passed_count,
        "results": results
    }

    # Save to JSON log file
    report_json_path = os.path.join(OUT_DIR, "test_results.json")
    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n================ Summary: [{passed_count}/{len(tests)}] Passed ================")
    
    if passed_count < len(tests):
        sys.exit(1)
