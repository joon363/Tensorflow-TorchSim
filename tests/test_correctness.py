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
        # Reset kernel counter and torch.compile cache between tests (과제 2)
        torch._dynamo.reset()
        try:
            from Tensorflow.TensorFlowFrontend.tf_mlir_conversion import reset_tf_kernel_counter
            reset_tf_kernel_counter()
        except ImportError:
            pass
        
        # 1. Compile TF to MLIR
        tf_to_MLIR(tf_fn, *inputs_tf)
        
        # 2. Run TF ground truth
        tf_out = tf_fn(*inputs_tf)
        if isinstance(tf_out, tuple):
            tf_out = tf_out[0]
        tf_out_np = tf_out.numpy()
        
        # Construct arg_attributes for tf_to_MLIR transformation
        arg_attributes = []
        for idx, t in enumerate(inputs_torch):
            arg_attributes.append((f"arg{idx}", [1, t.dtype, t.numel(), list(t.shape), list(t.stride())]))
        
        # Add output attribute
        out_shape = list(tf_out.shape)
        out_numel = tf_out_np.size
        out_stride = []
        current_stride = 1
        for dim in reversed(out_shape):
            out_stride.append(current_stride)
            current_stride *= dim
        out_stride.reverse()
        arg_attributes.append(("buf0", [2, torch.float32, out_numel, out_shape, out_stride]))

        # Perform the MLIR transformation in a separate process to avoid TF/MLIR library collision
        import json
        import subprocess
        
        serializable_arg_attrs = []
        for name, attr in arg_attributes:
            serializable_arg_attrs.append([
                name,
                [
                    attr[0],
                    str(attr[1]), # torch.dtype to string
                    attr[2],
                    attr[3],
                    attr[4]
                ]
            ])
            
        args_json = json.dumps(serializable_arg_attrs)
        
        cmd = [
            "python3", "-c",
            "import sys, json, torch\n"
            "from PyTorchSimFrontend.extension_codecache import transform_tf_mlir\n"
            "content = sys.stdin.read()\n"
            "arg_attrs = json.loads(sys.argv[1])\n"
            "for item in arg_attrs:\n"
            "    dtype_str = item[1][1]\n"
            "    if 'float32' in dtype_str: item[1][1] = torch.float32\n"
            "    elif 'float64' in dtype_str: item[1][1] = torch.float64\n"
            "    elif 'int64' in dtype_str: item[1][1] = torch.int64\n"
            "    elif 'int32' in dtype_str: item[1][1] = torch.int32\n"
            "    elif 'bool' in dtype_str: item[1][1] = torch.bool\n"
            "print(transform_tf_mlir(content, arg_attrs))",
            args_json
        ]
        
        tf_mlir_path = Path(OUT_DIR) / "tf-mlir.mlir"
        proc = subprocess.run(cmd, input=tf_mlir_path.read_text(), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MLIR transformation subprocess failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")
            
        tf_mlir_path.write_text(proc.stdout)

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

from tests_common import get_all_tests

def run_all_correctness_tests(tests=None, silent=False):
    if tests is None:
        tests = get_all_tests()
    
    # Tests expected to fail due to known limitations
    known_limitations = set()
    
    results = []
    passed_count = 0
    xfail_count = 0
    
    for idx, (name, fn, tf_in, pt_in, dummy) in enumerate(tests):
        passed, err = run_tf_test(name, fn, tf_in, pt_in, dummy)
        if passed:
            passed_count += 1
            status = "PASS"
            if not silent: print(f"Test [{idx+1}/{len(tests)}] {name}: PASSED")
        elif name in known_limitations:
            xfail_count += 1
            status = "XFAIL"
            if not silent: print(f"Test [{idx+1}/{len(tests)}] {name}: XFAIL (known limitation)")
        else:
            status = "FAIL"
            if not silent:
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
        "xfail_tests": xfail_count,
        "results": results
    }

    # Save to JSON log file
    report_json_path = os.path.join(OUT_DIR, "test_results.json")
    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if not silent:
        print(f"\n================ Summary: [{passed_count}/{len(tests)}] Passed, [{xfail_count}] Known Limitations ================")
    
    return passed_count == len(tests) - xfail_count

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    tests = get_all_tests()
    success = run_all_correctness_tests(tests)
    if not success:
        sys.exit(1)
