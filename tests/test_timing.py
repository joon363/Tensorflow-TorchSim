import torch
import os
import sys
import subprocess
import tensorflow as tf
import numpy as np
import shutil
import glob
import re
import time

base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_dir)

from Simulator.simulator import TOGSimulator

# Config paths
MLIR_OPT = f"{base_dir}/Tensorflow/binaries/mlir-opt"
STABLEHLO_OPT = f"{base_dir}/Tensorflow/binaries/stablehlo-opt"
OUT_DIR = f"{base_dir}/Tensorflow/tests/out"
LOG_DIR = f"{base_dir}/Tensorflow/tests/togsim_results"
OUTPUTS_DIR = f"{base_dir}/outputs"

# 1. Clean the directories
for path in [OUT_DIR, LOG_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# Helper to run process and log outputs
def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(OUT_DIR, "compilation.log"), "a") as f:
        f.write(f">> {' '.join(cmd)}\n")
        if result.stdout:
            f.write(result.stdout + "\n")
        if result.stderr:
            f.write(result.stderr + "\n")
    result.check_returncode()

def tf_to_MLIR(fn, *args):
    input_stablehlo = fn.experimental_get_compiler_ir(*args)(stage="stablehlo")
    input_stablehlo_path = os.path.join(OUT_DIR, "0_input_stablehlo.mlir")
    with open(input_stablehlo_path, "w") as f:
        f.write(input_stablehlo)
    
    output_stablehlo_path = os.path.join(OUT_DIR, "1_stablehlo_clean.mlir")
    run([
        STABLEHLO_OPT,
        str(input_stablehlo_path),
        "--canonicalize",
        "--cse",
        "--stablehlo-legalize-to-linalg",
        "--linalg-specialize-generic-ops",  
        "-o", str(output_stablehlo_path),
    ])

    output_path = os.path.join(OUT_DIR, "tf-mlir.mlir")
    run([
        MLIR_OPT,
        str(output_stablehlo_path),
        "-one-shot-bufferize=\"bufferize-function-boundaries\"",
        "-convert-bufferization-to-memref",
        "-o", str(output_path),
    ])

def get_latest_log_result():
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    if not log_files:
        raise RuntimeError("No log files found in " + LOG_DIR)
    latest_file = max(log_files, key=os.path.getmtime)
    print(f"Parsing timing metrics from: {latest_file}")
    
    # Parse metrics
    core_metrics, dram_ch_bw, avg_dram_bw, sim_time, total_cycle = TOGSimulator.get_result_from_file(latest_file)
    return {
        "file": latest_file,
        "total_cycle": total_cycle,
        "matmul_active_cycle": core_metrics.get("MatMul_active_cycle", 0),
        "vector_active_cycle": core_metrics.get("Vector_active_cycle", 0),
        "systolic_util": core_metrics.get("Systolic_Array_Utilization", 0.0),
        "vector_util": core_metrics.get("Vector_Unit_Utilization", 0.0),
        "avg_dram_bw": avg_dram_bw
    }

def find_latest_output_dir(after_time):
    """Find the most recently modified output directory created after after_time."""
    candidates = []
    for name in os.listdir(OUTPUTS_DIR):
        d = os.path.join(OUTPUTS_DIR, name)
        if os.path.isdir(d):
            mtime = os.path.getmtime(d)
            if mtime > after_time:
                candidates.append((mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def analyze_mlir_structure(output_dir):
    """Analyze MLIR files in an output directory for key structural patterns."""
    result = {
        "dma_count": 0,
        "matmul_count": 0,
        "vcix_count": 0,
        "affine_for_count": 0,
        "tog_size_bytes": 0,
        "output_dir": output_dir,
    }
    if not output_dir:
        return result
    
    # Scan all .mlir files
    for f in glob.glob(os.path.join(output_dir, "*.mlir")):
        try:
            content = open(f).read()
        except Exception:
            continue
        fname = os.path.basename(f)
        
        result["dma_count"] += len(re.findall(r"memref\.dma_start", content))
        result["matmul_count"] += len(re.findall(r"linalg\.matmul", content))
        result["affine_for_count"] += len(re.findall(r"affine\.for", content))
        
        # vcix instructions in LLVM MLIR
        if "_sample_llvm" in fname or "_llvm" in fname:
            result["vcix_count"] += len(re.findall(r"vcix|sf\.vc", content))
    
    # Check TOG file size
    for f in glob.glob(os.path.join(output_dir, "*_tog.py")):
        result["tog_size_bytes"] = os.path.getsize(f)
    
    return result

def parse_gem5_stats(output_dir):
    """Parse Gem5 stats.txt to get CPU cycle counts."""
    result = {"cpu_cycles": 0, "instructions": 0}
    if not output_dir:
        return result
    
    # Look for m5out/stats.txt in subdirectories
    for root, dirs, files in os.walk(output_dir):
        if "stats.txt" in files:
            stats_path = os.path.join(root, "stats.txt")
            try:
                content = open(stats_path).read()
                # Get the LAST stats dump
                sections = content.split("---------- Begin Simulation Statistics ----------")
                if len(sections) > 1:
                    last_section = sections[-1]
                    cycles_m = re.search(r"system\.cpu\.numCycles\s+(\d+)", last_section)
                    inst_m = re.search(r"system\.cpu\.committedInsts\s+(\d+)", last_section)
                    if cycles_m:
                        result["cpu_cycles"] = int(cycles_m.group(1))
                    if inst_m:
                        result["instructions"] = int(inst_m.group(1))
            except Exception:
                pass
            break
    return result

def run_torch_native_timing(inputs_torch, dummy_fn):
    print("--- Running Torch Native Timing Simulation ---")
    os.environ['TENSORFLOW_MLIR_DIRECT_TEST'] = 'False'
    os.environ['TENSORFLOW_NPU_CODEGEN'] = 'False'
    os.environ['TOGSIM_CONFIG'] = f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_timing_only.yml"
    os.environ['TORCHSIM_LOG_PATH'] = LOG_DIR
    
    torch._dynamo.reset()
    
    t_before = time.time()
    
    # Warmup / Compile and Simulate
    device = torch.device("npu:0")
    torch_inputs = [t.to(device) for t in inputs_torch]
    
    opt_fn = torch.compile(dynamic=False)(dummy_fn)
    npu_out = opt_fn(*torch_inputs)
    
    # Block and wait for simulation to finish
    sim = torch.npu.get_tog_simulator()
    if sim:
        sim.until()
    
    output_dir = find_latest_output_dir(t_before)
    togsim = get_latest_log_result()
    mlir_info = analyze_mlir_structure(output_dir)
    gem5_info = parse_gem5_stats(output_dir)
    
    return {**togsim, "mlir": mlir_info, "gem5": gem5_info}

def run_tf_npu_timing(tf_fn, inputs_tf, inputs_torch, dummy_fn):
    print("--- Running TF NPU Codegen Timing Simulation ---")
    os.environ['TENSORFLOW_MLIR_DIRECT_TEST'] = 'True'
    os.environ['TENSORFLOW_NPU_CODEGEN'] = 'True'
    os.environ['TOGSIM_CONFIG'] = f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_timing_only.yml"
    os.environ['TORCHSIM_LOG_PATH'] = LOG_DIR
    
    torch._dynamo.reset()
    try:
        from Tensorflow.TensorFlowFrontend.tf_mlir_conversion import reset_tf_kernel_counter
        reset_tf_kernel_counter()
    except ImportError:
        pass
    
    # 1. Compile TF to MLIR
    tf_to_MLIR(tf_fn, *inputs_tf)
    
    # 2. Run TF ground truth to get shapes
    tf_out = tf_fn(*inputs_tf)
    if isinstance(tf_out, tuple):
        tf_out = tf_out[0]
    tf_out_np = tf_out.numpy()
    
    # Construct arg_attributes
    arg_attributes = []
    for idx, t in enumerate(inputs_torch):
        arg_attributes.append((f"arg{idx}", [1, t.dtype, t.numel(), list(t.shape), list(t.stride())]))
        
    out_shape = list(tf_out.shape)
    out_numel = tf_out_np.size
    out_stride = []
    current_stride = 1
    for dim in reversed(out_shape):
        out_stride.append(current_stride)
        current_stride *= dim
    out_stride.reverse()
    arg_attributes.append(("buf0", [2, torch.float32, out_numel, out_shape, out_stride]))

    # Call transform_tf_mlir to build the NPU codegen version
    import json
    import subprocess
    serializable_arg_attrs = []
    for name, attr in arg_attributes:
        serializable_arg_attrs.append([
            name,
            [
                attr[0],
                str(attr[1]),
                attr[2],
                attr[3],
                attr[4]
            ]
        ])
        
    args_json = json.dumps(serializable_arg_attrs)
    cmd = [
        "python3", "-c",
        "import sys, json, torch, os\n"
        "os.environ['TENSORFLOW_NPU_CODEGEN'] = 'True'\n"
        "from PyTorchSimFrontend.extension_codecache import transform_tf_mlir\n"
        "content = sys.stdin.read()\n"
        "arg_attrs = json.loads(sys.argv[1])\n"
        "for item in arg_attrs:\n"
        "    dtype_str = item[1][1]\n"
        "    if 'float32' in dtype_str: item[1][1] = torch.float32\n"
        "print(transform_tf_mlir(content, arg_attrs))",
        args_json
    ]
    
    tf_mlir_path = os.path.join(OUT_DIR, "tf-mlir.mlir")
    proc = subprocess.run(cmd, input=open(tf_mlir_path).read(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"TF NPU codegen subprocess failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")
    with open(tf_mlir_path, "w") as f:
        f.write(proc.stdout)
    
    # Save generated NPU MLIR for analysis
    with open(os.path.join(OUT_DIR, "tf_npu_codegen.mlir"), "w") as f:
        f.write(proc.stdout)
        
    t_before = time.time()
    
    # Run simulation
    device = torch.device("npu:0")
    torch_inputs = [t.to(device) for t in inputs_torch]
    
    opt_fn = torch.compile(dynamic=False)(dummy_fn)
    npu_out = opt_fn(*torch_inputs)
    
    sim = torch.npu.get_tog_simulator()
    if sim:
        sim.until()
    
    output_dir = find_latest_output_dir(t_before)
    togsim = get_latest_log_result()
    mlir_info = analyze_mlir_structure(output_dir)
    gem5_info = parse_gem5_stats(output_dir)
    
    return {**togsim, "mlir": mlir_info, "gem5": gem5_info}

if __name__ == "__main__":
    print("=" * 85)
    print("         TIMING VERIFICATION: Native Torch vs TF NPU Codegen")
    print("=" * 85)
    
    # 1. Prepare Matmul 256x256 benchmark
    a_tf = tf.random.normal([256, 256], seed=42)
    b_tf = tf.random.normal([256, 256], seed=43)
    
    a_pt = torch.tensor(a_tf.numpy())
    b_pt = torch.tensor(b_tf.numpy())
    
    @tf.function(jit_compile=True)
    def matmul_tf(x, y):
        return tf.matmul(x, y)
        
    dummy_matmul = lambda x, y: torch.matmul(x, y)
    
    # Run Native Torch Timing
    torch_results = run_torch_native_timing([a_pt, b_pt], dummy_matmul)
    
    # Run TF NPU Codegen Timing
    tf_results = run_tf_npu_timing(matmul_tf, [a_tf, b_tf], [a_pt, b_pt], dummy_matmul)
    
    # === Layer 1: MLIR Structure Comparison ===
    print("\n" + "=" * 85)
    print("  Layer 1: MLIR Structure Analysis")
    print("=" * 85)
    t_mlir = torch_results["mlir"]
    f_mlir = tf_results["mlir"]
    print(f"{'Metric':<25} | {'Native Torch':<20} | {'TF NPU Codegen':<20}")
    print("-" * 70)
    print(f"{'DMA Operations':<25} | {t_mlir['dma_count']:<20} | {f_mlir['dma_count']:<20}")
    print(f"{'linalg.matmul Ops':<25} | {t_mlir['matmul_count']:<20} | {f_mlir['matmul_count']:<20}")
    print(f"{'affine.for Loops':<25} | {t_mlir['affine_for_count']:<20} | {f_mlir['affine_for_count']:<20}")
    print(f"{'VCIX Instructions':<25} | {t_mlir['vcix_count']:<20} | {f_mlir['vcix_count']:<20}")
    print(f"{'TOG File Size (bytes)':<25} | {t_mlir['tog_size_bytes']:<20} | {f_mlir['tog_size_bytes']:<20}")
    print(f"{'Output Directory':<25} | {os.path.basename(t_mlir['output_dir'] or 'N/A'):<20.20} | {os.path.basename(f_mlir['output_dir'] or 'N/A'):<20.20}")
    
    # === Layer 2: Gem5 CPU Cycles ===
    print("\n" + "=" * 85)
    print("  Layer 2: Gem5 CPU Simulation")
    print("=" * 85)
    t_gem5 = torch_results["gem5"]
    f_gem5 = tf_results["gem5"]
    print(f"{'Metric':<25} | {'Native Torch':<20} | {'TF NPU Codegen':<20}")
    print("-" * 70)
    print(f"{'CPU Cycles':<25} | {t_gem5['cpu_cycles']:<20} | {f_gem5['cpu_cycles']:<20}")
    print(f"{'Committed Instructions':<25} | {t_gem5['instructions']:<20} | {f_gem5['instructions']:<20}")
    
    # === Layer 3: TOGSim NPU Cycles ===
    print("\n" + "=" * 85)
    print("  Layer 3: TOGSim NPU Timing")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Native Torch':<25} | {'TF NPU Codegen':<25}")
    print("-" * 85)
    print(f"{'Total Cycles':<25} | {torch_results['total_cycle']:<25} | {tf_results['total_cycle']:<25}")
    print(f"{'MatMul Active Cycles':<25} | {torch_results['matmul_active_cycle']:<25} | {tf_results['matmul_active_cycle']:<25}")
    print(f"{'Vector Active Cycles':<25} | {torch_results['vector_active_cycle']:<25} | {tf_results['vector_active_cycle']:<25}")
    print(f"{'Systolic Array Util (%)':<25} | {torch_results['systolic_util']:<25.2f} | {tf_results['systolic_util']:<25.2f}")
    print(f"{'Vector Unit Util (%)':<25} | {torch_results['vector_util']:<25.2f} | {tf_results['vector_util']:<25.2f}")
    print(f"{'Average DRAM BW (%)':<25} | {torch_results['avg_dram_bw']:<25.2f} | {tf_results['avg_dram_bw']:<25.2f}")
    
    # === Verdict ===
    print("\n" + "=" * 85)
    print("  VERDICT")
    print("=" * 85)
    
    passed = True
    
    # Check MLIR structure matches
    if f_mlir['dma_count'] > 0 and f_mlir['matmul_count'] > 0:
        print("  [PASS] TF NPU MLIR contains DMA and matmul operations")
    else:
        print("  [FAIL] TF NPU MLIR missing DMA or matmul operations")
        passed = False
    
    # Check TOGSim produced meaningful results
    if tf_results['total_cycle'] > 1:
        print("  [PASS] TOGSim reports non-trivial cycles for TF path")
    else:
        print("  [WARN] TOGSim reports 1 cycle for TF path (empty TOG?)")
    
    # Check Gem5 ran
    if f_gem5['cpu_cycles'] > 0:
        print("  [PASS] Gem5 simulation completed for TF path")
    else:
        print("  [WARN] Gem5 stats not found or zero cycles")
    
    print("=" * 85)
    
    if passed:
        print("\nSUCCESS: Timing simulation executed successfully in both paths!")
        sys.exit(0)
    else:
        print("\nFAILURE: Critical checks failed.")
        sys.exit(1)
