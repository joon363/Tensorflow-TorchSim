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

def get_result_from_file(result_path):
    core_metrics = {}
    dram_channel_bw = {}  # 현재 로그에는 채널별 정보 없음
    avg_dram_bw = 0.0
    simulation_time = float("inf")
    total_cycle = float("inf")

    with open(result_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # DRAM summary
        m = re.search(
            r'channels\s+\d+\.\.\d+\s+combined\s+\|\s+([\d.]+)\s+GB/s aggregate,\s+([\d.]+)% of utilization.*\|\s+(\d+)\s+reads,\s+(\d+)\s+writes',
            line
        )
        if m:
            core_metrics["DRAM_BW_GBps"] = float(m.group(1))
            avg_dram_bw = float(m.group(2))
            core_metrics["DRAM_reads"] = int(m.group(3))
            core_metrics["DRAM_writes"] = int(m.group(4))
            continue

        # Instruction counts
        m = re.search(
            r'Core \[(\d+)\] : (\w+)\s+inst_count:\s+(\d+)',
            line
        )
        if m:
            core_id = int(m.group(1))
            inst_type = m.group(2)
            inst_count = int(m.group(3))

            core_metrics.setdefault("Instruction_Counts", {})
            core_metrics["Instruction_Counts"].setdefault(core_id, {})
            core_metrics["Instruction_Counts"][core_id][inst_type] = inst_count

            if inst_type == "COMP":
                gemm = re.search(r'GEMM:\s*(\d+)', line)
                vector = re.search(r'Vector:\s*(\d+)', line)

                if gemm:
                    core_metrics["Instruction_Counts"][core_id]["COMP_GEMM"] = int(gemm.group(1))
                if vector:
                    core_metrics["Instruction_Counts"][core_id]["COMP_Vector"] = int(vector.group(1))

            continue

        # Systolic array stats
        m = re.search(
            r'Core \[(\d+)\] : Systolic array \[(\d+)\] utilization\(%\): ([\d.]+), active_cycles: (\d+), idle_cycles: (\d+)',
            line
        )
        if m:
            core_id = int(m.group(1))
            sa_id = int(m.group(2))
            util = float(m.group(3))
            active = int(m.group(4))
            idle = int(m.group(5))

            core_metrics.setdefault("Systolic_Arrays", {})
            core_metrics["Systolic_Arrays"][(core_id, sa_id)] = {
                "utilization": util,
                "active_cycles": active,
                "idle_cycles": idle,
            }

            # 기존 리포트 호환용
            if sa_id == 0:
                core_metrics["Systolic_Array_Utilization"] = util
                core_metrics["MatMul_active_cycle"] = active

            continue

        # DMA stats
        m = re.search(
            r'Core \[(\d+)\] : DMA active_cycles: (\d+), DMA idle_cycles: (\d+), DRAM BW: ([\d.]+) GB/s \((\d+) responses\)',
            line
        )
        if m:
            core_id = int(m.group(1))

            core_metrics.setdefault("DMA", {})
            core_metrics["DMA"][core_id] = {
                "active_cycles": int(m.group(2)),
                "idle_cycles": int(m.group(3)),
                "dram_bw_gbps": float(m.group(4)),
                "responses": int(m.group(5)),
            }
            continue

        # Vector unit stats
        m = re.search(
            r'Core \[(\d+)\] : Vector unit utilization\(%\): ([\d.]+), active cycle: (\d+), idle_cycle: (\d+)',
            line
        )
        if m:
            core_id = int(m.group(1))
            util = float(m.group(2))
            active = int(m.group(3))
            idle = int(m.group(4))

            core_metrics.setdefault("Vector_Unit", {})
            core_metrics["Vector_Unit"][core_id] = {
                "utilization": util,
                "active_cycles": active,
                "idle_cycles": idle,
            }

            # 기존 리포트 호환용
            core_metrics["Vector_Unit_Utilization"] = util
            core_metrics["Vector_active_cycle"] = active

            continue

        # NUMA stats
        m = re.search(
            r'Core \[(\d+)\] : NUMA local memory: (\d+) requests, remote memory: (\d+) requests',
            line
        )
        if m:
            core_id = int(m.group(1))

            core_metrics.setdefault("NUMA", {})
            core_metrics["NUMA"][core_id] = {
                "local_requests": int(m.group(2)),
                "remote_requests": int(m.group(3)),
            }
            continue

        # Core total cycles
        m = re.search(
            r'Core \[(\d+)\] : Total_cycles: (\d+)',
            line
        )
        if m:
            core_id = int(m.group(1))

            core_metrics.setdefault("Core_Total_Cycles", {})
            core_metrics["Core_Total_Cycles"][core_id] = int(m.group(2))
            continue

        # Global total cycles
        m = re.search(
            r'Total execution cycles: (\d+)',
            line
        )
        if m:
            total_cycle = int(m.group(1))
            core_metrics["Total_cycle"] = total_cycle
            continue

        # Simulation time
        m = re.search(
            r'Wall-clock time for simulation: ([\d.]+) seconds',
            line
        )
        if m:
            simulation_time = float(m.group(1))
            continue

    # 기본값 보장
    core_metrics.setdefault("MatMul_active_cycle", 0)
    core_metrics.setdefault("Vector_active_cycle", 0)
    core_metrics.setdefault("Systolic_Array_Utilization", 0.0)
    core_metrics.setdefault("Vector_Unit_Utilization", 0.0)
    core_metrics.setdefault("Total_cycle", total_cycle)

    return (
        core_metrics,
        dram_channel_bw,
        avg_dram_bw,
        simulation_time,
        total_cycle,
    )

def get_latest_log_result():
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    if not log_files:
        raise RuntimeError("No log files found in " + LOG_DIR)
    latest_file = max(log_files, key=os.path.getmtime)
    print(f"Parsing timing metrics from: {latest_file}")
    
    # Parse metrics
    core_metrics, dram_ch_bw, avg_dram_bw, sim_time, total_cycle = get_result_from_file(latest_file)
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
    result = {"cpu_cycles": 0, "cpi": 0}
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
                    cpi_m = re.search(r"system\.cpu\.cpi\s+([\d.]+)", last_section)
                    if cycles_m:
                        result["cpu_cycles"] = int(cycles_m.group(1))
                    if cpi_m:
                        result["cpi"] = float(cpi_m.group(1))
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
    
    import json
    
    tests = []
    
    # 1. Matmul 256x256
    a_tf = tf.random.normal([256, 256], seed=42)
    b_tf = tf.random.normal([256, 256], seed=43)
    a_pt = torch.tensor(a_tf.numpy())
    b_pt = torch.tensor(b_tf.numpy())
    @tf.function(jit_compile=True)
    def matmul_tf(x, y): return tf.matmul(x, y)
    tests.append(("Matmul 256x256", matmul_tf, [a_tf, b_tf], [a_pt, b_pt], lambda x, y: torch.matmul(x, y)))
    
    # 2. Relu 32x32
    r_tf = tf.random.normal([32, 32], seed=44)
    r_pt = torch.tensor(r_tf.numpy())
    @tf.function(jit_compile=True)
    def relu_tf(x): return tf.nn.relu(x)
    tests.append(("Relu 32x32", relu_tf, [r_tf], [r_pt], lambda x: torch.relu(x)))
    
    # 3. Dense Relu 32x32
    @tf.function(jit_compile=True)
    def dense_relu_tf(x, y): return tf.nn.relu(tf.matmul(x, y))
    tests.append(("Dense Relu 32x32", dense_relu_tf, [a_tf[:32,:32], b_tf[:32,:32]], [a_pt[:32,:32], b_pt[:32,:32]], lambda x, y: torch.relu(torch.matmul(x, y))))
    
    results_log = []
    
    for name, tf_fn, inputs_tf, inputs_torch, dummy_fn in tests:
        print(f"\n[{name}]")
        torch_results = run_torch_native_timing(inputs_torch, dummy_fn)
        tf_results = run_tf_npu_timing(tf_fn, inputs_tf, inputs_torch, dummy_fn)
        
        # Log to JSON
        results_log.append({
            "Test_Name": name,
            "Torch_Cycles": torch_results.get("total_cycle", 0),
            "TF_Cycles": tf_results.get("total_cycle", 0),
            "Torch_Matmul_Active": torch_results.get("MatMul_active_cycle", 0),
            "TF_Matmul_Active": tf_results.get("MatMul_active_cycle", 0),
            "Torch_Vector_Active": torch_results.get("Vector_active_cycle", 0),
            "TF_Vector_Active": tf_results.get("Vector_active_cycle", 0),
            "Torch_Systolic_Util": torch_results.get("Systolic_Array_Utilization", 0),
            "TF_Systolic_Util": tf_results.get("Systolic_Array_Utilization", 0),
            "Torch_Vector_Util": torch_results.get("Vector_Unit_Utilization", 0),
            "TF_Vector_Util": tf_results.get("Vector_Unit_Utilization", 0),
            "Torch_DRAM_BW": torch_results.get("avg_dram_bw", 0),
            "TF_DRAM_BW": tf_results.get("avg_dram_bw", 0),
            "Torch_Gem5_CPU": torch_results.get("gem5", {}).get("cpu_cycles", 0),
            "TF_Gem5_CPU": tf_results.get("gem5", {}).get("cpu_cycles", 0),
            "Torch_Gem5_CPI": torch_results.get("gem5", {}).get("cpi", 0),
            "TF_Gem5_CPI": tf_results.get("gem5", {}).get("cpi", 0)
        })
        
        print(f"{name} -> Native Cycles: {results_log[-1]['Torch_Cycles']}, TF Cycles: {results_log[-1]['TF_Cycles']}")
        
    with open("timing_results.json", "w") as f:
        json.dump(results_log, f, indent=4)
        
    print("\nSUCCESS: Timing simulation executed for all paths! Results saved to timing_results.json.")
    sys.exit(0)
