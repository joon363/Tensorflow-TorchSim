import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results(silent=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    json_path = os.path.join(out_dir, "timing_results.json")
    
    if not os.path.exists(json_path):
        if not silent: print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    if not data:
        if not silent: print("Error: JSON is empty.")
        return

    # Filter out cases where both cycles are 1
    filtered_data = []
    for item in data:
        if item.get("Torch_Cycles", 0) == 1 and item.get("TF_Cycles", 0) == 1:
            continue
        if item.get("Torch_Cycles", 0) == 0 and item.get("TF_Cycles", 0) == 0:
            continue
        filtered_data.append(item)
        
    if not filtered_data:
        if not silent: print("Error: Filtered JSON is empty (all cycles are 1).")
        return
        
    data = filtered_data

    test_names = [item["Test_Name"] for item in data]
    x = np.arange(len(test_names))
    width = 0.35

    # 1. Total Cycles Comparison
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    torch_cycles = [item["Torch_Cycles"] for item in data]
    tf_cycles = [item["TF_Cycles"] for item in data]
    
    rects1 = ax1.bar(x - width/2, torch_cycles, width, label='Native Torch', color='royalblue')
    rects2 = ax1.bar(x + width/2, tf_cycles, width, label='TF NPU Codegen', color='darkorange')
    
    ax1.set_ylabel('Total TOGSim Cycles')
    ax1.set_title('TOGSim Cycle Comparison by Test Case')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "togsim_cycles_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Hardware Utilization Metrics (Systolic Array, Vector, DRAM BW)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Systolic Array Util
    torch_sys = [item["Torch_Systolic_Util"] for item in data]
    tf_sys = [item["TF_Systolic_Util"] for item in data]
    ax1.bar(x - width/2, torch_sys, width, label='Native Torch', color='royalblue')
    ax1.bar(x + width/2, tf_sys, width, label='TF NPU Codegen', color='darkorange')
    ax1.set_ylabel('Systolic Array Util (%)')
    ax1.set_title('Hardware Utilization Metrics')
    ax1.legend()
    
    # Vector Unit Util
    torch_vec = [item["Torch_Vector_Util"] for item in data]
    tf_vec = [item["TF_Vector_Util"] for item in data]
    ax2.bar(x - width/2, torch_vec, width, label='Native Torch', color='royalblue')
    ax2.bar(x + width/2, tf_vec, width, label='TF NPU Codegen', color='darkorange')
    ax2.set_ylabel('Vector Unit Util (%)')
    
    # DRAM BW
    torch_dram = [item["Torch_DRAM_BW"] for item in data]
    tf_dram = [item["TF_DRAM_BW"] for item in data]
    ax3.bar(x - width/2, torch_dram, width, label='Native Torch', color='royalblue')
    ax3.bar(x + width/2, tf_dram, width, label='TF NPU Codegen', color='darkorange')
    ax3.set_ylabel('DRAM BW Util (%)')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names, rotation=45, ha='right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "hardware_util_comparison.png"), dpi=300)
    plt.close()
    
    if not silent:
        print("Successfully generated 'togsim_cycles_comparison.png' and 'hardware_util_comparison.png'.")

if __name__ == "__main__":
    plot_results()
