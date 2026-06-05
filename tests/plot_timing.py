import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    if not os.path.exists("timing_results.json"):
        print("Error: timing_results.json not found.")
        return
        
    with open("timing_results.json", "r") as f:
        data = json.load(f)
        
    if not data:
        print("Error: JSON is empty.")
        return

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
    plt.savefig("togsim_cycles_comparison.png", dpi=300)
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
    plt.savefig("hardware_util_comparison.png", dpi=300)
    plt.close()
    
    print("Successfully generated 'togsim_cycles_comparison.png' and 'hardware_util_comparison.png'.")

if __name__ == "__main__":
    plot_results()
