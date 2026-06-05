import torch
import os
import sys
base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_dir)
os.environ['TOGSIM_CONFIG']=f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_timing_only.yml"
os.environ['TORCHSIM_LOG_PATH'] = os.path.join(os.getcwd(), "togsim_results")
os.environ['TENSORFLOW_MLIR_DIRECT_TEST'] = "False"

device = torch.device("npu:0")

a = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
b = torch.tensor([3.0, 4.0], dtype=torch.float32, device=device)

opt_fn = torch.compile(dynamic=False)(torch.add)
npu_out = opt_fn(a,b)

# root@09058bde5670:/workspace/PyTorchSim/Tensorflow/tests# python a.py
# [2026-06-05 05:17:42.792] [INFO] [pytorchsimfrontend.mlir.generated_wrapper] Wrapper Codegen Path = /workspace/PyTorchSim/outputs/.torchinductor/z7/cz7ea5yj2bekfifgdr25aqrmlamk24r6mwrsoy4jzwgqjatx3vw6.py
# [2026-06-05 05:17:42.835] [INFO] [simulator.simulator] [TOGSim] TOGSim simulation started
# [2026-06-05 05:17:42.855] [INFO] [simulator.simulator] [TOGSim] Simulation log is stored to "/workspace/PyTorchSim/Tensorflow/tests/togsim_results/20260605_051742_36d9849c.log"