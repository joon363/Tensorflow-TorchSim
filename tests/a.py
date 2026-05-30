import torch
import os
import sys
base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_dir)
os.environ['TOGSIM_CONFIG']=f"{base_dir}/tutorial/session1/togsim_configs/togsim_config_timing_only.json"
os.environ['TORCHSIM_DUMP_LOG_PATH']=os.path.join(os.getcwd(), "togsim_results")
os.environ['TOGSIM_DEBUG_LEVEL']="trace"

from Scheduler.scheduler import PyTorchSimRunner
device = PyTorchSimRunner.setup_device().custom_device()

a = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
b = torch.tensor([3.0, 4.0], dtype=torch.float32, device=device)

opt_fn = torch.compile(dynamic=False)(torch.add)
npu_out = opt_fn(a,b)