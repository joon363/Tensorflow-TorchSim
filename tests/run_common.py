import sys
import os
sys.path.append(os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim'))

from tests_common import tests
import test_correctness
import torch

device = torch.device("npu:0")
test_correctness.run_tf_test(*tests[0], device)
