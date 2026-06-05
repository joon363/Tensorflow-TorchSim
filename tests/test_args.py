import sys
import os
import torch
from unittest.mock import MagicMock

sys.path.append('/workspace/PyTorchSim')

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
from Tensorflow.TensorFlowFrontend.tf_npu_codegen import MockNode, mock_graph
from torch._inductor.virtualized import V

dtype = torch.float32
X = MockNode("X", [32, 32], dtype)
W = MockNode("W", [32, 32], dtype)
Y = MockNode("Y", [32, 32], dtype)

mock_graph.buffers = [Y]
mock_graph.graph_inputs = { "X": X, "W": W }
mock_graph.constants = {}

with V.set_graph_handler(mock_graph):
    kernel = MLIRTemplateKernel(
        kernel_name="test_kernel",
        input_nodes=[X, W],
        call_size=[32, 32]
    )
    
    # Let's call def_kernel directly
    print("Calling def_kernel...")
    res = kernel.def_kernel(inputs=[X, W, None], outputs=[Y], names_str="X, W, Bias, Y")
    print("def_kernel returned:", res)
    
    # Evaluate the hook inside context manager
    with kernel:
        hook_fn = kernel.render_hooks["<DEF_KERNEL>"][1]
        signature = hook_fn()
        print("Generated signature inside with kernel:", signature)
        print("output_buffers inside with kernel:", kernel.kernel_group.args.output_buffers)
