import torch
from unittest.mock import MagicMock
from torch._inductor.virtualized import V

mock_graph = MagicMock()
mock_graph.get_current_device_or_throw.return_value = torch.device('npu:0')
mock_graph.get_dtype = lambda name: torch.float32

try:
    with V.set_graph_handler(mock_graph):
        print("V.graph inside block:", V.graph)
except Exception as e:
    print("Error:", e)
