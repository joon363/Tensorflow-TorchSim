
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.automatic_dynamic_shapes = False

torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.2.0
# torch cuda version: 12.1
# torch git version: 8ac9b20d4b090c213799e81acf48a55ea8d437d6


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (add,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8, device=device(type='npu', index=0))
    reader.tensor(buf0, (2,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8, device=device(type='npu', index=0))
    reader.tensor(buf1, (2,), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
