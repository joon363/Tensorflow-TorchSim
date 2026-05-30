
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from PyTorchSimFrontend.extension_codecache import CustomAsyncCompile
from PyTorchSimFrontend.extension_config import CONFIG_SRAM_BUFFER_PLAN, CONFIG_TOGSIM_EAGER_MODE
from Simulator.simulator import TOGSimulator
from PyTorchSimFrontend.extension_op import sparse_mm_dummy_stonne_outer
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.aten._reinterpret_tensor
custom_async_compile = CustomAsyncCompile()
os.environ["TORCHSIM_LAST_COMPILED_MODULE"] = __file__

def sram_plan_prefix(buffer_name, buffer):
    if CONFIG_SRAM_BUFFER_PLAN and (buffer_name not in CONFIG_SRAM_BUFFER_PLAN):
        return
    buffer_size = buffer.untyped_storage().size()
    start = buffer.data_ptr()
    end = start + buffer_size
    # print(f'Alloc {buffer_name}(0x{start:x} ~ 0x{end:x})')
    TOGSimulator.sram_alloc(buffer_name, [start, end])

def sram_plan_postfix(buffer_name, buffer):
    if CONFIG_SRAM_BUFFER_PLAN and (buffer_name not in CONFIG_SRAM_BUFFER_PLAN):
        return
    buffer_size = buffer.untyped_storage().size()
    start = buffer.data_ptr()
    end = start + buffer_size
    # print(f'Dealloc {buffer_name}(0x{start:x} ~ 0x{end:x})')
    TOGSimulator.sram_dealloc(buffer_name, [start, end])

def host2device_memcopy(buffer):
    pass

def device2host_memcpy(buffer):
    pass

print(f'Wrapper Codegen Path = {__file__}')
arg_attributes = [['arg0_1', [1, torch.float32, 1, [1], [1]]], ['arg1_1', [1, torch.float32, 2, [2], [1]]], ['buf0', [2, torch.float32, 2, [2], [1]]]]


extension_kernel_0 = custom_async_compile.mlir('''memref.global @buf0_spad : memref<2xf32, 1>
memref.global @buf1_spad : memref<2xf32, 1>
memref.global @buf2_spad : memref<2xf32, 1>
func.func @kernel(%in_ptr0: memref<1xf32>,
                       %in_ptr1: memref<2xf32>,
                       %out_ptr0: memref<2xf32>)
{
    %const0 = arith.constant 0 : index
    %const1 = arith.constant 2 : index
    %const2 = arith.constant 3 : index
    %alloc0 = memref.alloc() : memref<1xi32> // 0
    %alloc1 = memref.alloc() : memref<1xi32> // 1
    %alloc2 = memref.alloc() : memref<1xi32> // 2
    %spad0 = memref.get_global @buf0_spad : memref<2xf32, 1>
    %spad1 = memref.get_global @buf1_spad : memref<2xf32, 1>
    %spad2 = memref.get_global @buf2_spad : memref<2xf32, 1>
    affine.for %index0 = 0 to 2 step 2
    {
        memref.dma_start %in_ptr0[%const0], %spad0[%const0], %const1, %alloc0[%const0], %const0, %const1 : memref<1xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[0], sram_stride=[1], padding=0}
        memref.dma_start %in_ptr1[%index0], %spad1[%const0], %const1, %alloc1[%const0], %const0, %const1 : memref<2xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
        affine.for %compute_idx = 0 to 2 step 2
        {
            %tmp0 = affine.vector_load %spad0[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp1 = affine.vector_load %spad1[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp2 = arith.addf %tmp0, %tmp1 : vector<2xf32>
            affine.vector_store %tmp2, %spad2[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
        } {inner_loop=false}
        memref.dma_start %spad2[%const0], %out_ptr0[%index0], %const2, %alloc2[%const0], %const0, %const1 : memref<2xf32, 1>, memref<2xf32>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
    } {outer_loop=true}
    return
}
''', 
vectorlane_size=128,
loop_size=None,
spad_info={'spad_vaddr': 3489660928, 'spad_paddr': 137438953472, 'spad_size': 131072},
origins={'add'},
arg_attributes=arg_attributes,
vlen=256)

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (2, ), (1, ))
    sram_plan_prefix('arg0_1', arg0_1)
    sram_plan_prefix('arg1_1', arg1_1)
    buf0 = empty((2, ), device='npu', dtype=torch.float32)
    sram_plan_prefix('buf0', buf0)
    extension_kernel_0(arg0_1, arg1_1, buf0)
    sram_plan_postfix('arg0_1', arg0_1)
    del arg0_1
    sram_plan_postfix('arg1_1', arg1_1)
    del arg1_1
    sram_plan_postfix('buf0', buf0)
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, ), (1, ), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
