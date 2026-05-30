```python
opt_fn = torch.compile(dynamic=False)(exponent2)
res = opt_fn(x)
```

1. `torch/__init__.py`의 `compile`
2. `torch/_dynamo/eval_frame.py`의 `optimize`
3. **[torch code → FX Graph]** 
`torch/_dynamo/convert_frame.py`의  `_compile` 
4. **[inductor 진입점]**
`torch/_dynamo/symbolic_convert.py`의 `InstructionTranslatorBase`의  `RETURN_VALUE`
5. `torch/_dynamo/output_graph.py`의 `OutputGraph`의
 `compile_subgraph` →`compile_and_call_fx_graph`**→** `call_user_compiler`
6. `torch/__init__.py`의 `_TorchCompileInductorWrapper` 의 `__call__` 
7. `torch/_inductor/compile_fx.py`의 
`compile_fx` → `compile_fx_inner` → `fx_codegen_and_compile` 
    1. `graph.**run**`
        1. `torch/fx/Interpreter.py` 의 run
        2. `torch/_inductor/graph.py의 GraphLowering`의 `run_node`
        3. `torch/_inductor/graph.py`의 `GraphLowering`의 
        `call_function`, `call_module`, `call_method`, `get_attr`중 하나
        4. **[FX Graph → Buffer instance with Loop-Level TorchInductor IR]**
        `lowerings[target]`
            1. 이때 실행되는 것: `torch/_inductor/virtualized.py`의 `OpsWrapper`의 `__getattr__`
            2. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_common.py`의 `CSEProxy`의 `__getattr__`
            `load`, `indirect_load`, `store_reduction`, `store`, `redution`, `bucketize` 중 하나
        5. **[Loop-Level TorchInductor IR → MLIR]**
            1. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_codegen_backend.py`의 
            `MLIRKernel` 또는 `ExtensionOverrides` 중 하나
            `ops.mul`, `ops.store`, … → `arith.mul` …
            2. `torch/_inductor/codegen/common.py`의 `CSE`의 `generate`
            → `%tmp4 = arith.mulf ...`
    2. `graph.compile_to_fn()`
        1. `torch/_inductor/graph.py`의 `compile_to_module` → `codegen`
        2. `torch/_inductor/scheduler.py`의 `Scheduler`의 `codegen`
        3. `self.get_backend(device).codegen_nodes`
        4. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_scheduling.py`의 `MLIRScheduling`의 `codegen_nodes`
            1. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_codegen_backend.py`의 
            `MLIRKernel.codegen_nodes`
                1. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_common.py`의 `BaseMLIRKernel`의 
                `codegen_nodes`
                    1. `torch/_inductor/scheduler.py`의 `SchedulerNode`의 `codegen`
                    2. **[Wrapper Codegen 파이썬 코드 생성]**
                    `PyTorchSim/PyTorchSimFrontend/mlir/m1lir_codegen_backend.py`의 `MLIRKernel`의 `codegen_loops`
                    3. `meta_kernel`
            2. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_scheduling.py`의 `MLIRScheduling`의 `define_kernel`
            3. `PyTorchSim/PyTorchSimFrontend/mlir/mlir_codegen_backend.py`의 
            `MLIRKernel.call_kernel`

1. 방금 생성한 코드는 `compile_to_module`에 들어가 있다.
`torch/_inductor/graph.py`의 `GraphLowering`의 `compile_to_fn(self)`에서
`return self.compile_to_module().call`
2. `<generated code>.py` 의 `call`
3. `PyTorchSim/PyTorchSimFrontend/extension_codecache.py`의 `CustomAsyncCompile.run`
4. `PyTorchSim/PyTorchSimFrontend/extension_codecache.py`의 `MLIRCodeCache.load`
    1. if Functional Mode
        1. **[MLIR → LLVM IR → RISC-V] (for functional)**
        `PyTorchSim/PyTorchSimFrontend/extension_codecache.py`의 `mlir_compile_command` 
        2. **[RISC-V 실행 파일 생성]**
        `PyTorchSim/PyTorchSimFrontend/mlir/mlir_caller_codegen.py`의 `MLIRKernelCallerCodeGen`의 `compile_wih_kernel` 
    2. else if Timing Mode
        1. **[MLIR → LLVM IR → RISC-V] (for timing)**
        `PyTorchSim/PyTorchSimFrontend/extension_codecache.py`의 `mlir_gem5_compile_command`
        2. **[RISC-V 실행 파일 생성]**
        `PyTorchSim/PyTorchSimFrontend/mlir/mlir_caller_codegen.py`의 `MLIRKernelCallerCodeGen`의 `compile_wih_kernel` 
        3. **[Gem5 Timing 시뮬레이션]**
        `PyTorchSim/Simulator/simulator.py`의  `CycleSimulator`의 `compile_and_simulate`
5. 
    1. If Functional Mode
        1. **[Spike Functional 시뮬레이션]**
        `PyTorchSim/PyTorchSimFrontend/extension_codecache.py` 의 `dummy_simulator` 중 `run_spike` 
    2. Else
        1. **[TOGSim 시뮬레이션]**
        `PyTorchSim/PyTorchSimFrontend/extension_codecache.py` 의 `dummy_simulator` 중 `run_spike`