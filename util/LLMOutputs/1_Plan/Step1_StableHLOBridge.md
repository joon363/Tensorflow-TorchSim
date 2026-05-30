# TensorFlow Integration for PyTorchSim (StableHLO Bridge)

## Overview

This plan outlines the steps to integrate TensorFlow support into `PyTorchSim` by creating a translation layer that converts **StableHLO** (TF's intermediate representation) into the specific hardware-optimized MLIR format required by the existing simulator.

The strategy focuses on maximum reuse of the existing `mlir_codegen_backend.py` and `ExtensionOverrides` logic, avoiding a rewrite of the hardware generation core.

## Current State Analysis

- **Existing Pipeline:** PyTorch `FX Graph` → `TorchInductor` → `Aten IR` → `codegen_loops()` → Custom MLIR.
- **Target Input:** TensorFlow → XLA → **StableHLO**.
- **The Gap:**
    - StableHLO is high-level (tensor-based, no memory hierarchy).
    - PyTorchSim backend requires low-level specifics (tiling info, SRAM addresses, explicit DMAs) usually provided by the PyTorch Inductor Scheduler.
- **Key Asset:** `mlir_codegen_backend.py` contains the vital logic for converting abstract math into HW-specific instructions (DMA/Vector), but it currently expects PyTorch `SchedulerNode` objects.

## Desired End State

A functional pipeline where a user can provide a StableHLO file (or string), and the system generates the same valid, tiled, DMA-explicit MLIR that the PyTorch frontend currently produces.

### Key Discoveries:

- **Codegen Logic:** `codegen_loops()` in `CodegenResearch.md` strictly enforces the dataflow (Alloc -> Outer Loop -> DMA Load -> Compute -> DMA Store). We must respect this order.
- **Abstraction Gap:** StableHLO lacks the concept of "Outer Loop" (Tiling) and "Memory Space" (SRAM vs DRAM). We must inject this information before calling the backend.

## What We're NOT Doing

- **Full Autotuning:** We will not implement a complex search algorithm for optimal tile sizes in Phase 1. We will use fixed/heuristic tiling.
- **Complex Control Flow:** We are focusing on static computational graphs (Add/Mul/MatMul), not dynamic control flow (If/While) inside the kernels yet.
- **Re-implementing XLA:** We assume the input is already valid, optimized StableHLO (e.g., output from `stablehlo-opt`).

## Implementation Approach

We will implement an **"Adapter Pattern"**. Instead of modifying the core backend to understand StableHLO, we will create a `StableHLOAdapter` that parses StableHLO and generates "Mock Nodes" that mimic the interface of the PyTorch `SchedulerNode` and `Buffer` objects expected by `mlir_codegen_backend.py`.

## Phase 1: The "Manual Driver" (Proof of Concept)

### Overview

Before writing a parser, we must prove we can drive the `MLIRKernel` and `codegen_loops` manually without `torch.compile` or `FX Graph`. We will write a standalone script that instantiates the backend classes and forces them to generate code for a simple "Add" operation.

### Changes Required:

### 1. Backend Decoupling Script

**File**: `tests/manual_backend_driver.py` (New File)

**Changes**:

- Instantiate `MLIRKernel` (from `mlir_codegen_backend.py`).
- Manually populate `kernel.args` (inputs/outputs).
- Manually trigger the `codegen_loops` logic by mocking the data structures it usually reads from the scheduler.

```
# Pseudo-code for manual driver
from mlir_codegen_backend import MLIRKernel
from mlir_ops import ExtensionOverrides

def test_manual_generation():
    # 1. Setup Kernel Context
    kernel = MLIRKernel()

    # 2. Define Mock Nodes (mimicking FX Graph Nodes/Scheduler Nodes)
    # We need to simulate: "Load A", "Load B", "Add", "Store C"
    # This proves we can inject data without a real FX Graph

    # 3. Define Tiling Config (Hardcoded for POC)
    # e.g., Total=128, Tile=64

    # 4. Call generation
    # kernel.codegen_loops(...)

    print(kernel.render())

```

### Success Criteria:

### Manual Verification:

- [ ]  Script runs without importing `torch` (or at least without using `torch.compile`).
- [ ]  Output MLIR matches the "PyTorchSim Hand-Written MLIR" format in `Research.md`.
- [ ]  Generated code includes `memref.dma_start` and `affine.vector_load/store`.

**Implementation Note**: Pause here. If we cannot drive the backend manually, we cannot support TF. Solve dependency injection issues here first.

## Phase 2: StableHLO Parser & IR Normalization

### Overview

We need to parse the text-based StableHLO MLIR and extract the operation types (`stablehlo.add`), data types (`f32`), and shapes (`tensor<2x2>`).

### Changes Required:

### 1. StableHLO Parser Module

**File**: `frontend/stablehlo_parser.py` (New File)

**Changes**:

- Implement a lightweight parser (or use `MLIR-Python-Bindings` if available, but text parsing is often sufficient for MVP).
- Create a `SimNode` class to hold the parsed data (Op type, Inputs, Output Shape).

```
class SimNode:
    def __init__(self, op_type, inputs, name, shape, dtype):
        self.op_type = op_type # e.g., "stablehlo.add"
        self.inputs = inputs   # List of other SimNodes
        self.name = name
        self.shape = shape
        self.dtype = dtype

def parse_stablehlo(mlir_text):
    # Iterate lines, regex capture "stablehlo.op", create SimNodes
    # Return graph (list of SimNodes topologically sorted)
    pass

```

### 2. Op Mapper

**File**: `frontend/op_mapper.py` (New File)

**Changes**:

- Map `stablehlo.add` → `ExtensionOverrides.add`.
- Map `stablehlo.multiply` → `ExtensionOverrides.mul`.
- Map `stablehlo.convolution` → Custom Kernel Call.

### Success Criteria:

### Manual Verification:

- [ ]  Parser correctly identifies inputs/outputs of the `add_fn` example from `Research.md`.
- [ ]  Parser correctly extracts shapes (`2xf32`).

## Phase 3: The "Mock Scheduler" (Tiling & Memory Assignment)

### Overview

This is the most critical phase. StableHLO provides the *what* (Math), but PyTorchSim needs the *how* (Tiling/Memory). We will implement a `StaticScheduler` that takes the `SimNodes` from Phase 2 and wraps them in objects that look like PyTorch's `SchedulerNode`.

### Changes Required:

### 1. Scheduler Adapter

**File**: `bridge/tf_scheduler_adapter.py` (New File)

**Changes**:

- Create `MockSchedulerNode` class.
- Implement methods required by `codegen_loops` (e.g., `.get_ranges()`, `.get_buffer_name()`).
- Implement a simple tiling strategy:
    - If `shape > 4096`: Split into tiles of size 1024 (example).
    - If `shape < 4096`: Single tile.
- Assign memory spaces:
    - Inputs/Outputs → DRAM (Space 0).
    - Intermediates → SPAD (Space 1).

```
class MockSchedulerNode:
    def __init__(self, sim_node, tiling_config):
        self.node = sim_node
        # These fields are accessed by mlir_codegen_backend.py
        self.computed_buffer = ...
        self.ranges = tiling_config

    def get_op_name(self):
        return mapper.map(self.node.op_type)

```