# Unified StableHLO-Based Backend Pipeline

## Overview

This plan restructures the **PyTorchSim backend** to use **StableHLO** as its native **Intermediate Representation (IR)**.  
Instead of creating a generic wrapper, the backend **explicitly adopts StableHLO semantics** as the standard IR for simulation and code generation.

This moves the project toward a **unified compilation pipeline**:
TensorFlow : StableHLO → Scheduler → Backend
PyTorch : (Future) FX Graph → StableHLO → Scheduler → Backend

---

## Current State Analysis

### Dependency
- `mlir_codegen_backend.py` currently depends on  
  `torch.inductor.scheduler.SchedulerNode`
- Uses it for:
  - Loop ranges
  - Buffer names
  - Operation types

### Missing Link
- **StableHLO provides**:
  - Compute graph
  - Ops
  - Shapes and types
- **StableHLO lacks**:
  - Execution semantics (tiling, loops, memory movement, DMA)
- These execution details were previously provided by `SchedulerNode`

---

## Desired End State

### Core Concept
The backend consumes a **ScheduledHloNode**, which fully describes both:

- **What to compute** (StableHLO semantics)
- **How to execute** (tiling, loops, memory hierarchy)

### ScheduledHloNode Contains

#### StableHLO Op Info
- Op name (e.g. `stablehlo.add`)
- Input / output shapes
- Data types

#### Scheduling Info (Injected by Our Scheduler)
- Tile sizes
- Loop ranges
- Memory placement (DRAM / SPAD)
- DMA behavior

### Backend
- Refactored to generate code **directly from `stablehlo.*` ops**
- No dependency on PyTorch-specific scheduling logic

---

## Implementation Plan


## Phase 1: Define Core IR (`ScheduledHloNode`)

### Overview
We introduce a new IR that combines **StableHLO semantics** with **explicit tiling metadata**.  
This replaces all reliance on PyTorch’s `SchedulerNode`.

---

### 1. Node Structure Definition

**File** core/hlo_ir.py (NEW)

**Class: `ScheduledHloNode`**

| Field | Type | Description |
|-----|-----|------------|
| `op_type` | `str` | StableHLO op name (e.g. `stablehlo.add`) |
| `inputs` | `List[HloBuffer]` | Input buffers |
| `outputs` | `List[HloBuffer]` | Output buffers |
| `ranges` | `List[int]` | Loop bounds (e.g. `[1024, 1024]`) |
| `tile_sizes` | `List[int]` | Tile sizes per dimension (e.g. `[64, 64]`) |

This node represents a **tiled, lowered form** of a StableHLO op.

---

### 2. Buffer Structure Definition

**File** core/hlo_ir.py


**Class: `HloBuffer`**

| Field | Description |
|-----|-------------|
| `name` | Buffer identifier |
| `dtype` | Element type |
| `shape` | Tensor shape |
| `memory_space` | `DRAM = 0`, `SPAD = 1` |


### Success Criteria

- [ ] `ScheduledHloNode` can fully describe the **hand-written MLIR logic** in `Research.md`

---

## Phase 2: Static Scheduler (Lowering)

### Overview
StableHLO has **no concept of loops or tiles**.  
We introduce a **static scheduler pass** that injects execution semantics.

This replaces the functionality previously provided by TorchInductor.

---

### 1. StableHLO Parser

**File**
frontend/tf/stablehlo_parser.py

**Responsibilities**
- Parse textual StableHLO MLIR
- Produce lightweight Python objects:
  - Op
  - Shape
  - Type

---

### 2. Static Scheduler Logic

**File** compiler/hlo_scheduler.py (NEW)


#### Input
- Parsed StableHLO ops

#### Output
- `List[ScheduledHloNode]`

#### Heuristic Rules

**Tiling**
- If dimension > 128 → tile size = 64
- Else → tile size = full dimension

**Memory Allocation**
- Inputs / outputs → DRAM
- Intermediate tile buffers → SPAD

**Loop Generation**
loop_range = total_shape / tile_size


---

### Success Criteria

- [ ] `hlo_scheduler.py` converts the `add_fn` StableHLO (from `Research.md`)
- [ ] Produces:
ranges = [2]
tile_sizes = [2]


---

## Phase 3: Backend Refactoring (Native StableHLO)

### Overview
The backend is refactored to consume **ScheduledHloNode directly**, making it **StableHLO-native**.

---

### 1. Codegen Signature Update

**File**backend/mlir_codegen_backend.py


#### Change
```diff
- node: SchedulerNode
+ node: ScheduledHloNode
```
### Op Mapping

- Remove PyTorch op lookup
- Dispatch directly on StableHLO ops

**Example**

```
if node.op_type == "stablehlo.add":
    emit(ExtensionOverrides.add)
```
---

### 2. PyTorch Compatibility Adapter (Temporary)

**File** 
frontend/torch/legacy_adapter.py

### Purpose

- Keep existing PyTorch tests passing
- Acts as a bridge during migration

### Function

- Convert `SchedulerNode` → `ScheduledHloNode`
- Map:
    
    ```
    aten.add → stablehlo.add
    ```
    

---

### Success Criteria

- [ ]  Backend generates **valid RISC-V MLIR** from `ScheduledHloNode`
- [ ]  Generated code strictly follows:
    
    ```
    DRAM → SPAD → Compute
    ```
    
- [ ]  Matches the hierarchy defined in `CodegenResearch.md`


## Phase 4: Verification

### Overview

Validate the **TensorFlow → StableHLO → Backend** pipeline end-to-end.

---

### Integration Test

**File**

```
tests/test_tf_stablehlo.py
```

### Input

- StableHLO string from `Research.md`

### Pipeline

```
StableHLO Parser
→ Static Scheduler
→ Backend Codegen
```

### Verification

Check generated MLIR contains:

- `memref.dma_start`
- `affine.for`
- Vector operations

---

## References

- **Research.md**
    - Defines StableHLO input
    - Defines target HW-specific MLIR output
- **CodegenResearch.md**
    - Defines required loop nesting order
    - Defines memory movement semantics