# Phase 1: Manual Backend Driver - Architecture & Extension Guide

## Overview

This document describes the Phase 1 implementation and provides guidance for implementing Phase 2 (StableHLO Parser) and Phase 3 (Mock Scheduler).

## Architecture

### Layer 1: MLIR Codegen Backend (Existing)
```
MLIRKernel
├── codegen_loops()        [Main entry point]
├── setup_for_add()        [Operation-specific setup]
├── _populate_add_operations()  [Manual operation population]
└── render_mlir_kernel()   [Final MLIR assembly]
```

### Layer 2: Mock/Adapter Infrastructure (Phase 1)
```
MockBuffer              [Simulates torch._inductor.ir.Buffer]
├── shape, dtype, stride
├── get_size(), get_dtype(), etc.
└── Matches PyTorchSim expectations

MockMLIRKernelArgs      [Simulates MLIRKernelArgs]
├── add_input/output()
├── mlir_argdefs()       [Generates MLIR function signatures]
└── Manages buffer metadata

MockKernelGroup         [Simulates KernelGroup]
├── args (MLIRKernelArgs)
├── tile_desc (MLIRMultiDimTile)
└── set_tile_info()

SimpleMLIRKernel        [Extends MLIRKernel for testing]
├── setup_for_add()      [Configure kernel]
├── _populate_add_operations()  [Generate ops]
└── render_mlir_kernel() [Output MLIR]
```

### Layer 3: Input Representation (Future - Phase 2/3)
```
StableHLOParser (Phase 2)
├── parse_stablehlo()
├── Extract ops, operands, shapes
└── Create SimNode graph

SimNode (Phase 2)
├── op_type (stablehlo.add, etc.)
├── inputs, output
├── shape, dtype
└── name

TFSchedulerAdapter (Phase 3)
├── Convert SimNode -> MockSchedulerNode
├── Implement tiling strategy
├── Manage memory assignment
└── Create objects that MLIRKernel expects
```

## Data Flow

```
Input (Phase 2)              Processing (Phase 2)         Backend (Phase 1)
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│  StableHLO Text     │      │  StableHLO Parser    │      │  MLIRKernel      │
│  ────────────────── │      │  ──────────────────  │      │  ──────────────  │
│  func @add_fn(...)  │  →   │  Extract ops         │  →   │  Generate loops  │
│  %0 = stablehlo.add │      │  Create SimNodes     │      │  Generate ops    │
│  ...                │      │  Build graph         │      │  Render MLIR     │
└─────────────────────┘      └──────────────────────┘      └──────────────────┘
                                      ↓
                             ┌─────────────────────────┐
                             │  Mock Scheduler (Ph 3)  │
                             │  ─────────────────────  │
                             │  Apply tiling          │
                             │  Assign memory         │
                             │  Create mock nodes     │
                             └─────────────────────────┘
```

## Phase 1: Manual Backend Driver

### What It Does
1. Creates mock objects that match PyTorchSim's expected interfaces
2. Instantiates `MLIRKernel` WITHOUT torch.compile
3. Manually populates load/compute/store operations
4. Generates valid MLIR code

### Key Classes

#### MockBuffer
```python
buffer = MockBuffer(
    name="tensor_a",
    dtype="f32",
    shape=[128, 128],
    stride=[128, 1]  # Optional, computed if not provided
)
```

Features:
- Matches `torch._inductor.ir.Buffer` interface
- Provides `get_name()`, `get_dtype()`, `get_numel()`, `get_size()`, `get_stride()`
- Supports any number of dimensions

#### MockMLIRKernelArgs
```python
args = MockMLIRKernelArgs()
args.add_input("tensor_a", buffer_a)
args.add_input("tensor_b", buffer_b)
args.add_output("tensor_c", buffer_c)

arg_defs, _, _, buffer_types = args.mlir_argdefs()
```

Features:
- Tracks input/output buffers
- Generates MLIR function argument definitions
- Manages buffer metadata (shape, stride, dtype)

#### SimpleMLIRKernel
```python
kernel_group = MockKernelGroup()
kernel = SimpleMLIRKernel(kernel_group)

kernel.setup_for_add(buf_a, buf_b, buf_c)
mlir_output = kernel.render_mlir_kernel(kernel_name="my_kernel")
```

Features:
- Extends `MLIRKernel` for testing
- Initializes tile descriptors
- Manually populates operations
- Renders final MLIR code

### Running Phase 1

```bash
cd /workspace/PyTorchSim/tests
python manual_backend_driver.py
```

Expected output: All 2 tests pass with MLIR generation

## Phase 2: StableHLO Parser (TODO)

### Objectives
1. Parse StableHLO text format
2. Extract operation types, operands, shapes, dtypes
3. Build a topologically sorted graph
4. Create `SimNode` objects

### File Structure
```
PyTorchSimFrontend/
├── stablehlo_parser.py      [New - Phase 2]
│   ├── StableHLOParser class
│   ├── parse_stablehlo()
│   └── SimNode class
└── op_mapper.py             [New - Phase 2]
    ├── map_op_type()
    └── Mapping: stablehlo.* → ExtensionOverrides.*
```

### Key Components

#### SimNode
```python
@dataclass
class SimNode:
    op_type: str           # "stablehlo.add", "stablehlo.multiply", etc.
    inputs: List[SimNode]  # Input operands
    output_shape: Tuple    # Output shape
    output_dtype: str      # "f32", "i64", etc.
    name: str              # Unique identifier
```

#### StableHLOParser
```python
parser = StableHLOParser()
graph = parser.parse_stablehlo(stablehlo_mlir_text)
# Returns: List[SimNode] topologically sorted

# Example:
# graph[0] = SimNode(op_type="stablehlo.add", inputs=[...], ...)
# graph[1] = SimNode(op_type="stablehlo.multiply", inputs=[graph[0]], ...)
```

### Implementation Strategy

1. **Text Parsing**: Use regex to extract operations
   - Match patterns like `%result = stablehlo.add %a, %b : tensor<2xf32>`
   - Extract op_type, operands, type information

2. **Type Resolution**: Infer shapes and dtypes
   - Parse tensor shapes from type signatures
   - Map StableHLO dtypes to PyTorchSim dtypes

3. **Graph Building**: Topological sort
   - Build dependency graph
   - Ensure inputs are processed before consumers

4. **Validation**: Check that all operations are supported
   - Error on unsupported ops (control flow, dynamic shapes)
   - Warn on operations without ExtensionOverrides mapping

### Example: Parsing Simple Add

Input StableHLO:
```
func.func @add_fn(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
  return %0 : tensor<2xf32>
}
```

Expected Output:
```python
[
    SimNode(
        op_type="stablehlo.add",
        inputs=[arg0, arg1],  # References to inputs
        output_shape=(2,),
        output_dtype="f32",
        name="%0"
    )
]
```

### Operation Mapping (op_mapper.py)

Map StableHLO ops to ExtensionOverrides:
```python
STABLEHLO_TO_OPS = {
    "stablehlo.add": "add",
    "stablehlo.multiply": "mul",
    "stablehlo.subtract": "sub",
    "stablehlo.divide": "truediv",
    "stablehlo.negate": "neg",
    "stablehlo.exponential": "exp",
    "stablehlo.sine": "sin",
    "stablehlo.cosine": "cos",
    # ... more mappings
}
```

## Phase 3: Mock Scheduler Adapter (TODO)

### Objectives
1. Convert `SimNode` graph to mock scheduler nodes
2. Implement tiling strategy
3. Assign memory spaces (DRAM vs SRAM)
4. Create objects matching PyTorchSim's `SchedulerNode` interface

### File Structure
```
bridge/
├── tf_scheduler_adapter.py  [New - Phase 3]
│   ├── StaticScheduler class
│   ├── MockSchedulerNode class
│   └── tiling_strategy()
└── memory_assignment.py     [New - Phase 3]
    ├── assign_memory_spaces()
    └── optimize_sram_usage()
```

### Key Components

#### MockSchedulerNode
```python
class MockSchedulerNode:
    def __init__(self, sim_node, tiling_config):
        self.node = sim_node
        self.computed_buffer = ...
        self.ranges = tiling_config.ranges
        self.group = (None, (None, None))
    
    def get_op_name(self):
        return op_mapper.map(self.node.op_type)
    
    def run(self, vars, reduction_vars):
        # Called by MLIRKernel to process node
        pass
```

#### StaticScheduler
```python
scheduler = StaticScheduler(sram_size=65536)
scheduler_nodes = scheduler.schedule(sim_graph)
# Returns: List of MockSchedulerNode ready for backend
```

### Tiling Strategy

For Phase 1 (PoC), use simple heuristics:
```python
def compute_tile_size(total_size, max_sram_elements):
    """
    Simple tiling: if size > 4096, split into tiles of 1024
    Otherwise single tile
    """
    if total_size > 4096:
        return 1024
    return total_size
```

For future: Implement search-based autotuning

### Memory Assignment

```python
MEMORY_SPACES = {
    0: "DRAM",    # Main memory
    1: "SRAM",    # Scratchpad
}

def assign_memory(buffer_type, buffer_size):
    if buffer_type == "input" or buffer_type == "output":
        return MEMORY_SPACES[0]  # DRAM
    elif buffer_type == "intermediate":
        if buffer_size < SRAM_LIMIT:
            return MEMORY_SPACES[1]  # SRAM
        else:
            return MEMORY_SPACES[0]  # Fallback to DRAM
```

## Integration Points

### Phase 2 → Phase 3
```python
# Phase 2 output
sim_graph = stablehlo_parser.parse_stablehlo(mlir_text)

# Phase 3 input
scheduler = StaticScheduler()
scheduler_nodes = scheduler.schedule(sim_graph)
```

### Phase 3 → Phase 1 (Backend)
```python
# Phase 3 output
scheduler_nodes = [MockSchedulerNode(...), ...]

# Phase 1 input (Backend)
kernel_group = MockKernelGroup()
kernel = SimpleMLIRKernel(kernel_group)

for node in scheduler_nodes:
    node.run(vars, reduction_vars)

mlir_output = kernel.render_mlir_kernel()
```

## Testing Strategy

### Phase 1 Tests (Existing)
- ✅ Kernel initialization
- ✅ Add operation generation
- ✅ MLIR format validation

### Phase 2 Tests (TODO)
```python
def test_parse_simple_add():
    mlir_text = "func.func @f(...) { %0 = stablehlo.add ... }"
    graph = parser.parse_stablehlo(mlir_text)
    assert len(graph) == 1
    assert graph[0].op_type == "stablehlo.add"

def test_parse_complex_graph():
    # Test mul → add chain
    pass

def test_unsupported_ops():
    # Test error handling for if/while
    pass
```

### Phase 3 Tests (TODO)
```python
def test_simple_tiling():
    # 128x128 tensor → 64x64 tiles
    pass

def test_memory_assignment():
    # Inputs/outputs → DRAM, intermediates → SRAM
    pass

def test_node_execution():
    # MockSchedulerNode runs correctly
    pass
```

### Integration Tests (TODO)
```python
def test_stablehlo_to_mlir_e2e():
    # Full pipeline: StableHLO → Parser → Scheduler → Backend → MLIR
    stablehlo_code = """
    func.func @add_kernel(...) {
        %0 = stablehlo.add %a, %b : tensor<128x128xf32>
        return %0
    }
    """
    mlir_output = compile_stablehlo(stablehlo_code)
    assert "affine.for" in mlir_output
    assert "affine.vector_load" in mlir_output
```

## Debugging Tips

### Phase 1
- Print CSE variables: `print(kernel.cse.cache)`
- Print buffers: `print(kernel.loads.getvalue())`
- Inspect tile descriptor: `print(kernel.kernel_group.tile_desc.get_tile_size())`

### Phase 2
- Print parsed graph: `for node in graph: print(node)`
- Verify topological sort: `print([n.name for n in graph])`
- Check type resolution: `print(node.output_shape, node.output_dtype)`

### Phase 3
- Print tiling decisions: `print(f"Tile size: {tile_config.tile_size}")`
- Verify memory assignment: `print(f"{buf_name} → {memory_space}")`
- Check node execution: `print(f"Executed {node.get_op_name()}")`

## Common Issues & Solutions

### Issue: CSE Variable Generation Fails
**Cause**: V.kernel not set (virtualized context)
**Solution**: For PoC, bypass CSE and write directly to buffers (as done in Phase 1)

### Issue: Stride Computation Wrong
**Cause**: Assumed C-contiguous layout
**Solution**: Verify with actual tensor layout; adjust stride computation if needed

### Issue: MLIR Validation Fails
**Cause**: Invalid type signatures or memory space usage
**Solution**: Run through stablehlo-opt for validation; check memref attributes

## References

- PyTorchSim Backend: `PyTorchSimFrontend/mlir/mlir_codegen_backend.py`
- MLIR Docs: https://mlir.llvm.org/docs/
- StableHLO Spec: https://github.com/openxla/stablehlo
- PyTorch Inductor: `torch._inductor.codegen`

## Contact & Questions

For questions about Phase 1 implementation or extension guidance:
1. Review `PHASE1_IMPLEMENTATION_COMPLETE.md`
2. Check example output in `tests/manual_backend_driver.py`
3. Trace through `mlir_codegen_backend.py` for backend details
