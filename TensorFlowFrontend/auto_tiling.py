"""
auto_tiling.py - Automatic tiling pass for TF MLIR (과제 4)

Transforms standard linalg operations (matmul, elementwise) into
DMA-based SRAM-tiled loops, following PyTorchSim's tiling strategy:
  - GEMM: 3-level nested tiling (M, N, K tiles) with SRAM scratchpads
  - Elementwise: 1-level flat tiling with DMA read/write

References:
  - PyTorchSimFrontend/mlir/mlir_gemm_template.py (GEMM_TEMPLATE)
  - PyTorchSimFrontend/mlir/mlir_codegen_backend.py (elementwise DMA)
"""

import re
import math
import logging

logger = logging.getLogger("TensorFlowFrontend.auto_tiling")
logger.setLevel(logging.WARNING)

# Default NPU hardware parameters (matching PyTorchSim defaults)
DEFAULT_SPAD_SIZE = 131072   # 128KB SRAM scratchpad
DEFAULT_VECTOR_LANE = 16     # Systolic array vector lane
DEFAULT_SPAD_VADDR = 0xD0000000
DEFAULT_SPAD_PADDR = 0x2000000000


def _compute_gemm_tile_sizes(M, N, K, spad_size, dtype_bytes=4):
    """Compute tile sizes for GEMM following PyTorchSim's heuristic.
    
    PyTorchSim allocates 3 SRAM buffers: X_tile (TILE_M x TILE_K),
    W_tile (TILE_K x TILE_N), Y_tile (TILE_M x TILE_N).
    
    Constraint: (TILE_M*TILE_K + TILE_K*TILE_N + TILE_M*TILE_N) * dtype_bytes <= spad_size
    
    Strategy: Start with equal tile sizes and shrink to fit SRAM.
    """
    max_elements = spad_size // dtype_bytes
    
    # Try square tiles first
    tile = int(math.sqrt(max_elements / 3))
    tile = max(1, min(tile, M, N, K))
    
    # Round down to power of 2 for alignment
    tile = 2 ** int(math.log2(tile)) if tile > 1 else 1
    
    TILE_M = min(tile, M)
    TILE_N = min(tile, N)
    TILE_K = min(tile, K)
    
    # Verify fits in SRAM
    while (TILE_M * TILE_K + TILE_K * TILE_N + TILE_M * TILE_N) * dtype_bytes > spad_size:
        if TILE_M > 1: TILE_M //= 2
        elif TILE_N > 1: TILE_N //= 2
        elif TILE_K > 1: TILE_K //= 2
        else: break
    
    return TILE_M, TILE_N, TILE_K


def _compute_elementwise_tile_size(total_elements, spad_size, n_buffers=3, dtype_bytes=4):
    """Compute tile size for elementwise ops (add, mul, relu).
    
    PyTorchSim uses 2-3 SRAM buffers for elementwise:
    - Binary ops (add, mul): 3 buffers (in0, in1, out)
    - Unary ops (relu): 2 buffers (in, out)
    """
    max_elements_per_buffer = spad_size // (n_buffers * dtype_bytes)
    tile = min(max_elements_per_buffer, total_elements)
    # Round down to multiple of 8 for vector alignment
    tile = max(8, (tile // 8) * 8)
    return min(tile, total_elements)


def generate_tiled_gemm_mlir(op_idx, in0_name, in1_name, out_name,
                              M, N, K, dtype_str="f32",
                              spad_size=DEFAULT_SPAD_SIZE):
    """Generate DMA-tiled GEMM MLIR following PyTorchSim's GEMM_TEMPLATE pattern.
    
    Structure (from mlir_gemm_template.py):
      affine.for %i = 0 to M step TILE_M:
        affine.for %j = 0 to N step TILE_N:
          // Zero-initialize Y_buffer
          affine.for %k = 0 to K step TILE_K:
            DMA_IN X_tile[TILE_M x TILE_K]
            DMA_IN W_tile[TILE_K x TILE_N]
            linalg.matmul ins(X_buf, W_buf) outs(Y_buf)
          DMA_OUT Y_tile[TILE_M x TILE_N]
    """
    TILE_M, TILE_N, TILE_K = _compute_gemm_tile_sizes(M, N, K, spad_size)
    vec_size = min(8, TILE_M * TILE_N)
    
    x_buf_size = TILE_M * TILE_K
    w_buf_size = TILE_K * TILE_N
    y_buf_size = TILE_M * TILE_N
    
    global_vars = f"""memref.global @gemm_{op_idx}_X_buffer : memref<{TILE_M}x{TILE_K}x{dtype_str}, 1>
memref.global @gemm_{op_idx}_W_buffer : memref<{TILE_K}x{TILE_N}x{dtype_str}, 1>
memref.global @gemm_{op_idx}_Y_buffer : memref<{TILE_M}x{TILE_N}x{dtype_str}, 1>"""

    body = f"""
    // Tiled GEMM: M={M}, N={N}, K={K}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}
    %c0_{op_idx} = arith.constant 0 : index
    %c1_{op_idx} = arith.constant 1 : index
    %c_tm_{op_idx} = arith.constant {TILE_M} : index
    %c_tn_{op_idx} = arith.constant {TILE_N} : index
    %c_tk_{op_idx} = arith.constant {TILE_K} : index
    %v0_{op_idx} = arith.constant dense<0.0> : vector<{y_buf_size}x{dtype_str}>
    
    %alloc_x_{op_idx} = memref.alloc() : memref<1xi32>
    %alloc_w_{op_idx} = memref.alloc() : memref<1xi32>
    %alloc_y_{op_idx} = memref.alloc() : memref<1xi32>
    
    %X_buf_{op_idx} = memref.get_global @gemm_{op_idx}_X_buffer : memref<{TILE_M}x{TILE_K}x{dtype_str}, 1>
    %W_buf_{op_idx} = memref.get_global @gemm_{op_idx}_W_buffer : memref<{TILE_K}x{TILE_N}x{dtype_str}, 1>
    %Y_buf_{op_idx} = memref.get_global @gemm_{op_idx}_Y_buffer : memref<{TILE_M}x{TILE_N}x{dtype_str}, 1>
    
    %X_flat_{op_idx} = memref.reinterpret_cast %X_buf_{op_idx} to offset: [0], sizes: [{x_buf_size}], strides: [1] : memref<{TILE_M}x{TILE_K}x{dtype_str}, 1> to memref<{x_buf_size}x{dtype_str}, 1>
    %W_flat_{op_idx} = memref.reinterpret_cast %W_buf_{op_idx} to offset: [0], sizes: [{w_buf_size}], strides: [1] : memref<{TILE_K}x{TILE_N}x{dtype_str}, 1> to memref<{w_buf_size}x{dtype_str}, 1>
    %Y_flat_{op_idx} = memref.reinterpret_cast %Y_buf_{op_idx} to offset: [0], sizes: [{y_buf_size}], strides: [1] : memref<{TILE_M}x{TILE_N}x{dtype_str}, 1> to memref<{y_buf_size}x{dtype_str}, 1>
    
    affine.for %i_{op_idx} = 0 to {M} step {TILE_M} {{
      affine.for %j_{op_idx} = 0 to {N} step {TILE_N} {{
        // Zero-initialize accumulator
        affine.vector_store %v0_{op_idx}, %Y_flat_{op_idx}[0] : memref<{y_buf_size}x{dtype_str}, 1>, vector<{y_buf_size}x{dtype_str}>
        affine.for %k_{op_idx} = 0 to {K} step {TILE_K} {{
          // DMA: DRAM -> SRAM (X tile)
          scf.for %dm_{op_idx} = %c0_{op_idx} to %c_tm_{op_idx} step %c1_{op_idx} {{
            %src_row_{op_idx} = arith.addi %i_{op_idx}, %dm_{op_idx} : index
            %src_offset_{op_idx} = arith.muli %src_row_{op_idx}, %c_tk_{op_idx} : index
            scf.for %dk_{op_idx} = %c0_{op_idx} to %c_tk_{op_idx} step %c1_{op_idx} {{
              %src_col_{op_idx} = arith.addi %k_{op_idx}, %dk_{op_idx} : index
              %val_x_{op_idx} = memref.load {in0_name}[%src_row_{op_idx}, %src_col_{op_idx}] : memref<{M}x{K}x{dtype_str}>
              memref.store %val_x_{op_idx}, %X_buf_{op_idx}[%dm_{op_idx}, %dk_{op_idx}] : memref<{TILE_M}x{TILE_K}x{dtype_str}, 1>
            }}
          }}
          // DMA: DRAM -> SRAM (W tile)
          scf.for %dk2_{op_idx} = %c0_{op_idx} to %c_tk_{op_idx} step %c1_{op_idx} {{
            %src_row_w_{op_idx} = arith.addi %k_{op_idx}, %dk2_{op_idx} : index
            scf.for %dn_{op_idx} = %c0_{op_idx} to %c_tn_{op_idx} step %c1_{op_idx} {{
              %src_col_w_{op_idx} = arith.addi %j_{op_idx}, %dn_{op_idx} : index
              %val_w_{op_idx} = memref.load {in1_name}[%src_row_w_{op_idx}, %src_col_w_{op_idx}] : memref<{K}x{N}x{dtype_str}>
              memref.store %val_w_{op_idx}, %W_buf_{op_idx}[%dk2_{op_idx}, %dn_{op_idx}] : memref<{TILE_K}x{TILE_N}x{dtype_str}, 1>
            }}
          }}
          // SRAM matmul
          linalg.matmul ins(%X_buf_{op_idx}, %W_buf_{op_idx} : memref<{TILE_M}x{TILE_K}x{dtype_str}, 1>, memref<{TILE_K}x{TILE_N}x{dtype_str}, 1>)
                  outs(%Y_buf_{op_idx} : memref<{TILE_M}x{TILE_N}x{dtype_str}, 1>)
        }} {{ accumulation_loop=true, subtile_loop="k" }}
        // DMA: SRAM -> DRAM (Y tile)
        scf.for %dm2_{op_idx} = %c0_{op_idx} to %c_tm_{op_idx} step %c1_{op_idx} {{
          %dst_row_{op_idx} = arith.addi %i_{op_idx}, %dm2_{op_idx} : index
          scf.for %dn2_{op_idx} = %c0_{op_idx} to %c_tn_{op_idx} step %c1_{op_idx} {{
            %dst_col_{op_idx} = arith.addi %j_{op_idx}, %dn2_{op_idx} : index
            %val_y_{op_idx} = memref.load %Y_buf_{op_idx}[%dm2_{op_idx}, %dn2_{op_idx}] : memref<{TILE_M}x{TILE_N}x{dtype_str}, 1>
            memref.store %val_y_{op_idx}, {out_name}[%dst_row_{op_idx}, %dst_col_{op_idx}] : memref<{M}x{N}x{dtype_str}>
          }}
        }}
      }} {{ outer_loop=true, subtile_loop="n" }}
    }} {{ outer_loop=true, subtile_loop="m" }}
    """
    return global_vars, body


def tile_linalg_matmul(mlir_content, spad_size=DEFAULT_SPAD_SIZE):
    """Replace linalg.matmul with tiled GEMM loop structure.
    
    Parses linalg.matmul ins(%A, %B : type, type) outs(%C : type) 
    and replaces with generate_tiled_gemm_mlir output.
    """
    pattern = re.compile(
        r"linalg\.matmul\s+ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*"
        r"memref<(\d+)x(\d+)x([a-z0-9]+)>\s*,\s*memref<(\d+)x(\d+)x([a-z0-9]+)>\)\s+"
        r"outs\((%[a-zA-Z0-9_]+)\s*:\s*memref<(\d+)x(\d+)x([a-z0-9]+)>\)"
    )
    
    op_idx = 0
    all_global_vars = []
    
    def replacer(match):
        nonlocal op_idx
        in0 = match.group(1)
        in1 = match.group(2)
        M = int(match.group(3))
        K = int(match.group(4))
        dtype = match.group(5)
        K2 = int(match.group(6))
        N = int(match.group(7))
        out = match.group(9)
        
        g_vars, body = generate_tiled_gemm_mlir(op_idx, in0, in1, out, M, N, K, dtype, spad_size)
        all_global_vars.append(g_vars)
        op_idx += 1
        return body
    
    result = pattern.sub(replacer, mlir_content)
    
    if all_global_vars:
        # Insert globals after module {
        global_section = "\n".join(all_global_vars)
        result = result.replace("module {", f"module {{\n{global_section}", 1)
    
    return result


def tile_linalg_generic_elementwise(mlir_content, spad_size=DEFAULT_SPAD_SIZE):
    """Tile elementwise linalg operations (add, mul) with DMA buffers.
    This is a simpler tiling pass for 1D flat elementwise operations.
    """
    # For now, elementwise ops are already handled by generate_elementwise_npu in tf_npu_codegen.py
    # This function provides a fallback for operations not matched by the template system
    return mlir_content
