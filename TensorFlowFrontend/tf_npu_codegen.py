import os
import re
import shlex
import subprocess
import logging
import sympy
import torch
from unittest.mock import MagicMock
from torch._inductor.virtualized import V

# Configure mock graph for PyTorchSim template rendering
mock_graph = MagicMock()
mock_graph.get_current_device_or_throw.return_value = torch.device("npu:0")
mock_graph.get_dtype = lambda name: torch.float32
V.graph = mock_graph

logger = logging.getLogger("TensorFlowFrontend.tf_npu_codegen")
logger.setLevel(logging.WARNING)

class MockLayout:
    def __init__(self, size, stride, dtype=torch.float32, device="npu:0"):
        self.size = size
        self.stride = stride
        self.dtype = dtype
        self.device = torch.device(device)
        self.offset = 0

class MockNode:
    def __init__(self, name, shape, dtype=torch.float32):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        
        # Calculate contiguous strides
        strides = []
        current = 1
        for s in reversed(shape):
            strides.append(current)
            current *= s
        strides.reverse()
        self.stride = strides
        self.layout = MockLayout(shape, strides, dtype)
        
    def get_name(self):
        return self.name
        
    def get_size(self):
        return self.shape
        
    def get_stride(self):
        return self.stride
        
    def get_dtype(self):
        return self.dtype
        
    def get_device(self):
        return torch.device("npu:0")
        
    def get_layout(self):
        return self.layout
        
    def get_numel(self):
        numel = 1
        for s in self.shape:
            numel *= s
        return numel

def generate_gemm_kernel(name, X_shape, W_shape, Y_shape, dtype_str="f32"):
    from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
    from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
    
    dtype = torch.float32 if dtype_str == "f32" else torch.float64
    X = MockNode("X", X_shape, dtype)
    W = MockNode("W", W_shape, dtype)
    Y = MockNode("Y", Y_shape, dtype)
    inputs = [X, W]
    
    mlir_template = MLIRGemmTemplate(inputs, Y.get_layout())
    
    kernel = MLIRTemplateKernel(
        kernel_name=name,
        input_nodes=inputs,
        call_size=Y.get_layout().size,
        kernel_arg_attributes=None
    )
    kernel.spad_info = {
        'spad_vaddr': 3489660928,
        'spad_paddr': 137438953472,
        'spad_size': 131072
    }
    kernel.vector_lane = 16
    kernel.num_cores = 1
    
    code = mlir_template.render(
        kernel=kernel,
        template_buffer_node=Y,
        epilogue_nodes=None,
        prologue_nodes=None,
        tile_info=None
    )
    
    # Resolve hooks in template
    for hook_key in list(kernel.render_hooks.keys()):
        priority, hook_fn = kernel.render_hooks[hook_key]
        rendered_val = hook_fn()
        if hook_key == "<GLOBAL_VARS>":
            # Strip global variables from the helper function body and return separately
            global_vars = rendered_val
            code = code.replace(hook_key, "")
        else:
            code = code.replace(hook_key, rendered_val)
            
    return global_vars, code

def generate_conv_kernel(name, X_shape, W_shape, Y_shape, stride, padding, dilation, groups, dtype_str="f32"):
    from PyTorchSimFrontend.mlir.mlir_conv_template import MLIRConvTemplate
    from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
    
    dtype = torch.float32 if dtype_str == "f32" else torch.float64
    X = MockNode("X", X_shape, dtype)
    W = MockNode("W", W_shape, dtype)
    Y = MockNode("Y", Y_shape, dtype)
    inputs = [X, W, None] # No bias
    
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": False,
        "output_padding": (0, 0),
        "groups": groups,
    }
    
    mlir_template = MLIRConvTemplate(inputs, Y.get_layout(), **kwargs)
    
    kernel = MLIRTemplateKernel(
        kernel_name=name,
        input_nodes=inputs,
        call_size=Y.get_layout().size,
        kernel_arg_attributes=None
    )
    kernel.spad_info = {
        'spad_vaddr': 3489660928,
        'spad_paddr': 137438953472,
        'spad_size': 131072
    }
    kernel.vector_lane = 16
    kernel.num_cores = 1
    
    code = mlir_template.render(
        kernel=kernel,
        template_buffer_node=Y,
        epilogue_nodes=None,
        tile_info=None,
        **kwargs
    )
    
    for hook_key in list(kernel.render_hooks.keys()):
        priority, hook_fn = kernel.render_hooks[hook_key]
        rendered_val = hook_fn()
        if hook_key == "<GLOBAL_VARS>":
            global_vars = rendered_val
            code = code.replace(hook_key, "")
        else:
            code = code.replace(hook_key, rendered_val)
            
    return global_vars, code

def generate_elementwise_npu(op_idx, op_type, in0_name, in1_name, out_name, shape, dtype_str="f32"):
    size = 1
    for dim in shape:
        size *= dim
        
    tile_size = size if size <= 512 else 512
    vec_size = size if size <= 8 else 8
    
    global_vars = f"""memref.global @buf_{op_idx}_spad0 : memref<{tile_size}x{dtype_str}, 1>
memref.global @buf_{op_idx}_spad1 : memref<{tile_size}x{dtype_str}, 1>
memref.global @buf_{op_idx}_spad2 : memref<{tile_size}x{dtype_str}, 1>"""

    body = f"""
    %c0_{op_idx} = arith.constant 0 : index
    %c_tile_{op_idx} = arith.constant {tile_size} : index
    %c_vec_{op_idx} = arith.constant {vec_size} : index
    
    %alloc0_{op_idx} = memref.alloc() : memref<1xi32>
    %alloc1_{op_idx} = memref.alloc() : memref<1xi32>
    %alloc2_{op_idx} = memref.alloc() : memref<1xi32>
    
    %spad0_{op_idx} = memref.get_global @buf_{op_idx}_spad0 : memref<{tile_size}x{dtype_str}, 1>
    %spad1_{op_idx} = memref.get_global @buf_{op_idx}_spad1 : memref<{tile_size}x{dtype_str}, 1>
    %spad2_{op_idx} = memref.get_global @buf_{op_idx}_spad2 : memref<{tile_size}x{dtype_str}, 1>
    
    affine.for %index0_{op_idx} = 0 to {size} step {tile_size} {{
        memref.dma_start {in0_name}[%index0_{op_idx}], %spad0_{op_idx}[%c0_{op_idx}], %c_tile_{op_idx}, %alloc0_{op_idx}[%c0_{op_idx}], %c0_{op_idx}, %c_tile_{op_idx} : memref<{size}x{dtype_str}>, memref<{tile_size}x{dtype_str}, 1>, memref<1xi32> {{dram_stride=[1], sram_stride=[1], padding=0}}
        memref.dma_start {in1_name}[%index0_{op_idx}], %spad1_{op_idx}[%c0_{op_idx}], %c_tile_{op_idx}, %alloc1_{op_idx}[%c0_{op_idx}], %c0_{op_idx}, %c_tile_{op_idx} : memref<{size}x{dtype_str}>, memref<{tile_size}x{dtype_str}, 1>, memref<1xi32> {{dram_stride=[1], sram_stride=[1], padding=0}}
        affine.for %compute_idx_{op_idx} = 0 to {tile_size} step {vec_size} {{
            %v0_{op_idx} = affine.vector_load %spad0_{op_idx}[%compute_idx_{op_idx}] : memref<{tile_size}x{dtype_str}, 1>, vector<{vec_size}x{dtype_str}>
            %v1_{op_idx} = affine.vector_load %spad1_{op_idx}[%compute_idx_{op_idx}] : memref<{tile_size}x{dtype_str}, 1>, vector<{vec_size}x{dtype_str}>
            %res_{op_idx} = arith.{op_type}f %v0_{op_idx}, %v1_{op_idx} : vector<{vec_size}x{dtype_str}>
            affine.vector_store %res_{op_idx}, %spad2_{op_idx}[%compute_idx_{op_idx}] : memref<{tile_size}x{dtype_str}, 1>, vector<{vec_size}x{dtype_str}>
        }} {{inner_loop=false}}
        memref.dma_start %spad2_{op_idx}[%c0_{op_idx}], {out_name}[%index0_{op_idx}], %c_tile_{op_idx}, %alloc2_{op_idx}[%c0_{op_idx}], %c0_{op_idx}, %c_tile_{op_idx} : memref<{tile_size}x{dtype_str}, 1>, memref<{size}x{dtype_str}>, memref<1xi32> {{dram_stride=[1], sram_stride=[1], padding=0}}
    }} {{outer_loop=true}}
    """
    return global_vars, body

def generate_relu_npu(op_idx, in_name, out_name, shape, dtype_str="f32"):
    size = 1
    for dim in shape:
        size *= dim
        
    tile_size = size if size <= 512 else 512
    vec_size = size if size <= 8 else 8
    
    global_vars = f"""memref.global @buf_{op_idx}_spad0 : memref<{tile_size}x{dtype_str}, 1>
memref.global @buf_{op_idx}_spad1 : memref<{tile_size}x{dtype_str}, 1>"""

    body = f"""
    %c0_{op_idx} = arith.constant 0 : index
    %c_tile_{op_idx} = arith.constant {tile_size} : index
    %c_vec_{op_idx} = arith.constant {vec_size} : index
    
    %alloc0_{op_idx} = memref.alloc() : memref<1xi32>
    %alloc1_{op_idx} = memref.alloc() : memref<1xi32>
    
    %spad0_{op_idx} = memref.get_global @buf_{op_idx}_spad0 : memref<{tile_size}x{dtype_str}, 1>
    %spad1_{op_idx} = memref.get_global @buf_{op_idx}_spad1 : memref<{tile_size}x{dtype_str}, 1>
    
    %zero_{op_idx} = arith.constant dense<0.0> : vector<{vec_size}x{dtype_str}>
    
    affine.for %index0_{op_idx} = 0 to {size} step {tile_size} {{
        memref.dma_start {in_name}[%index0_{op_idx}], %spad0_{op_idx}[%c0_{op_idx}], %c_tile_{op_idx}, %alloc0_{op_idx}[%c0_{op_idx}], %c0_{op_idx}, %c_tile_{op_idx} : memref<{size}x{dtype_str}>, memref<{tile_size}x{dtype_str}, 1>, memref<1xi32> {{dram_stride=[1], sram_stride=[1], padding=0}}
        affine.for %compute_idx_{op_idx} = 0 to {tile_size} step {vec_size} {{
            %v0_{op_idx} = affine.vector_load %spad0_{op_idx}[%compute_idx_{op_idx}] : memref<{tile_size}x{dtype_str}, 1>, vector<{vec_size}x{dtype_str}>
            %res_{op_idx} = arith.maxf %v0_{op_idx}, %zero_{op_idx} : vector<{vec_size}x{dtype_str}>
            affine.vector_store %res_{op_idx}, %spad1_{op_idx}[%compute_idx_{op_idx}] : memref<{tile_size}x{dtype_str}, 1>, vector<{vec_size}x{dtype_str}>
        }} {{inner_loop=false}}
        memref.dma_start %spad1_{op_idx}[%c0_{op_idx}], {out_name}[%index0_{op_idx}], %c_tile_{op_idx}, %alloc1_{op_idx}[%c0_{op_idx}], %c0_{op_idx}, %c_tile_{op_idx} : memref<{tile_size}x{dtype_str}, 1>, memref<{size}x{dtype_str}>, memref<1xi32> {{dram_stride=[1], sram_stride=[1], padding=0}}
    }} {{outer_loop=true}}
    """
    return global_vars, body

def parse_shape(memref_str):
    # Parses e.g. "memref<32x32xf32>" into ([32, 32], "f32")
    m = re.search(r"memref<([^>]+)>", memref_str)
    if not m:
        return [1], "f32"
    parts = m.group(1).split("x")
    dtype = parts[-1]
    shape = []
    for p in parts[:-1]:
        if p == "?":
            shape.append(1)
        else:
            shape.append(int(p))
    return shape, dtype

def compile_tf_to_npu_mlir(tf_mlir_content, arg_attributes):
    # Regex parser and builder
    lines = tf_mlir_content.splitlines()
    
    global_decls = []
    helper_funcs = []
    body_lines = []
    
    # We will first parse the original inputs and allocate reinterpret casts
    in_function_body = False
    op_idx = 0
    
    # Locate all generic blocks
    generic_blocks = []
    in_generic = False
    generic_lines = []
    
    for line in lines:
        if "linalg.generic" in line and "{" in line:
            in_generic = True
            generic_lines = [line]
            continue
        if in_generic:
            generic_lines.append(line)
            if line.strip() == "}":
                in_generic = False
                generic_blocks.append("\n".join(generic_lines))
            continue
            
    # Process lines sequentially
    generic_block_idx = 0
    for line in lines:
        if "func.func" in line:
            in_function_body = True
            body_lines.append(line)
            continue
        if not in_function_body:
            # Skip module attributes or helper declarations
            continue
        if line.strip() == "}":
            in_function_body = False
            body_lines.append(line)
            continue
            
        # Skip generic parts already extracted
        if "linalg.generic" in line and "{" in line:
            in_generic = True
            # Replace generic block with parsed loop
            block_text = generic_blocks[generic_block_idx]
            generic_block_idx += 1
            
            # Determine if it's relu or other elementwise
            # Match ins/outs
            ins_m = re.search(r"ins\((%[a-zA-Z0-9_]+)\s*:\s*([^)]+)\)", block_text)
            outs_m = re.search(r"outs\((%[a-zA-Z0-9_]+)\s*:\s*([^)]+)\)", block_text)
            
            if ins_m and outs_m:
                in_name = ins_m.group(1)
                out_name = outs_m.group(1)
                in_type = ins_m.group(2)
                out_type = outs_m.group(2)
                shape, dtype_str = parse_shape(in_type)
                
                # Check for relu
                if "arith.maxf" in block_text:
                    g_decls, g_body = generate_relu_npu(op_idx, in_name, out_name, shape, dtype_str)
                    global_decls.append(g_decls)
                    body_lines.append(g_body)
                    op_idx += 1
                elif "arith.addf" in block_text or "arith.mulf" in block_text:
                    # Generic binary elementwise (if multiple inputs)
                    # For simplicity, if we have 2 inputs in ins:
                    ins_list_m = re.search(r"ins\(([^)]+)\)", block_text)
                    if ins_list_m:
                        ins_parts = ins_list_m.group(1).split(",")
                        if len(ins_parts) == 2:
                            in0_name = ins_parts[0].split(":")[0].strip()
                            in1_name = ins_parts[1].split(":")[0].strip()
                            op_type = "add" if "arith.addf" in block_text else "mul"
                            g_decls, g_body = generate_elementwise_npu(op_idx, op_type, in0_name, in1_name, out_name, shape, dtype_str)
                            global_decls.append(g_decls)
                            body_lines.append(g_body)
                            op_idx += 1
            continue
            
        if in_generic:
            if line.strip() == "}":
                in_generic = False
            continue
            
        # Match elementwise add
        add_m = re.search(r"linalg\.add\s+ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*([^,)]+),\s*([^,)]+)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*([^)]+)\)", line)
        if add_m:
            in0_name = add_m.group(1)
            in1_name = add_m.group(2)
            out_name = add_m.group(5)
            type_str = add_m.group(3)
            shape, dtype_str = parse_shape(type_str)
            g_decls, g_body = generate_elementwise_npu(op_idx, "add", in0_name, in1_name, out_name, shape, dtype_str)
            global_decls.append(g_decls)
            body_lines.append(g_body)
            op_idx += 1
            continue
            
        # Match matmul
        matmul_m = re.search(r"linalg\.matmul\s+ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*([^,)]+),\s*([^,)]+)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*([^)]+)\)", line)
        if matmul_m:
            in0_name = matmul_m.group(1)
            in1_name = matmul_m.group(2)
            out_name = matmul_m.group(5)
            X_shape, _ = parse_shape(matmul_m.group(3))
            W_shape, _ = parse_shape(matmul_m.group(4))
            Y_shape, dtype_str = parse_shape(matmul_m.group(6))
            
            kernel_name = f"gemm_kernel_{op_idx}"
            g_decls, helper_code = generate_gemm_kernel(kernel_name, X_shape, W_shape, Y_shape, dtype_str)
            global_decls.append(g_decls)
            helper_funcs.append(helper_code)
            
            # Add call to matmul kernel
            body_lines.append(f"    call @{kernel_name}({in0_name}, {in1_name}, {out_name}) : (memref<{X_shape[0]}x{X_shape[1]}x{dtype_str}>, memref<{W_shape[0]}x{W_shape[1]}x{dtype_str}>, memref<{Y_shape[0]}x{Y_shape[1]}x{dtype_str}>) -> ()")
            op_idx += 1
            continue
            
        # Match conv2d
        conv_m = re.search(r"linalg\.conv_2d_nhwc_fhwc\s*(\{dilations\s*=\s*dense<([^>]+)>,\s*strides\s*=\s*dense<([^>]+)>\})?\s*ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*([^,)]+),\s*([^,)]+)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*([^)]+)\)", line)
        if conv_m:
            in0_name = conv_m.group(4)
            in1_name = conv_m.group(5)
            out_name = conv_m.group(8)
            X_shape, _ = parse_shape(conv_m.group(6))
            W_shape, _ = parse_shape(conv_m.group(7))
            Y_shape, dtype_str = parse_shape(conv_m.group(9))
            
            # Strides and Dilations
            stride = [1, 1]
            if conv_m.group(3):
                stride = [int(x.strip()) for x in conv_m.group(3).split(",")]
            dilation = [1, 1]
            if conv_m.group(2):
                dilation = [int(x.strip()) for x in conv_m.group(2).split(",")]
                
            kernel_name = f"conv_kernel_{op_idx}"
            # Padding is default (0,0) or SAME compiled in. Let's pass 0 padding and let MLIRConvTemplate compute tiling
            padding = [0, 0] # or we can determine from shape
            g_decls, helper_code = generate_conv_kernel(kernel_name, X_shape, W_shape, Y_shape, stride, padding, dilation, groups=1, dtype_str=dtype_str)
            global_decls.append(g_decls)
            helper_funcs.append(helper_code)
            
            X_type = f"memref<{'x'.join(map(str, X_shape))}x{dtype_str}>"
            W_type = f"memref<{'x'.join(map(str, W_shape))}x{dtype_str}>"
            Y_type = f"memref<{'x'.join(map(str, Y_shape))}x{dtype_str}>"
            body_lines.append(f"    call @{kernel_name}({in0_name}, {in1_name}, {out_name}) : ({X_type}, {W_type}, {Y_type}) -> ()")
            op_idx += 1
            continue
            
        # For standard non-linalg lines, keep as-is
        body_lines.append(line)
        
    # Reassemble module
    module_header = "module {"
    global_section = "\n".join(global_decls)
    helpers_section = "\n".join(helper_funcs)
    main_func_section = "\n".join(body_lines)
    module_footer = "}"
    
    final_mlir = f"{module_header}\n{global_section}\n{helpers_section}\n{main_func_section}\n{module_footer}"
    return final_mlir
