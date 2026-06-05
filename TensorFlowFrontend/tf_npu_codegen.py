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
mock_graph.scheduler = None
mock_graph.get_current_device_or_throw.return_value = torch.device("npu:0")
mock_graph.get_dtype = lambda name: torch.float32

logger = logging.getLogger("TensorFlowFrontend.tf_npu_codegen")
logger.setLevel(logging.WARNING)

def inline_helper_body(helper_code, actual_args):
    lines = helper_code.splitlines()
    sig_idx = -1
    for idx, line in enumerate(lines):
        if "func.func" in line:
            sig_idx = idx
            break
            
    if sig_idx == -1:
        return ""
        
    sig_line = lines[sig_idx]
    formal_params = re.findall(r'(%[a-zA-Z0-9_]+)\s*:', sig_line)
    
    body_lines = []
    brace_depth = 0
    in_body = False
    
    for line in lines[sig_idx:]:
        if not in_body:
            if '{' in line:
                in_body = True
                brace_depth = line.count('{') - line.count('}')
                start_body_part = line[line.find('{')+1:].strip()
                if start_body_part:
                    body_lines.append("    " + start_body_part)
        else:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                end_part = line[:line.rfind('}')].strip()
                if end_part and not end_part.startswith("return"):
                    body_lines.append("    " + end_part)
                break
            else:
                if line.strip() == "return":
                    continue
                body_lines.append(line)
                
    inlined_code_lines = []
    for line in body_lines:
        new_line = line
        for formal, actual in zip(formal_params, actual_args):
            pattern = re.compile(rf'{re.escape(formal)}(?![a-zA-Z0-9_])')
            new_line = pattern.sub(actual, new_line)
        inlined_code_lines.append(new_line)
        
    return "\n".join(inlined_code_lines)

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
    
    with V.set_graph_handler(mock_graph):
        dtype = torch.float32 if dtype_str == "f32" else torch.float64
        X = MockNode("X", X_shape, dtype)
        W = MockNode("W", W_shape, dtype)
        Y = MockNode("Y", Y_shape, dtype)
        inputs = [X, W]
        
        # Configure mock_graph properties needed by mlir_argdefs
        mock_graph.buffers = [Y]
        mock_graph.graph_inputs = { "X": X, "W": W }
        mock_graph.constants = {}
        
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
        from PyTorchSimFrontend import extension_config
        kernel.vector_lane = extension_config.vpu_num_lanes
        kernel.num_cores = 1
        
        code = mlir_template.render(
            kernel=kernel,
            template_buffer_node=Y,
            epilogue_nodes=None,
            prologue_nodes=None,
            tile_info=None
        )
        
        # Resolve hooks in template in priority order
        with kernel:
            sorted_hooks = kernel._sort_hooks_by_priority()
            for hook_key, hook_fn in sorted_hooks.items():
                rendered_val = hook_fn()
                if hook_key == "<GLOBAL_VARS>":
                    # Strip global variables from the helper function body and return separately
                    global_vars = rendered_val
                    code = code.replace(hook_key, "")
                else:
                    code = code.replace(hook_key, rendered_val)
                
        # Rename function name
        code = code.replace("func.func @kernel", f"func.func @{name}")
        return global_vars, code

def generate_conv_kernel(name, X_shape, W_shape, Y_shape, stride, padding, dilation, groups, dtype_str="f32"):
    from PyTorchSimFrontend.mlir.mlir_conv_template import MLIRConvTemplate
    from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
    
    with V.set_graph_handler(mock_graph):
        dtype = torch.float32 if dtype_str == "f32" else torch.float64
        X = MockNode("X", X_shape, dtype)
        W = MockNode("W", W_shape, dtype)
        Y = MockNode("Y", Y_shape, dtype)
        inputs = [X, W, None] # No bias
        
        # Configure mock_graph properties needed by mlir_argdefs
        mock_graph.buffers = [Y]
        mock_graph.graph_inputs = { "X": X, "W": W }
        mock_graph.constants = {}
        
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
        from PyTorchSimFrontend import extension_config
        kernel.vector_lane = extension_config.vpu_num_lanes
        kernel.num_cores = 1
        
        code = mlir_template.render(
            kernel=kernel,
            template_buffer_node=Y,
            epilogue_nodes=None,
            tile_info=None,
            **kwargs
        )
        
        with kernel:
            sorted_hooks = kernel._sort_hooks_by_priority()
            for hook_key, hook_fn in sorted_hooks.items():
                rendered_val = hook_fn()
                if hook_key == "<GLOBAL_VARS>":
                    global_vars = rendered_val
                    code = code.replace(hook_key, "")
                else:
                    code = code.replace(hook_key, rendered_val)
                
        # Rename function name
        code = code.replace("func.func @kernel", f"func.func @{name}")
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

def split_types(args_str):
    args = []
    current = []
    depth_angle = 0
    depth_bracket = 0
    for char in args_str:
        if char == '<':
            depth_angle += 1
        elif char == '>':
            depth_angle -= 1
        elif char == '[':
            depth_bracket += 1
        elif char == ']':
            depth_bracket -= 1
        
        if char == ',' and depth_angle == 0 and depth_bracket == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return args

def parse_shape(memref_str):
    # Parses e.g. "memref<32x32xf32, strided<[?, ?], offset: ?>>" into ([32, 32], "f32")
    # Clean up anything starting from the first comma
    clean_str = re.sub(r",.*", "", memref_str).rstrip('>')
    m = re.search(r"memref<([^>]+)", clean_str)
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
    
    # Map casted variables to their source flat arguments
    cast_to_source = {}
    for line in lines:
        cast_m = re.search(r"(%[a-zA-Z0-9_]+)\s*=\s*memref\.reinterpret_cast\s+(%[a-zA-Z0-9_]+)\s+", line)
        if cast_m:
            cast_name = cast_m.group(1)
            source_name = cast_m.group(2)
            cast_to_source[cast_name] = source_name

    # Parse variable types (including signature arguments)
    var_types = {}
    for line in lines:
        if "func.func @kernel" in line or "func.func @main" in line:
            m = re.search(r"func\.func\s+@\w+\((.*?)\)", line)
            if m:
                for decl in split_types(m.group(1)):
                    if ":" in decl:
                        name, type_str = decl.split(":", 1)
                        var_types[name.strip()] = type_str.strip()

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
            
    def get_or_create_flat_arg(var_name, shape, dtype_str):
        nonlocal op_idx
        src_name = cast_to_source.get(var_name, var_name)
        numel = 1
        for dim in shape:
            numel *= dim
            
        var_type = var_types.get(src_name)
        if not var_type:
            return src_name, f"memref<{numel}x{dtype_str}>"
            
        if var_type == f"memref<{numel}x{dtype_str}>":
            return src_name, var_type
            
        flat_name = f"%flat_{src_name.lstrip('%')}_{op_idx}"
        cast_line = f"    {flat_name} = memref.reinterpret_cast {src_name} to offset: [0], sizes: [{numel}], strides: [1] : {var_type} to memref<{numel}x{dtype_str}>"
        body_lines.append(cast_line)
        # Register the new flat variable's type in var_types
        var_types[flat_name] = f"memref<{numel}x{dtype_str}>"
        return flat_name, f"memref<{numel}x{dtype_str}>"

    # Process lines sequentially
    generic_block_idx = 0
    brace_depth = 0
    for line in lines:
        # Parse any local variable definition to build var_types
        alloc_m = re.search(r"(%[a-zA-Z0-9_]+)\s*=\s*memref\.alloc\(\).*?:\s*(memref<[^>]+>)", line)
        if alloc_m:
            var_types[alloc_m.group(1)] = alloc_m.group(2)
            
        cast_m = re.search(r"(%[a-zA-Z0-9_]+)\s*=\s*memref\.reinterpret_cast\s+.*?\s+to\s+(memref<[^>]+>)", line)
        if cast_m:
            var_types[cast_m.group(1)] = cast_m.group(2)

        if "func.func" in line:
            in_function_body = True
            brace_depth = 1
            body_lines.append(line)
            continue
        if not in_function_body:
            # Skip module attributes or helper declarations
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
                    flat_in, _ = get_or_create_flat_arg(in_name, shape, dtype_str)
                    flat_out, _ = get_or_create_flat_arg(out_name, shape, dtype_str)
                    g_decls, g_body = generate_relu_npu(op_idx, flat_in, flat_out, shape, dtype_str)
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
                            flat_in0, _ = get_or_create_flat_arg(in0_name, shape, dtype_str)
                            flat_in1, _ = get_or_create_flat_arg(in1_name, shape, dtype_str)
                            flat_out, _ = get_or_create_flat_arg(out_name, shape, dtype_str)
                            op_type = "add" if "arith.addf" in block_text else "mul"
                            g_decls, g_body = generate_elementwise_npu(op_idx, op_type, flat_in0, flat_in1, flat_out, shape, dtype_str)
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
            flat_in0, _ = get_or_create_flat_arg(in0_name, shape, dtype_str)
            flat_in1, _ = get_or_create_flat_arg(in1_name, shape, dtype_str)
            flat_out, _ = get_or_create_flat_arg(out_name, shape, dtype_str)
            g_decls, g_body = generate_elementwise_npu(op_idx, "add", flat_in0, flat_in1, flat_out, shape, dtype_str)
            global_decls.append(g_decls)
            body_lines.append(g_body)
            op_idx += 1
            continue
            
        # Match matmul
        matmul_m = re.search(r"linalg\.matmul\s*(?:\{.*?\}\s*)?ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)", line)
        if matmul_m:
            in0_name = matmul_m.group(1)
            in1_name = matmul_m.group(2)
            out_name = matmul_m.group(4)
            ins_types = split_types(matmul_m.group(3))
            X_shape, _ = parse_shape(ins_types[0])
            W_shape, _ = parse_shape(ins_types[1])
            Y_shape, dtype_str = parse_shape(matmul_m.group(5))
            
            flat_in0, in0_flat_type = get_or_create_flat_arg(in0_name, X_shape, dtype_str)
            flat_in1, in1_flat_type = get_or_create_flat_arg(in1_name, W_shape, dtype_str)
            flat_out, out_flat_type = get_or_create_flat_arg(out_name, Y_shape, dtype_str)
            
            kernel_name = f"gemm_kernel_{op_idx}"
            g_decls, helper_code = generate_gemm_kernel(kernel_name, X_shape, W_shape, Y_shape, dtype_str)
            global_decls.append(g_decls)
            inlined_code = inline_helper_body(helper_code, [flat_in0, flat_in1, flat_out])
            body_lines.append(inlined_code)
            op_idx += 1
            continue
            
        # Match conv2d
        conv_m = re.search(r"linalg\.conv_2d_nhwc_fhwc\s*(?:\{.*?\}\s*)?ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)", line)
        if conv_m:
            in0_name = conv_m.group(1)
            in1_name = conv_m.group(2)
            out_name = conv_m.group(4)
            ins_types = split_types(conv_m.group(3))
            X_shape, _ = parse_shape(ins_types[0])
            W_shape, _ = parse_shape(ins_types[1])
            Y_shape, dtype_str = parse_shape(conv_m.group(5))
            
            flat_in0, in0_flat_type = get_or_create_flat_arg(in0_name, X_shape, dtype_str)
            flat_in1, in1_flat_type = get_or_create_flat_arg(in1_name, W_shape, dtype_str)
            flat_out, out_flat_type = get_or_create_flat_arg(out_name, Y_shape, dtype_str)
            
            # Strides and Dilations
            stride = [1, 1]
            strides_match = re.search(r"strides\s*=\s*dense<([^>]+)>", line)
            if strides_match:
                stride = [int(x.strip()) for x in strides_match.group(1).split(",")]
            dilation = [1, 1]
            dilations_match = re.search(r"dilations\s*=\s*dense<([^>]+)>", line)
            if dilations_match:
                dilation = [int(x.strip()) for x in dilations_match.group(1).split(",")]
                
            kernel_name = f"conv_kernel_{op_idx}"
            # Padding is default (0,0) or SAME compiled in. Let's pass 0 padding and let MLIRConvTemplate compute tiling
            padding = [0, 0] # or we can determine from shape
            g_decls, helper_code = generate_conv_kernel(kernel_name, X_shape, W_shape, Y_shape, stride, padding, dilation, groups=1, dtype_str=dtype_str)
            global_decls.append(g_decls)
            inlined_code = inline_helper_body(helper_code, [flat_in0, flat_in1, flat_out])
            body_lines.append(inlined_code)
            op_idx += 1
            continue
            
        # For standard non-linalg lines, keep as-is
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            in_function_body = False
        body_lines.append(line)
        
    # Reassemble module
    # Reassemble module
    module_header = "module {"
    
    # Separate alias definitions (e.g., #t_map0) from other global declarations (e.g., memref.global)
    # as MLIR requires symbol aliases to be defined BEFORE the module block.
    aliases = []
    other_globals = []
    
    # Extract original globals and aliases from the original MLIR
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("memref.global ") or line_stripped.startswith("func.func private "):
            other_globals.append(line)
        elif line_stripped.startswith("#") or line_stripped.startswith("!"):
            aliases.append(line)
            
    for decl in global_decls:
        for line in decl.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith("#") or line_stripped.startswith("!"):
                if line not in aliases:
                    aliases.append(line)
            else:
                if line not in other_globals:
                    other_globals.append(line)
                
    alias_section = "\n".join(aliases)
    global_section = "\n".join(other_globals)
    helpers_section = "\n".join(helper_funcs)
    main_func_section = "\n".join(body_lines)
    module_footer = "}"
    
    final_mlir = f"{alias_section}\n{module_header}\n{global_section}\n{helpers_section}\n{main_func_section}\n{module_footer}"
    return final_mlir
