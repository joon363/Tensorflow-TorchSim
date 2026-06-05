import os
import re
import shlex
import subprocess
import logging

logger = logging.getLogger("TensorFlowFrontend.tf_mlir_conversion")
logger.setLevel(logging.WARNING)

def transform_tf_mlir(content, arg_attributes=None):
    # We will run the standard structural transformation and flattening first
    # to rename functions and flatten arguments, then pass it to compile_tf_to_npu_mlir.

    from mlir import ir
    import mlir.dialects.func as func
    import mlir.dialects.memref as memref
    import mlir.dialects.scf as scf
    import mlir.dialects.arith as arith
    import mlir.dialects.linalg as linalg

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(content)

        # 0. Structural transformations for linalg.map and memref.copy
        to_erase = []

        def walk_ops(op, callback):
            callback(op)
            for region in op.regions:
                for block in region.blocks:
                    for inner_op in list(block.operations):
                        walk_ops(inner_op, callback)

        def callback(op):
            if op.name == "linalg.map":
                block = op.regions[0].blocks[0]
                yield_op = block.operations[-1]
                if yield_op.name == "linalg.yield":
                    yielded_val = yield_op.operands[0]
                    out_val = op.operands[-1]
                    with ir.InsertionPoint(op):
                        linalg.fill(yielded_val, outs=[out_val])
                    to_erase.append(op)
            elif op.name == "memref.copy":
                src = op.operands[0]
                dst = op.operands[1]
                src_type = ir.MemRefType(src.type)
                shape = src_type.shape
                
                with ir.InsertionPoint(op):
                    idx_type = ir.IndexType.get()
                    c0 = arith.ConstantOp(idx_type, 0)
                    c1 = arith.ConstantOp(idx_type, 1)
                    
                    current_ip = ir.InsertionPoint(op)
                    loop_indices = []
                    for dim in shape:
                        dim_val = arith.ConstantOp(idx_type, dim, ip=current_ip)
                        for_op = scf.ForOp(c0.result, dim_val.result, c1.result, ip=current_ip)
                        loop_indices.append(for_op.induction_variable)
                        
                        loop_body = for_op.regions[0].blocks[0]
                        yield_ip = ir.InsertionPoint.at_block_begin(loop_body)
                        scf.YieldOp([], ip=yield_ip)
                        current_ip = ir.InsertionPoint(loop_body.operations[0])
                        
                    val = memref.LoadOp(src, loop_indices, ip=current_ip)
                    memref.StoreOp(val.result, dst, loop_indices, ip=current_ip)
                to_erase.append(op)

        walk_ops(module.operation, callback)

        for op in to_erase:
            op.operation.erase()

        
        # 1. Find entry func.func
        func_op = None
        for op in module.body.operations:
            if op.operation.name == "func.func":
                func_op = op
                break
        if not func_op:
            return content
            
        # Rename function
        func_op.attributes['sym_name'] = ir.StringAttr.get('kernel')
        
        # 2. Check returns
        ft = func_op.type
        inputs = list(ft.inputs)
        results = list(ft.results)
        
        if len(results) == 1 and isinstance(results[0], ir.MemRefType):
            ret_type = results[0]
            
            # Map TF inputs to PyTorch arg_attributes if provided
            tf_to_pt_mapping = {}
            pt_inputs = []
            pt_outputs = []
            if arg_attributes:
                from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
                for arg_name, arg_attr in arg_attributes:
                    if MLIRKernelArgs.is_mlir_arg_in(arg_attr[0]):
                        pt_inputs.append((arg_name, arg_attr))
                    elif MLIRKernelArgs.is_mlir_arg_out(arg_attr[0]):
                        pt_outputs.append((arg_name, arg_attr))
                
                matched_tf_indices = set()
                for pt_idx, (pt_name, pt_attr) in enumerate(pt_inputs):
                    pt_size = pt_attr[2]
                    found_match = False
                    for tf_idx, tf_type in enumerate(inputs):
                        if tf_idx in matched_tf_indices:
                            continue
                        if isinstance(tf_type, ir.MemRefType):
                            tf_size = 1
                            for dim in tf_type.shape:
                                tf_size *= dim
                            if tf_size == pt_size:
                                tf_to_pt_mapping[tf_idx] = pt_idx
                                matched_tf_indices.add(tf_idx)
                                found_match = True
                                break
                        else:
                            tf_to_pt_mapping[tf_idx] = pt_idx
                            matched_tf_indices.add(tf_idx)
                            found_match = True
                            break
                    if not found_match:
                        logger.warning(f"Could not match PyTorch argument {pt_name} (size {pt_size}) to any TF input.")
            else:
                for tf_idx in range(len(inputs)):
                    tf_to_pt_mapping[tf_idx] = tf_idx
                for tf_idx, tf_type in enumerate(inputs):
                    pt_inputs.append((f"arg{tf_idx}", [1, None, None]))
            
            # Calculate flat types for inputs
            new_input_types = []
            dyn = ir.ShapedType.get_dynamic_stride_or_offset()
            if arg_attributes:
                for pt_name, pt_attr in pt_inputs:
                    pt_size = pt_attr[2]
                    element_type = ir.F32Type.get()
                    flat_type = ir.MemRefType.get([pt_size], element_type)
                    new_input_types.append(flat_type)
                
                for pt_name, pt_attr in pt_outputs:
                    pt_size = pt_attr[2]
                    element_type = ir.F32Type.get()
                    flat_type = ir.MemRefType.get([pt_size], element_type)
                    new_input_types.append(flat_type)
            else:
                for t in inputs:
                    if isinstance(t, ir.MemRefType):
                        total_elements = 1
                        for dim in t.shape:
                            total_elements *= dim
                        flat_type = ir.MemRefType.get([total_elements], t.element_type)
                        new_input_types.append(flat_type)
                    else:
                        new_input_types.append(t)
                
                ret_total_elements = 1
                for dim in ret_type.shape:
                    ret_total_elements *= dim
                ret_flat_type = ir.MemRefType.get([ret_total_elements], ret_type.element_type)
                new_input_types.append(ret_flat_type)
            
            # Update function type
            new_func_type = ir.FunctionType.get(new_input_types, [])
            func_op.attributes['function_type'] = ir.TypeAttr.get(new_func_type)
            
            # Update block arguments
            entry_block = func_op.entry_block
            new_args = []
            for t in new_input_types:
                new_args.append(entry_block.add_argument(t, ir.Location.unknown()))
                
            # Insert reinterpret casts at start of block
            ip = ir.InsertionPoint.at_block_begin(entry_block)
            for tf_idx in range(len(inputs)):
                old_arg = entry_block.arguments[tf_idx]
                pt_idx = tf_to_pt_mapping.get(tf_idx, tf_idx)
                new_flat_arg = new_args[pt_idx]
                
                if isinstance(old_arg.type, ir.MemRefType):
                    arg_shape = old_arg.type.shape
                    strides = []
                    current_stride = 1
                    for dim in reversed(arg_shape):
                        strides.append(current_stride)
                        current_stride *= dim
                    strides.reverse()
                    
                    cast_op = memref.ReinterpretCastOp(
                        old_arg.type,
                        new_flat_arg,
                        [], [], [],
                        [0], arg_shape, strides,
                        ip=ip
                    )
                    old_arg.replace_all_uses_with(cast_op.result)
                else:
                    old_arg.replace_all_uses_with(new_flat_arg)
                    
            # Cast output argument
            ret_shape = ret_type.shape
            ret_strides = []
            current_stride = 1
            for dim in reversed(ret_shape):
                ret_strides.append(current_stride)
                current_stride *= dim
            ret_strides.reverse()
            
            new_flat_out_arg = new_args[len(pt_inputs)]
            cast_out_op = memref.ReinterpretCastOp(
                ret_type,
                new_flat_out_arg,
                [], [], [],
                [0], ret_shape, ret_strides,
                ip=ip
            )
            out_dest = cast_out_op.result
            
            # Erase old arguments
            for _ in range(len(inputs)):
                entry_block.erase_argument(0)
                
            # Locate ReturnOp
            return_op = None
            for op in entry_block.operations:
                if op.name == "func.return":
                    return_op = op
                    break
            
            if return_op:
                ret_val = return_op.operands[0]
                
                # The returned value could be a cast or direct alloc
                alloc_0 = ret_val
                if alloc_0.owner.name == "memref.cast":
                    alloc_0 = alloc_0.owner.operands[0]
                
                # Find the linalg.copy that writes to alloc_0
                copy_op = None
                computation_buf = None
                for op in list(entry_block.operations):
                    if op.name == "linalg.copy" and op.operands[-1] == alloc_0:
                        copy_op = op
                        computation_buf = op.operands[0]
                        break
                        
                if computation_buf:
                    # Replace all uses of computation buffer with output destination.
                    computation_buf.replace_all_uses_with(out_dest)
                    
                    # Create void return before erasing old return
                    func.ReturnOp([], ip=ir.InsertionPoint(return_op))
                    return_op.operation.erase()
                    
                    # Erase copy_op
                    copy_op.operation.erase()
                    
                    # Erase all remaining operations that use alloc_0 (e.g. memref.cast)
                    for op in reversed(list(entry_block.operations)):
                        if alloc_0 in op.operands:
                            op.operation.erase()
                            
                    # Erase alloc_0
                    if alloc_0.owner.name == "memref.alloc":
                        alloc_0.owner.operation.erase()
                        
                    # Erase computation_buf
                    if computation_buf.owner.name == "memref.alloc":
                        computation_buf.owner.operation.erase()
                
        flat_mlir = str(module)
        if os.environ.get('TENSORFLOW_NPU_CODEGEN') == "True":
            from Tensorflow.TensorFlowFrontend.tf_npu_codegen import compile_tf_to_npu_mlir
            return compile_tf_to_npu_mlir(flat_mlir, arg_attributes)
            
        return flat_mlir

# Global kernel counter for multi-kernel tracking (과제 2: Option B)
_tf_kernel_counter = 0
_tf_kernel_mlir_cache = {}  # Maps kernel_index -> transformed MLIR

def reset_tf_kernel_counter():
    """Reset kernel counter between test runs."""
    global _tf_kernel_counter, _tf_kernel_mlir_cache
    _tf_kernel_counter = 0
    _tf_kernel_mlir_cache = {}

def _split_tf_mlir_for_kernel(tf_mlir_content, arg_attributes, kernel_index):
    """과제 2 (Option B): Extract the relevant subgraph from TF MLIR
    based on the arg_attributes of the current kernel.
    
    For single-kernel models, the full MLIR is returned unchanged.
    For multi-kernel models, we match kernel args to TF args by element size
    and generate a per-kernel MLIR function.
    """
    if not arg_attributes:
        return tf_mlir_content
    
    # Count PT inputs and outputs from arg_attributes
    from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
    pt_inputs = [(name, attr) for name, attr in arg_attributes if MLIRKernelArgs.is_mlir_arg_in(attr[0])]
    pt_outputs = [(name, attr) for name, attr in arg_attributes if MLIRKernelArgs.is_mlir_arg_out(attr[0])]
    pt_n_args = len(pt_inputs)  # number of input args (not including output)
    
    # Parse the TF MLIR to count its function arguments
    func_match = re.search(r'func\.func\s+@\w+\(([^)]*)\)', tf_mlir_content)
    if not func_match:
        return tf_mlir_content
    
    # Count TF function args (each starts with %)
    args_str = func_match.group(1)
    tf_arg_count = args_str.count('%')
    
    # Extract sizes from tf-mlir args
    tf_arg_sizes = []
    for m in re.finditer(r'memref<(\d+)x', args_str):
        tf_arg_sizes.append(int(m.group(1)))
    
    # If arg counts match, this MLIR is correct for single-kernel
    # TF MLIR already has inputs + output args, PT has inputs + output
    if tf_arg_count == pt_n_args + len(pt_outputs):
        return tf_mlir_content
    
    # Multi-kernel detected: tf_arg_count > kernel arg count
    # For multi-kernel, the TF MLIR represents the FULL computation.
    # We cannot easily split it into subgraphs without deep IR analysis.
    # Instead, we return the full MLIR and let transform_tf_mlir do its best
    # to match args by element size.
    #
    # The transform_tf_mlir function already has size-matching logic:
    # it matches PT args to TF args by element count, so if the kernel
    # only uses a subset of TF args, the unmatched ones become extra params.
    #
    # KNOWN LIMITATION: This approach only works when ALL kernel inputs
    # correspond to original TF function inputs (not intermediate buffers).
    # For intermediate buffers (like buf1 between kernel_0 and kernel_1),
    # there is no matching TF arg, causing a Spike execution failure.
    
    logger.info(f"Multi-kernel detected: kernel {kernel_index} has "
                f"{pt_n_args} inputs + {len(pt_outputs)} outputs, "
                f"TF MLIR has {tf_arg_count} args total")
    
    return tf_mlir_content

def handle_tensorflow_direct_test(source_code, get_write_path_fn, arg_attributes=None):
    global _tf_kernel_counter
    if os.environ.get('TENSORFLOW_MLIR_DIRECT_TEST') == "True":
        torchsim_dir = os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim')
        tf_mlir_path = os.path.join(torchsim_dir, "Tensorflow", "tests", "out", "tf-mlir.mlir")
        if not os.path.exists(tf_mlir_path):
            tf_mlir_path = os.path.join(torchsim_dir, "TensorFlow", "tests", "out", "tf-mlir.mlir")
        if os.path.exists(tf_mlir_path):
            original_source_code = source_code
            with open(tf_mlir_path, "r") as f:
                tf_mlir_content = f.read()
            
            # 과제 2: Per-kernel MLIR matching
            source_code = _split_tf_mlir_for_kernel(tf_mlir_content, arg_attributes, _tf_kernel_counter)
            _tf_kernel_counter += 1
            
            # Copy header files from dummy write path to TF write path
            dummy_write_path = get_write_path_fn(original_source_code)
            tf_write_path = get_write_path_fn(source_code)
            os.makedirs(tf_write_path, exist_ok=True)
            import shutil
            for header in ["global_var.h", "gem5_global_var.h"]:
                src_h = os.path.join(dummy_write_path, header)
                dst_h = os.path.join(tf_write_path, header)
                if os.path.exists(src_h):
                    shutil.copy(src_h, dst_h)
    return source_code
