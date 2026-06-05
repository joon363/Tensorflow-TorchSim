from mlir import ir
import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.dialects.arith as arith
import mlir.dialects.memref as memref

ctx = ir.Context()
ctx.allow_unregistered_dialects = True
with ctx, ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        # Create a function
        f_type = ir.FunctionType.get([], [])
        func_op = func.FuncOp("test_func", f_type)
        entry_block = func_op.add_entry_block()
        
        with ir.InsertionPoint(entry_block):
            idx_type = ir.IndexType.get()
            c0 = arith.ConstantOp(idx_type, 0)
            c1 = arith.ConstantOp(idx_type, 1)
            c32 = arith.ConstantOp(idx_type, 32)
            
            # Create nested loop with manual yield
            for_op1 = scf.ForOp(c0.result, c32.result, c1.result)
            loop_body1 = for_op1.regions[0].blocks[0]
            
            yield_ip1 = ir.InsertionPoint.at_block_begin(loop_body1)
            scf.YieldOp([], ip=yield_ip1)
            current_ip = ir.InsertionPoint(loop_body1.operations[0])
            
            for_op2 = scf.ForOp(c0.result, c32.result, c1.result, ip=current_ip)
            loop_body2 = for_op2.regions[0].blocks[0]
            
            yield_ip2 = ir.InsertionPoint.at_block_begin(loop_body2)
            scf.YieldOp([], ip=yield_ip2)
            current_ip2 = ir.InsertionPoint(loop_body2.operations[0])
            
            arith.ConstantOp(idx_type, 42, ip=current_ip2)
            
            func.ReturnOp([])
            
    print(module)
