from mlir import ir
import mlir.dialects.linalg as linalg
import mlir.dialects.memref as memref
import mlir.dialects.arith as arith
import mlir.dialects.scf as scf
import sys

def walk_ops(op, callback):
    callback(op)
    for region in op.regions:
        for block in region.blocks:
            for inner_op in list(block.operations):
                walk_ops(inner_op, callback)

def test_copy():
    mlir_text = """
    module {
      func.func @test_copy(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) {
        memref.copy %arg0, %arg1 : memref<8x16xf32> to memref<8x16xf32>
        return
      }
    }
    """
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(mlir_text)
        
        to_erase = []
        def callback(op):
            if op.name == "memref.copy":
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
                        
                    # Inner load and store
                    val = memref.LoadOp(src, loop_indices, ip=current_ip)
                    memref.StoreOp(val.result, dst, loop_indices, ip=current_ip)
                    
                to_erase.append(op)
                
        walk_ops(module.operation, callback)
        
        for op in to_erase:
            op.operation.erase()
            
        print("--- Transformed MLIR ---")
        print(module)

if __name__ == "__main__":
    test_copy()
