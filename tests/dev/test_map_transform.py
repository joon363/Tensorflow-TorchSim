from mlir import ir
import mlir.dialects.linalg as linalg
import mlir.dialects.memref as memref
import sys

def walk_ops(op, callback):
    callback(op)
    for region in op.regions:
        for block in region.blocks:
            for inner_op in list(block.operations):
                walk_ops(inner_op, callback)

def test_map():
    mlir_text = """
    module {
      func.func @test_map(%arg0: memref<10xf32>) {
        %cst = arith.constant 0.000000e+00 : f32
        linalg.map outs(%arg0 : memref<10xf32>) (%init : f32) {
          linalg.yield %cst : f32
        }
        return
      }
    }
    """
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(mlir_text)
        
        to_erase = []
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
                    
        walk_ops(module.operation, callback)
                    
        for op in to_erase:
            op.operation.erase()
            
        print("--- Transformed MLIR ---")
        print(module)

if __name__ == "__main__":
    test_map()
