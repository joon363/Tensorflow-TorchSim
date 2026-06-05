from mlir import ir
import mlir.dialects.linalg as linalg
import mlir.dialects.memref as memref
import sys

def inspect():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        mlir_string = open("/workspace/PyTorchSim/Tensorflow/tests/out/tf-mlir.mlir").read()
        module = ir.Module.parse(mlir_string)
        print("--- Operations in module ---")
        for op in module.body.operations:
            print("Module Op:", op.operation.name)
            if op.operation.name == "func.func":
                for block in op.regions[0].blocks:
                    for inner_op in block.operations:
                        print("  Op:", inner_op.name)
                        print("    Operands:", [t.type for t in inner_op.operands])
                        print("    Results:", [t.type for t in inner_op.results])
                        # Print string representation of the operation
                        print("    Raw:", str(inner_op).strip())

if __name__ == "__main__":
    inspect()
