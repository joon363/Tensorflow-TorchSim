from mlir import ir

with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True

    module_text = """
    func.func @f() {
      return
    }
    """

    module = ir.Module.parse(module_text)

    print("Module class:", type(module))

    ops = list(module.walk())

    print("Num ops:", len(ops))

    for i, op in enumerate(ops):
        print("--- op", i, "type:", type(op))

        # show repr and dir of op (short)
        print("repr:", repr(op)[:200])

        d = [a for a in dir(op) if not a.startswith("_")]
        print("methods sample:", d[:40])

        break