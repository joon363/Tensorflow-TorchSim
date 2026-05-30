class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 1024]", arg1_1: "f32[1024, 1024]"):
        # File: /opt/conda/lib/python3.10/site-packages/torch/_dynamo/external_utils.py:17, code: return fn(*args, **kwargs)
        add: "f32[1024, 1024]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (add,)
        