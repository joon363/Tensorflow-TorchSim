import re

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

line = '    linalg.matmul {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} ins(%arg0, %arg1 : memref<32x32xf32, strided<[?, ?], offset: ?>>, memref<32x32xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<32x32xf32>)'

pattern = r"linalg\.matmul\s*(?:\{.*?\}\s*)?ins\((%[a-zA-Z0-9_]+),\s*(%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)\s+outs\((%[a-zA-Z0-9_]+)\s*:\s*(.*?)\)"

m = re.search(pattern, line)
if m:
    print("MATCH SUCCESS!")
    print("Group 1 (in0):", m.group(1))
    print("Group 2 (in1):", m.group(2))
    types = split_types(m.group(3))
    print("Type 0:", types[0])
    print("Type 1:", types[1])
    print("Group 4 (out):", m.group(4))
    print("Group 5 (out_type):", m.group(5))
else:
    print("NO MATCH")
