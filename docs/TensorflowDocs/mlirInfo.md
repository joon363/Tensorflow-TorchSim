
MLIR Python bindings — concise notes
=================================

Scope
-----
- Quick reference for using the MLIR Python bindings (`mlir.ir`) to parse
	and inspect MLIR `Module`s produced by TensorFlow/XLA (HLO / StableHLO).

Context and parsing
-------------------
- Create a `Context` and enable `allow_unregistered_dialects` when parsing
	MLIR that uses dialects not registered in the current Python environment.

Example:

```
from mlir import ir

with ir.Context() as ctx:
		ctx.allow_unregistered_dialects = True
		module = ir.Module.parse(mlir_string)
```

- `ir.Context()` scopes lifetime of dialect registrations and parsing.

Module structure (conceptual)
----------------------------
- A parsed `Module` is a top-level operation (ModuleOp) that contains one or
	more `Region`s. Each region contains `Block`s, and each block contains
	`Operation`s. This nested structure mirrors the C++ API described in the
	MLIR docs (`Operation -> Region -> Block -> Operation`).

Traversal and APIs
------------------
- `module.walk()` (or `op.walk(...)`) performs a nested traversal of all
	operations under the module. Use it to visit every `Operation` without
	manual recursion.

Example:

```
for op in module.walk():
		# printing the operation yields its textual MLIR form
		print(op)
```

- Operations expose basic properties (Python wrapper objects):
	- `op.operands` — list-like view of operand `Value`s
	- `op.results` — list-like view of result `Value`s
	- `op.attributes` — attribute dictionary (named attributes)
	- `op.regions` — iterable of nested regions under the op

- Regions and blocks:
	- `region.blocks` gives blocks contained in the region
	- each `block` has `block.arguments` and `block.operations` (or
		`block.op_iter()` depending on the binding)

Def-use and values
------------------
- A `Value` is either a `BlockArgument` or the result of an `Operation`.
- Use the def-use relationships to inspect producers/consumers. In Python you
	can inspect an operation's operands to find their defining operation.

Notes on safety and rewriting
----------------------------
- When parsing MLIR that references custom dialects (StableHLO, TensorFlow
	dialects, or project-specific dialects), set
	`ctx.allow_unregistered_dialects = True` before `Module.parse`.
- Prefer working with the Python `ir.Module` / `op` objects rather than
	ad-hoc string parsing; the Python API preserves structure and metadata and
	is resilient to formatting differences.

Useful workflow snippets
-----------------------

- Extract ops of a certain textual kind (safe fallback):

```
matches = [op for op in module.walk() if str(op).startswith("  " + "linalg.matmul") or "linalg.matmul" in str(op)]
```

- Inspect an op's attributes and operands (pseudo-example):

```
for op in module.walk():
		print('op:', op)
		print('  attrs:', dict(op.attributes))
		print('  #operands:', len(op.operands))
		for i, operand in enumerate(op.operands):
				print('    operand', i, '->', operand)
```

References
----------
- MLIR tutorial: Understanding the IR Structure — https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
- TF -> MLIR usage example: see `0531test2.py` and `test_lowering.ipynb` in
	this workspace for parsing TensorFlow-generated MLIR with `ir.Module.parse()`.

Tips for this project
---------------------
- Use `ir.Module.parse(...)` (not string regex) to ingest HLO/StableHLO MLIR.
- Use `module.walk()` to find Linalg/TF/StableHLO ops and implement lowering
	passes that operate on the parsed operation objects or rewrite the IR using
	MLIR pass tooling when available.
