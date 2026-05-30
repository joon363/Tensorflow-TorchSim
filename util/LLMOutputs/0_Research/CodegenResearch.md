**Q1 — `codegen_loops()` order & why**

- **Top-level setup (const/alloc/spad buffers):** `code.splice(self.const_buffer)` → `self.alloc_buffer` → `self.spad_buffer`
    - Purpose: declare constants, allocate memrefs and scratchpad globals before emitting loop bodies so later code can reference them.
- **Outer loop creation:** iterate `loops.loops`, emit `loop.lines()` and enter an indent with attribute `{outer_loop=true}`.
    - Purpose: represent tiling at coarse granularity (tile per outer loop). Attributes mark outer loop for autotuner/simulator.
- **Reduction prefix & reduction loops:** `code.splice(self.reduction_prefix)` then add accumulation loops (with `accumulation_loop=true` and epilogues).
    - Purpose: reductions need separate accumulation loops with special suffix/epilogue handling; placed before compute because they set up accumulation variables and control flow for inner accumulation.
- **Pre-compute splices:** `code.splice(self.applys)`, `self.indexed_buffer`, `self.dma_loads`
    - Purpose: emit pre-compute transforms and DMA loads (bring data from DRAM→SPAD) before the compute region.
- **Compute body:** `code.writelines(self.compute_body_loop.lines())` then indent inner loop with `{inner_loop=false}` and suffix `compute_body_loop.epilogue_line()`; inside splice `masks`, `loads`, `compute`, `stores`.
    - Purpose: inner (typically vectorized) computation occurs here — loads from SPAD, vector ops, stores back to SPAD. Epilogue supports partial tails/reduction yields.
- **DMA stores / reductions suffix:** `code.splice(self.dma_stores)` then `code.splice(self.reductions_suffix)`
    - Purpose: after compute, write back SPAD→DRAM, and finalize any reduction outputs.
- **Return at end.**
- **Special cases:** when `reduction_depth==0` a dummy loop is created to reuse the same generation pipeline for non-reduction kernels (keeps code paths uniform).Why this order? It strictly follows the dataflow: declare memory → iterate over coarse tiles (outer loops) → for each tile, load data to SPAD (DMA) → perform inner vectorized compute → write results back (DMA store) → finalize reductions. This ordering lets the backend:
    - separate concerns (allocation vs data movement vs compute),
    - emit DMA and vector ops at the right scope (tile vs lane),
    - annotate loops with metadata for simulator/autotuner,
    - handle reductions' special accumulation and epilogues.