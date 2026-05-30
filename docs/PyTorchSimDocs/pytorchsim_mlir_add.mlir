memref.global @buf0_spad : memref<2xf32, 1>
memref.global @buf1_spad : memref<2xf32, 1>
memref.global @buf2_spad : memref<2xf32, 1>
func.func @kernel(%in_ptr0: memref<2xf32>,
                       %in_ptr1: memref<2xf32>,
                       %out_ptr0: memref<2xf32>)
{
    %const0 = arith.constant 0 : index
    %const1 = arith.constant 2 : index
    %const2 = arith.constant 3 : index
    %alloc0 = memref.alloc() : memref<1xi32> // 0
    %alloc1 = memref.alloc() : memref<1xi32> // 1
    %alloc2 = memref.alloc() : memref<1xi32> // 2
    %spad0 = memref.get_global @buf0_spad : memref<2xf32, 1>
    %spad1 = memref.get_global @buf1_spad : memref<2xf32, 1>
    %spad2 = memref.get_global @buf2_spad : memref<2xf32, 1>
    affine.for %index0 = 0 to 2 step 2
    {
        memref.dma_start %in_ptr0[%index0], %spad0[%const0], %const1, %alloc0[%const0], %const0, %const1 : memref<2xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
        memref.dma_start %in_ptr1[%index0], %spad1[%const0], %const1, %alloc1[%const0], %const0, %const1 : memref<2xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
        affine.for %compute_idx = 0 to 2 step 2
        {
            %tmp0 = affine.vector_load %spad0[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp1 = affine.vector_load %spad1[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp2 = arith.addf %tmp0, %tmp1 : vector<2xf32>
            affine.vector_store %tmp2, %spad2[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
        } {inner_loop=false}
        memref.dma_start %spad2[%const0], %out_ptr0[%index0], %const2, %alloc2[%const0], %const0, %const1 : memref<2xf32, 1>, memref<2xf32>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
    } {outer_loop=true}
    return
}