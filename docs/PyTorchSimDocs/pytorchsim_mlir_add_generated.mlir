func.func @kernel(
  %arg0_arg: memref<2xf32>,
  %arg1_arg: memref<2xf32>,
  %buf0_arg: memref<2xf32>
) {
affine.for %dummy = 0 to 1 step 1
{
    affine.for %index0 = 0 to 2 step 512
    {
        affine.for %compute_idx = 0 to 1 step 1
        {
            %arg0_load = affine.vector_load %arg0[%compute_idx] : memref<2xf32>, vector<2xf32>
            %arg1_load = affine.vector_load %arg1[%compute_idx] : memref<2xf32>, vector<2xf32>
            %result = arith.addf %arg0_load, %arg1_load : vector<2xf32>
            affine.vector_store %result, %buf0[%compute_idx] : memref<2xf32>, vector<2xf32>
        } {inner_loop=false}
    } {accumulation_loop=true}
} {outer_loop=true}
return

}
