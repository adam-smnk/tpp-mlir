// Base kernel.

// Kernel 1.

// Base GEMM computation: C += A x B
func.func @entry(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}
