// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2147483648

// Basic linalg GEMM kernel.
func.func @entry(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%arg2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}
