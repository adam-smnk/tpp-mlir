// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(
      %arg0 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : tensor<2048x512xf32>
  
  // 1x1 Conv2D
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast:.*]] = memref.cast
  // CHECK: %[[cast1:.*]] = memref.cast
  // CHECK: %[[cast2:.*]] = memref.cast
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x7x7x512xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %2 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %1) -> (tensor<1x7x7x512xf32>) {
    %3 = scf.for %arg3 = %c0 to %c7 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x7x7x512xf32>) {
      %4 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x7x7x512xf32>) {
        %5 = scf.for %arg7 = %c0 to %c1 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x7x7x512xf32>) {
          %6 = affine.apply #map(%arg3, %arg5)
          %extracted_slice = tensor.extract_slice %arg0[%arg1, %6, %arg7, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : tensor<1x7x7x2048xf32> to tensor<7x2048xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> to tensor<7x512xf32>
          %7 = linalg.matmul ins(%extracted_slice, %cst : tensor<7x2048xf32>, tensor<2048x512xf32>) outs(%extracted_slice_1 : tensor<7x512xf32>) -> tensor<7x512xf32>
          %inserted_slice = tensor.insert_slice %7 into %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<7x512xf32> into tensor<1x7x7x512xf32>
          scf.yield %inserted_slice : tensor<1x7x7x512xf32>
        }
        scf.yield %5 : tensor<1x7x7x512xf32>
      }
      scf.yield %4 : tensor<1x7x7x512xf32>
    }
    scf.yield %3 : tensor<1x7x7x512xf32>
  }

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : tensor<1x7x7x512xf32>
}
