// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" \
// RUN: -convert-linalg-to-tpp -convert-tpp-to-xsmm \
// RUN: -convert-xsmm-to-func | \
// RUN: tpp-run -n 2000\
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \ 
// RUN: -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" \
// RUN: -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s
//

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<1x16xf32>, %output: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x16xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<4x16xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%1 : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<4x16xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<4x16xf32>
  return %3 : tensor<4x16xf32>
}
// Output
// CHECK-COUNT-4: ( 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 )
// Stats
// CHECK: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<8x16xf32>,
// TPP-SAME:  %[[ARG2:.+]]: memref<1x16xf32>,
// TPP-SAME:  %[[ARG3:.+]]: memref<4x16xf32>)
// TPP: tpp.identity ins(%[[ARG2]] : memref<1x16xf32>) out(%[[ARG3:.+]] : memref<4x16xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<4x8xf32>, %[[ARG1]] : memref<8x16xf32>) out(%[[ARG3]] : memref<4x16xf32>)
// TPP: tpp.relu out(%[[ARG3]] : memref<4x16xf32>)
// TPP: return
