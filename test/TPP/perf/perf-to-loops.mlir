// RUN: tpp-opt %s -convert-perf-to-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @perf_single_op
func.func @perf_single_op(%arg0: tensor<4x8xf32>,
          %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>, %arg3: i64) {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: %[[ub:.*]] = arith.index_cast %arg3 : i64 to index
  // CHECK: %[[deltas:.*]] = memref.alloc(%[[ub]])
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   %[[timer:.*]] = perf.start_timer
  // CHECK:   %[[val:.*]] = linalg.matmul
  // CHECK:   %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK:   perf.do_not_opt(%[[val]])
  // CHECK:   memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK: }
  %deltas = perf.bench (%arg3) {
    %D = linalg.matmul ins(%arg0, %arg1: tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.do_not_opt(%D) : tensor<4x4xf32>
  } -> memref<?xf64>

  // CHECK: perf.mean(%[[deltas]] {{.*}})
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  // CHECK:  memref.dealloc %[[deltas]]
  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @perf_multi_op
func.func @perf_multi_op(%arg0: tensor<4x8xf32>,
          %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>) {
  %f42 = arith.constant 42.0 : f32
  %c50 = arith.constant 50 : i64

  // CHECK-DAG: %[[numIter:.*]] = arith.constant 50 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: %[[deltas:.*]] = memref.alloc(%[[numIter]])
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[numIter]] step %[[step]] {
  // CHECK:   %[[timer:.*]] = perf.start_timer
  // CHECK:   tensor.empty
  // CHECK:   linalg.fill
  // CHECK:   linalg.matmul
  // CHECK:   %[[val:.*]] = linalg.generic
  // CHECK:   %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK:   perf.do_not_opt(%[[val]])
  // CHECK:   memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK: }
  %deltas = perf.bench (%c50) {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%f42 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %D = linalg.matmul ins(%arg0, %arg1: tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2: tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%D : tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) {
      ^bb0(%in : f32, %out: f32):
          %3 = arith.addf %in, %out : f32
          linalg.yield %3 : f32
    } -> tensor<4x4xf32>
    perf.do_not_opt(%2) : tensor<4x4xf32>
    perf.yield
  } -> memref<50xf64>

  // CHECK: perf.mean(%[[deltas]] {{.*}})
  %mean = perf.mean(%deltas : memref<50xf64>) : f64
  // CHECK:  memref.dealloc %[[deltas]]
  memref.dealloc %deltas : memref<50xf64>
  return
}
