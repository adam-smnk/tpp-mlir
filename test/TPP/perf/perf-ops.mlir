// RUN: tpp-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: @perf_matmul_bench
func.func @perf_matmul_bench(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: index) {
  // CHECK: perf.bench
  %deltas = perf.bench (%n) {
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.do_not_opt
    perf.do_not_opt(%D) : tensor<4x4xf32>
  } -> memref<?xf64>

  // CHECK: perf.mean
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  // CHECK: perf.stdev
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  return
}

// -----

// CHECK-LABEL: @perf_matmul_loops
func.func @perf_matmul_loops(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: memref.alloc
  %deltas = memref.alloc(%n) : memref<?xf64>
  scf.for %arg0 = %c0 to %n step %c1 {
    // CHECK: perf.start_timer
    %t = perf.start_timer : index
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.do_not_opt
    perf.do_not_opt(%D) : tensor<4x4xf32>
    // CHECK: perf.stop_timer
    %del = perf.stop_timer(%t : index) : f64
    memref.store %del, %deltas[%arg0] : memref<?xf64>
  }

  // CHECK: perf.mean
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  // CHECK: perf.stdev
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  return
}
