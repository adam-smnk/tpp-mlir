// RUN: tpp-opt %s -gpu-vectorize -canonicalize -split-input-file | FileCheck %s

func.func @vectorize_matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_matmul(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.matmul
// CHECK-COUNT-3:   vector.transfer_read
// CHECK:           vector.contract
// CHECK:           vector.transfer_write

// -----

func.func @vectorize_binary(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.sub ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_binary(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.sub
// CHECK-COUNT-2:   vector.transfer_read
// CHECK:           arith.subf
// CHECK:           vector.transfer_write

// -----

func.func @vectorize_unary(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.abs ins(%arg0 : memref<64x64xf32>)
               outs(%arg1 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_unary(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.abs
// CHECK-COUNT-1:   vector.transfer_read
// CHECK:           math.absf
// CHECK:           vector.transfer_write
