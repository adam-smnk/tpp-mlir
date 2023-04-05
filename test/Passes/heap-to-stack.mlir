// RUN: tpp-opt %s -heap-to-stack -canonicalize -split-input-file | FileCheck %s

func.func @alloc_small_1d(%arg0: memref<8x8xbf16>, %arg1: memref<8xbf16>) -> memref<8xbf16> {
  %alloc = memref.alloc() : memref<8xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x8xbf16>, memref<8xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<8xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @alloc_small_1d(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca()
// CHECK: linalg.matvec
// CHECK-NOT: memref.dealloc

// -----

func.func @alloc_small_2d(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  memref.dealloc %alloc : memref<8x8xbf16>
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @alloc_small_2d(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca()
// CHECK: linalg.matmul
// CHECK-NOT: memref.dealloc

// -----

func.func @alloc_large_1d(%arg0: memref<8x16384xbf16>, %arg1: memref<8xbf16>) -> memref<8xbf16> {
  %alloc = memref.alloc() : memref<16384xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x16384xbf16>, memref<16384xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<16384xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @alloc_large_1d(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matvec
// CHECK: memref.dealloc

// -----

func.func @alloc_large_2d(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) -> memref<512x512xbf16> {
  %alloc = memref.alloc() : memref<512x512xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<512x512xbf16>, memref<512x512xbf16>) outs(%arg1 : memref<512x512xbf16>)
  memref.dealloc %alloc : memref<512x512xbf16>
  return %arg1 : memref<512x512xbf16>
}

// CHECK-LABEL: func.func @alloc_large_2d(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matmul
// CHECK: memref.dealloc

// -----

func.func @no_dealloc(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @no_dealloc(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matmul

// -----

func.func @dynamic_size(%arg0: memref<8x?xbf16>, %arg1: memref<8xbf16>, %size: index) -> memref<8xbf16> {
  %alloc = memref.alloc(%size) : memref<?xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x?xbf16>, memref<?xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<?xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @dynamic_size(
// CHECK-NOT: memref.alloca(
// CHECK: memref.alloc(
// CHECK: linalg.matvec
// CHECK: memref.dealloc

// -----

func.func @alignment(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  memref.dealloc %alloc : memref<8x8xbf16>
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @alignment(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca() {alignment = 64 : i64}
// CHECK: linalg.matmul
// CHECK-NOT: memref.dealloc

// -----

func.func @scope_return(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>, %arg2: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() : memref<8x8xbf16>
  %alloc1 = memref.alloc() : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  linalg.copy ins(%arg1 : memref<8x8xbf16>) outs(%alloc1 : memref<8x8xbf16>)
  memref.dealloc %alloc : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc1 : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg2 : memref<8x8xbf16>)
  return %arg2 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @scope_return(
// CHECK-SAME:  %[[ARG0:[^ ]+]]: memref<8x8xbf16>,
// CHECK-SAME:  %[[ARG1:[^ ]+]]: memref<8x8xbf16>,
// CHECK-SAME:  %[[ARG2:[^ ]+]]: memref<8x8xbf16>)
// CHECK:     %[[scopeRet:.+]] = memref.alloca_scope
// CHECK-NOT:   memref.alloc()
// CHECK:       %[[alloca:.+]] = memref.alloca()
// CHECK:       %[[alloc:.+]] = memref.alloc()
// CHECK:       linalg.matmul ins(%[[ARG0]], %[[alloca]]
// CHECK:       linalg.copy
// CHECK-NOT:   memref.dealloc
// CHECK:       memref.alloca_scope.return %[[alloc]]
// CHECK:     }
// CHECK-NOT: memref.dealloc
// CHECK: linalg.matmul ins(%[[ARG0]], %[[scopeRet]]

// -----

func.func private @alloca_with_loop(%arg0: memref<64xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %arg1 = arith.constant 0.1 : f32
  %cst = arith.constant 0.000000e+00 : f32
  %cast = memref.cast %arg0 : memref<64xf32> to memref<?xf32>
  %dim = memref.dim %cast, %c0 : memref<?xf32>
  %a0 = memref.alloca_scope  -> (f32) {
    %buf = memref.alloca() : memref<1xf32>
    %0 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %al = memref.load %buf[%c0] : memref<1xf32>
      %5 = memref.load %cast[%arg2] : memref<?xf32>
      %6 = arith.subf %5, %arg1 : f32
      %7 = arith.mulf %6, %6 : f32
      %8 = arith.addf %5, %al : f32
      %9 = arith.addf %8, %al : f32
      memref.store %9, %buf[%c0] : memref<1xf32>
      scf.yield %9 : f32
    }
    %al = memref.load %buf[%c0] : memref<1xf32>
    %1 = arith.addf %0, %al : f32
    memref.alloca_scope.return %1 : f32
  }
  %1 = arith.index_cast %dim : index to i64
  %2 = arith.sitofp %1 : i64 to f32
  %3 = arith.divf %a0, %2 : f32
  %4 = math.sqrt %3 : f32
  return %4 : f32
}

// -----

func.func private @loop_with_alloca(%arg0: memref<64xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %arg1 = arith.constant 0.1 : f32
  %cst = arith.constant 0.000000e+00 : f32
  %cast = memref.cast %arg0 : memref<64xf32> to memref<?xf32>
  %dim = memref.dim %cast, %c0 : memref<?xf32>
  %0 = scf.for %it1 = %c0 to %c2 step %c1 iter_args(%arg3 = %cst) -> (f32) {
    %a0 = memref.alloca_scope  -> (f32) {
      %buf = memref.alloca() : memref<1xf32>
      %al = memref.load %buf[%c0] : memref<1xf32>
      %5 = memref.load %cast[%it1] : memref<?xf32>
      %6 = arith.subf %5, %arg1 : f32
      %7 = arith.mulf %6, %6 : f32
      %8 = arith.addf %5, %al : f32
      %9 = arith.addf %8, %al : f32
      memref.store %9, %buf[%c0] : memref<1xf32>
      %al1 = memref.load %buf[%c0] : memref<1xf32>
      %1 = arith.addf %arg3, %al1 : f32
      memref.alloca_scope.return %1 : f32
    }
    %1 = arith.addf %arg3, %a0 : f32
    scf.yield %1 : f32
  }
  %1 = arith.index_cast %dim : index to i64
  %2 = arith.sitofp %1 : i64 to f32
  %3 = arith.divf %0, %2 : f32
  %4 = math.sqrt %3 : f32
  return %4 : f32
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func private @scope_example(%arg0: memref<32x256xf32>, %arg1: memref<256x32xf32>, %arg2: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cUpper = arith.constant 5 : index
  scf.for %it1 = %c0 to %cUpper step %c1 {
    %r0 = memref.alloca_scope -> (f32) {
      %buf = memref.alloca() : memref<32x32xf32>
      linalg.matmul ins(%arg0, %arg1 : memref<32x256xf32>, memref<256x32xf32>) outs(%buf : memref<32x32xf32>)
      %sum = memref.alloca() : memref<f32>
      linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["reduction", "reduction"]}
        ins(%buf: memref<32x32xf32>) outs(%sum: memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %0 = arith.addf %in, %out : f32
            linalg.yield %0 : f32
      }
      %sumVal = memref.load %sum[] : memref<f32>
      memref.alloca_scope.return %sumVal : f32
    }
    %0 = memref.load %arg2[%c0] : memref<1xf32>
    %1 = arith.addf %r0, %0 : f32
    memref.store %1, %arg2[%c0] : memref<1xf32>
  }
  return
}

