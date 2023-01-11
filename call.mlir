%4 = call @xsmm_unary_dispatch(%c1_i64, %c7_i64, %c2048_i64, %c2048_i64, %c2048_i64, %c1_i64, %c4_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
scf.parallel (%arg1) = (%c0) to (%c7) step (%c1) {
  %subview = memref.subview %alloc_0[0, %arg1, 0, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : memref<1x7x7x2048xf32> to memref<7x2048xf32, strided<[2048, 1], offset: ?>>
  %cast = memref.cast %2 : memref<2048xf32> to memref<*xf32>
  %cast_3 = memref.cast %subview : memref<7x2048xf32, strided<[2048, 1], offset: ?>> to memref<*xf32>
  func.call @xsmm_unary_invoke(%c1_i64, %4, %cast, %cast_3) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
  scf.yield
}
