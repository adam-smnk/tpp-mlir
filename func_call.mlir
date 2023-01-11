scf.for %arg1 = %c0 to %c1 step %c1 {
  scf.for %arg2 = %c0 to %c7 step %c1 {
    %3 = affine.apply #map2()[%arg1, %arg2]
    %reinterpret_cast = memref.reinterpret_cast %alloc_0 to offset: [%3], sizes: [7, 512], strides: [512, 1] : memref<1x7x7x512xf32> to memref<7x512xf32, #map3>
    %4 = func.call @xsmm_unary_dispatch(%c1_i64, %c7_i64, %c512_i64, %c512_i64, %c512_i64, %c1_i64, %c4_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    %cast = memref.cast %2 : memref<512xf32> to memref<*xf32>
    %cast_3 = memref.cast %reinterpret_cast : memref<7x512xf32, #map3> to memref<*xf32>
    func.call @xsmm_unary_invoke(%c1_i64, %4, %cast, %cast_3) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
  }
}
