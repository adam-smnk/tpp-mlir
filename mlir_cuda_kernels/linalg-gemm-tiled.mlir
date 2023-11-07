// RUN: tpp-opt ../mlir_cuda_kernels/linalg-gemm.mlir \
// RUN: -tile-consumer-and-fuse-producers="tile-sizes=32,32" \
// RUN: -bufferize=dealloc=false \
// RUN: -convert-forall-to-parallel

// Kernel 2.

// First tile the GEMM computation.
// 
// One GEMM tile is mapped into one GPU threadblock.
// Each threadblock is mapped into number of warps -> 32 threads per warp.
// HARDWARE LIMIT: Max 1024 threads per block. Limit of blocks can be ignored.
module {
  func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    // Calculations assume only square matrices for simplicity.
    //
    // Tiled into <32x32> warp-sized tiles.
    // This utilizes 32 warps of 32 threads per a threadblock.
    // Each warp computes one sub-tile (row) of the output matrix C.
    // Each thread computes one element of the output matrix C.
    // 
    // Warp tiles can be larger, however, hardware limits
    // number of threads per block to: 1024 threads.
    // This requires coarse-grain threading i.e., each warp processes
    // multiple sub-tiles of the output matrix C.
    // Each warp computes multiple sub-tiles of the output matrix C.
    // Each thread computes multiple elements of the output matrix C.
    //
    // Number of threadblocks (grid size) is implicitly based on:
    //   (num threadblocks) = (matrix dim size) / (warp tile size)
    // Giving <2x2> grid -> 4 threadblocks of <32x32> threads -> 32 warps.
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
      %subview = memref.subview %arg0[%arg3, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %arg4] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
      linalg.matmul ins(%subview, %subview_0 : memref<32x64xf32, strided<[64, 1], offset: ?>>, memref<64x32xf32, strided<[64, 1], offset: ?>>) outs(%subview_1 : memref<32x32xf32, strided<[64, 1], offset: ?>>)
      scf.yield
    }
    return
  }
}
