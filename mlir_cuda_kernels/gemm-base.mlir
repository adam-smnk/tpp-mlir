// RUN: tpp-opt ../mlir_cuda_kernels/linalg-gemm-tiled.mlir -gpu-conversion

// Kernel 3.

// Affine map to find start tile element based on GPU block ID.
// Init tile element offset:
//   (block ID) * (tile size)
// Here, the tile size is square <32x32>.
// GEMM matrices are also square. Thus, the same map can be used for both
// row and column offsets.
#map = affine_map<(d0) -> (d0 * 32)>

// Convert tiled GEMM computation into a GPU kernel.
//
// Kernel launch (num of blocks and num of threads) is created
// based on the two nested parallel loops (num of tiles and tiles dims*).
// *might differ due to hardware limitations - max 1024 threads.
//
// Num blocks: output loop created by tiling.
// Num threads: inner loop created by lowering linalg op with a tile to a scalar representation.
//
// linalg.matmul is lowered into a GPU kernel.
// Scalar operations are mapped into individual GPU threads.
module attributes {gpu.container_module} {
  // Buffers passed as arguments are assumed to be already preallocated on GPU 
  // and reside in GPU global memory (GMEM).
  func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // Grid: 2D grid -> <2x2> threadblocks
    // Block: 2D threadblock -> 1024 threads -> <32x32> threads -> 32 warps
    //
    // Warp split:
    // "The way a block is split into warps is always the same;
    //  each warp contains threads of consecutive, increasing thread IDs with
    //  the first warp containing thread 0."
    // (section 3.2)
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c2, %c2, %c1) threads in (%c32, %c32, %c1)  args(%arg0 : memref<64x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<64x64xf32>, %c0 : index, %c64 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      // Block IDs are used to find the output tile location in the matrix C.
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // Affine maps are applied to the block IDs to find row and col offsets
      // to the first element of the output tile.
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      // Thread IDs are used to find the output element location in the output tile.
      //
      // Thread IDs:
      // “Each thread is identified by its thread ID, which is the thread number within the block.
      //  To help with complex addressing based on the thread ID, an application can also specify
      //  a block as a two- or three-dimensional array of arbitrary size and identify each thread
      //  using a 2- or 3-component index instead. For a two-dimensional block of size (Dx, Dy),
      //  the thread ID of a thread of index (x, y) is (x + y Dx) and for a three-dimensional
      //  block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).”
      // (section 2.2.1)
      //
      // Threads ID order increases first along the x-dimension and then the y-dimension.
      // For examples: T0(x: 0, y: 0), T1(x: 1, y: 0), T2(x: 0, y: 1), T3(x: 1, y: 1)
      //
      // <32x32> threadblock -> 32 threads per warp -> each warp is 1 row of <32x32> block
      //   -> 0 to 31 x IDs for each warp thread
      //   -> fixed y ID for each warp thread - 0 to 31 y IDs for each of 32 warps
      //
      // ID x is consecutive increase within warp threads. 
      %2 = gpu.thread_id  x
      // ID y is fixed for each warp thread.
      %3 = gpu.thread_id  y

      // Input tiles of matrices A and B are accessed based on the position of the output tile, i.e.,
      // row of matrix A tiles and column of matrix B tiles.
      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      // Load element of C tile from GMEM (global memory) -> init accumulator value.
      // All warp threads read consecutive C tile elements -> coalesced [fast] load to GMEM.
      %6 = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      // Iterate over K dimension (reduction dim) of the input tiles.
      // Accumulate value locally in a register.
      %7 = scf.for %arg6 = %arg3 to %arg4 step %arg5 iter_args(%arg7 = %6) -> (f32) {
        // Load elements of A and B tiles from GMEM.
        //
        // Threads access A tile elements in different rows.
        // '%subview' access:
        //   - row offset '%2' -> thread ID x -> consecutive for every
        //     thread in a warp -> access to different rows in GMEM
        //   - col offset '%arg6' -> loop iterator -> same for every thread
        //     in a warp -> access to the same column in GMEM
        // -> serialized [slow] load from GMEM.
        //
        // Threads access the same B tile element.
        // '%subview_0' access:
        //   - row offset '%arg6' -> loop iterator -> same for every thread
        //     in a warp -> access to the same row in GMEM
        //   - col offset '%3' -> thread ID y -> same for every thread in
        //     a warp -> access to the same column in GMEM
        // -> broadcast [fast] load from GMEM.
        //
        // L2 and L1 is caching present by default.
        // However, depending on SMs (streaming multiprocessors) and warps scheduling,
        // cache hit rate (especially in L1) may vary greatly.
        %8 = memref.load %subview[%2, %arg6] : memref<32x64xf32, strided<[64, 1], offset: ?>>
        %9 = memref.load %subview_0[%arg6, %3] : memref<64x32xf32, strided<[64, 1], offset: ?>>
        // Scalar GEMM - (C tile element accumulator) += (A tile element) * (B tile element)
        // Result values is accumulated by each thread in its register.
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }

      // Store new value of C tile element to GMEM.
      //
      // Each warp (32 threads) accesses one column of the C tile.
      // '%subview_1' access:
      //   - row offset '%2' -> thread ID x -> consecutive for every
      //     thread in a warp -> access to different rows in GMEM
      //   - col offset '%3' -> thread ID y -> same for every thread
      //     in a warp -> access to the same column in GMEM
      // -> serialized [slow] store to GMEM.
      memref.store %7, %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>
      gpu.return
    }
  }
}
