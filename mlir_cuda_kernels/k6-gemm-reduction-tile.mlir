// Manual optimizations.

// Kernel 6.

// Tile GEMM reduction dimension (K dim).
//
// Tiling GEMM also along the reduction dimension, allows SMEM (shared memory) tile buffers
// to have fixed size and limits SMEM usage.
// This allows for running large GEMMs and can improve parallelism. The latter is possible as
// SMs (streaming multiprocessors) can execute multiple threadblocks together if there are
// sufficient resources i.e., SMEM, registers, threads (max 2048 per SM).
#map = affine_map<(d0) -> (d0 * 32)>
module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c2, %c2, %c1) threads in (%c32, %c32, %c1)  args(%arg0 : memref<64x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<64x64xf32>, %c0 : index, %c64 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    // Split GEMM tiles - GEMM operand tiles coming from initial linalg.matmul tiling - into
    // smaller warp tiles. The warp tile sizes should picked considering the size of the warp.
    // In this case, the warp size is 32 (32 threads are executed together).
    // Thus, common warp tile sizes are combination of following dims: 8, 16, 32, 64, or 128.
    // 
    // Here the warp tile is chosen to match the threadblock size <32x32> with reduction dim
    // also tiled into 32 elements chunks to operate on square warp GEMM computations.
    //
    // Allocate to SMEM buffers to manually cache A and B tiles.
    // C tile element values can be accumulated in thread registers as the C tile <32x32> elements
    // fit in the threadblock <32x32> threads.
    memref.global "private" @smemTileA : memref<32x32xf32, #gpu.address_space<workgroup>> // LHS input tile
    memref.global "private" @smemTileB : memref<32x32xf32, #gpu.address_space<workgroup>> // RHS input tile

    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x32xf32, #gpu.address_space<workgroup>>
      %smemB = memref.get_global @smemTileB : memref<32x32xf32, #gpu.address_space<workgroup>>

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      %numSubTilesK = arith.ceildivsi %dimK, %bDimX : index

      // C tile consecutive elements (tID y, tID x) -> coalesced GMEM read [fast].
      %elemC = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      // Accumulate C tile element result in thread register.
      %res = scf.for %subtileIv = %c0 to %numSubTilesK step %c1 iter_args(%acc = %elemC) -> (f32) {
        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        // The sub-tiles are loaded cooperatively using all threads in a threadblock.
        // Find the start position of a sub-tile.
        %subTileStepRow = arith.muli %subtileIv, %bDimX : index
        %subTileStepCol = arith.muli %subtileIv, %bDimY : index
        %offsetA = arith.addi %3, %subTileStepRow : index
        %offsetB = arith.addi %2, %subTileStepCol : index

        // A tile consecutive elements (tID y, offset + tID x) -> coalesced GMEM load [fast].
        // B tile consecutive elements (offset + tID y, tID x) -> coalesced GMEM load [fast].
        %elemA = memref.load %subview[%2, %offsetA] : memref<32x64xf32, strided<[64, 1], offset: ?>>
        %elemB = memref.load %subview_0[%offsetB, %3] : memref<64x32xf32, strided<[64, 1], offset: ?>>

        memref.store %elemA, %smemA[%2, %3] : memref<32x32xf32, #gpu.address_space<workgroup>>
        memref.store %elemB, %smemB[%2, %3] : memref<32x32xf32, #gpu.address_space<workgroup>>

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        gpu.barrier

        // GEMM computation.
        // Loaded A and B sub-tiles are used to compute partial C tile results.
        //
        // C tile elements are accumulated in thread registers over the whole
        // threadblock as the size of threadblock <32x32> matches the size of C tile <32x32>.
        // That is each thread in a threadblock computes value of one C tile element.
        %7 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %acc) -> (f32) {
          // A tile same element (tID y, iv) -> broadcast [fast] load from SMEM.
          // B tile consecutive elements (iv, tID x) -> no bank conflicts [fast] load from SMEM.
          %8 = memref.load %smemA[%2, %arg6] : memref<32x32xf32, #gpu.address_space<workgroup>>
          %9 = memref.load %smemB[%arg6, %3] : memref<32x32xf32, #gpu.address_space<workgroup>>
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %arg7, %10 : f32
          scf.yield %11 : f32
        }

        // Synchronize all threads in a threadblock.
        // All current computations have to be finished before SMEM A and B tiles can be
        // replaced with new values (new tiles) from GMEM.
        gpu.barrier

        scf.yield %7 : f32
      }

      // Store the final C tile element value.
      // C tile consecutive elements (tID y, tID x) -> coalesced [fast] store to GMEM.
      memref.store %res, %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>
      gpu.return
    }
  }
}
