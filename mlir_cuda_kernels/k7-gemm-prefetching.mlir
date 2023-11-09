// Manual optimizations.

// Kernel 7.

// GEMM tile async prefetching.
//
// Use 2-stage buffer to parallelize data loads and computation using
// asynchronous copies from GMEM to SMEM.
//
// This approach improves operation pipeline and allows more efficient
// utilization by delegating copies to the hardware and allowing threads
// to perform computation in the meantime to hide memory latency.
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
    // Create N sub-tile buffers in SMEM where N is the number of stages.
    // In this case, 2 stages are used (double/ping-pong buffer).
    //
    // There is a tradeoff between hiding memory latency - fetching data
    // during computation - and resource consumption - increased SMEM consumption
    // per threadblock which can reduce parallism i.e., one SM might not be able
    // to run multiple threadblocks due to lack of SMEM.
    memref.global "private" @smemTileA : memref<2x32x32xf32, 3> // LHS input tile
    memref.global "private" @smemTileB : memref<2x32x32xf32, 3> // RHS input tile

    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %numStages = arith.constant 2 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<2x32x32xf32, 3>
      %smemB = memref.get_global @smemTileB : memref<2x32x32xf32, 3>

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      %numSubTilesK = arith.ceildivsi %dimK, %bDimX : index

      // C tile consecutive elements (tID y, tID x) -> coalesced GMEM read [fast].
      %elemC = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      // Load the first A and B sub-tiles.
      //
      // A tile consecutive elements (tID y, tID x) -> coalesced GMEM load [fast].
      // B tile consecutive elements (tID y, tID x) -> coalesced GMEM load [fast].
      %elemA = memref.load %subview[%2, %3] : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %elemB = memref.load %subview_0[%2, %3] : memref<64x32xf32, strided<[64, 1], offset: ?>>
      memref.store %elemA, %smemA[%c0, %2, %3] : memref<2x32x32xf32, 3>
      memref.store %elemB, %smemB[%c0, %2, %3] : memref<2x32x32xf32, 3>

      // Iterate over reduction dimension tiles minus the (N - 1) tail sub-tiles.
      // These are handled separately in tail computation after this main loop.
      //
      // For this 2-stage prefetching scheme N = 2 so, one last sub-tile
      // has to be skipped to avoid out of bound prefetching.
      %numTailTiles = arith.subi %numStages, %c1 : index
      %ub = arith.subi %numSubTilesK, %numTailTiles : index

      // Accumulate C tile element result in thread register.
      %partRes = scf.for %subtileIv = %c0 to %ub step %c1 iter_args(%acc = %elemC) -> (f32) {
        %nextTileIv = arith.addi %subtileIv, %c1 : index

        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        // The sub-tiles are loaded cooperatively using all threads in a threadblock.
        // Find the start position of a sub-tile.
        %subTileStepRow = arith.muli %nextTileIv, %bDimX : index
        %subTileStepCol = arith.muli %nextTileIv, %bDimY : index
        %offsetA = arith.addi %3, %subTileStepRow : index
        %offsetB = arith.addi %2, %subTileStepCol : index

        %bufferTile = arith.remui %nextTileIv, %numStages : index

        // Synchronize all threads in a threadblock.
        // All initial sub-tile loads or subsequent GEMM computations from the previous
        // step must finish before the sub-tile buffer from the previous iteration can
        // be overwritten with new values.
        gpu.barrier

        // No suitable GPU dialect abstraction to represent CUDA async copy.
        // 'gpu.memcpy async' has no lowering to nvgpu/nvvm operations and does not have suitable
        // element-wise representation (base memref offsets, number of elements).
        // Use directly specialized nvgpu dialect operations.
        //
        // '#gpu.address_space<workgroup>' does not work with nvgpu operations.
        // Instead, use 'NVVM::NVVMMemorySpace::kSharedMemorySpace' attr value 3 directly.
        // TODO: add type conversion to nvgpu-to-nvvm pass.
        //
        // Start asynchronous copy from GMEM to SMEM.
        // This delegates data movement to separate hardware and threads can perform
        // computations in parallel.
        //
        // Create async workload - in this case copies.
        %cp1 = nvgpu.device_async_copy %subview[%2, %offsetA], %smemA[%bufferTile, %2, %3], 1
                : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<2x32x32xf32, 3>
        %cp2 = nvgpu.device_async_copy %subview_0[%offsetB, %3], %smemB[%bufferTile, %2, %3], 1
              : memref<64x32xf32, strided<[64, 1], offset: ?>> to memref<2x32x32xf32, 3>

        // Commit the async work.
        // After this operations, the copies will start being performed.
        // Threads are not blocked and can continue until explicit synchronization barrier.
        %t1 = nvgpu.device_async_create_group %cp1, %cp2

        // No need to synchronize threads here anymore.
        // Continue with the computation while the new sub-tiles are being loaded.

        // Choose the already loaded sub-tiles:
        //   (current sub-tile ID) mod (number of prefetch stages)
        // In this example, two prefetch stages are used (ping-pong buffer).
        %activeTile = arith.remui %subtileIv, %numStages : index

        // GEMM computation.
        %7 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %acc) -> (f32) {
          // A tile same element (subtileIv,tID y, iv) -> broadcast [fast] load from SMEM.
          // B tile consecutive elements (subtileIv, iv, tID x) -> no bank conflicts [fast] load from SMEM.
          %8 = memref.load %smemA[%activeTile, %2, %arg6] : memref<2x32x32xf32, 3>
          %9 = memref.load %smemB[%activeTile, %arg6, %3] : memref<2x32x32xf32, 3>
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %arg7, %10 : f32
          scf.yield %11 : f32
        }

        // Async synchronization barrier.
        // Wait for the sub-tiles prefetching to complete.
        nvgpu.device_async_wait %t1

        scf.yield %7 : f32
      }

      gpu.barrier

      // Tail GEMM computation.
      //
      // Computate partial C tile results using remaining prefetched sub-tiles.
      // For 2-stage, only one extra sub-tile remains.
      %lastTile = arith.remui %ub, %numStages : index
      %res = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %partRes) -> (f32) {
        // A tile same element (subtileIv,tID y, iv) -> broadcast [fast] load from SMEM.
        // B tile consecutive elements (subtileIv, iv, tID x) -> no bank conflicts [fast] load from SMEM.
        %8 = memref.load %smemA[%lastTile, %2, %arg6] : memref<2x32x32xf32, 3>
        %9 = memref.load %smemB[%lastTile, %arg6, %3] : memref<2x32x32xf32, 3>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }

      // Store the final C tile element value.
      // C tile consecutive elements (tID y, tID x) -> coalesced [fast] store to GMEM.
      memref.store %res, %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>
      gpu.return
    }
  }
}
