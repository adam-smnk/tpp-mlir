// Manual optimizations.

// Kernel 8.
//
// Step back to 'Kernel 6' as memory prefetching of 'Kernel 7' requires
// use of specialized nvgpu operation.
// For now, let's try different optimizations without prefetching or asynchronous copies.

// Vector load elements by requesting 128 bytes from GMEM (4 f32 elements - CUDA float4 type).
//
// Thread loads are already coalesced for efficiency. However, high number of individual
// load requests creates higher memory controller contention which reduces effective memory
// bandwidth.
// By grouping loads into vector load instructions, the achieved memory bandwidth might be
// higher, increasing performance.
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
    // '#gpu.address_space<workgroup>' lacks proper conversion when lowering through
    // vector operations. Instead, use 'NVVM::NVVMMemorySpace::kSharedMemorySpace'
    // attr value 3 directly for correct PTX instruction generation.
    // TODO: add type conversion somewhere?
    memref.global "private" @smemTileA : memref<32x32xf32, 3> // LHS input tile
    memref.global "private" @smemTileB : memref<32x32xf32, 3> // RHS input tile

    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %vecLoadSize = arith.constant 4 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x32xf32, 3>
      %smemB = memref.get_global @smemTileB : memref<32x32xf32, 3>

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      %numSubTilesK = arith.ceildivsi %dimK, %bDimX : index

      // C tile consecutive elements (tID y, tID x) -> coalesced GMEM read [fast].
      %elemC = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      %res = scf.for %subtileIv = %c0 to %numSubTilesK step %c1 iter_args(%acc = %elemC) -> (f32) {
        // Find the start position of a sub-tile.
        %subTileStepRow = arith.muli %subtileIv, %bDimX : index
        %subTileStepCol = arith.muli %subtileIv, %bDimY : index
        %offsetA = arith.addi %3, %subTileStepRow : index
        %offsetB = arith.addi %2, %subTileStepCol : index

        // Perform vector loads from GMEM to SMEM for better memory bandwidth.
        //
        // Scalar loads are replaced with vector loads by selecting only
        // every 4th thread to perform the data transfer.
        // Since whole threadblock has to be synchronized anyway after loading
        // before computation can start, this branching thread behavior should not
        // bring any performance penalty.
        //
        // AFTER BENCH NOTE:
        //   Normal coalesced loads (See: 'Kernel 6') using all the threads appear
        //   to perform 18% better than this vector load implementation on A3000 GPU.
        //   Coalesced loads still perform 8% better than vector loads using whole warps
        //   instead of every 4th thread (See: 'Extra Kernel 1').
        //   Current guess is that CUDA compiler can optimize better simple coalesced
        //   scalar loads compared to hand writter vector code.
        //   Exact root cause of the performance difference is currently unknown.
        %threadMask = arith.remui %3, %vecLoadSize : index
        %isActive = arith.cmpi eq, %threadMask, %c0 : index
        scf.if %isActive {
          // A tile 4 consecutive elements (tID y, offset + tID x) -> vector GMEM load [fast].
          // B tile 4 consecutive elements (offset + tID y, tID x) -> vector GMEM load [fast].
          %elemsA = vector.load %subview[%2, %offsetA]
                    : memref<32x64xf32, strided<[64, 1], offset: ?>>, vector<4xf32>
          %elemsB = vector.load %subview_0[%offsetB, %3]
                    : memref<64x32xf32, strided<[64, 1], offset: ?>>, vector<4xf32>

          vector.store %elemsA, %smemA[%2, %3]
            : memref<32x32xf32, 3>, vector<4xf32>
          vector.store %elemsB, %smemB[%2, %3]
            : memref<32x32xf32, 3>, vector<4xf32>
        }

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        gpu.barrier

        // GEMM computation.
        %7 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %acc) -> (f32) {
          // A tile same element (tID y, iv) -> broadcast [fast] load from SMEM.
          // B tile consecutive elements (iv, tID x) -> no bank conflicts [fast] load from SMEM.
          %8 = memref.load %smemA[%2, %arg6] : memref<32x32xf32, 3>
          %9 = memref.load %smemB[%arg6, %3] : memref<32x32xf32, 3>
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
