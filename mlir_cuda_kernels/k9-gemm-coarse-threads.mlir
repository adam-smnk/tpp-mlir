// Manual optimizations.

// Kernel 9.
//
// Step back to 'Kernel 6' as naive load vectorization of 'Kernel 8'
// is currently detrimental to performance.

// Compute multiple output C tile elements in each thread.
//
// Increases arithmentic intensity of the kernel by computing multiple
// C tile elements in each thread. This leads to reduced number of SMEM
// accesses by reusing data in thread registers.
#map = affine_map<(d0) -> (d0 * 32)>
module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    // In this example, the matmul tile remains the same as before <32x32>.
    // However, each thread will compute <4x4> C tile elements.
    // Thus, the block size is reduced to <8x8> threads.
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c2, %c2, %c1) threads in (%c8, %c8, %c1)  args(%arg0 : memref<64x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<64x64xf32>, %c0 : index, %c64 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    memref.global "private" @smemTileA : memref<32x4xf32, #gpu.address_space<workgroup>>
    memref.global "private" @smemTileB : memref<4x32xf32, #gpu.address_space<workgroup>>

    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x4xf32, #gpu.address_space<workgroup>>
      %smemB = memref.get_global @smemTileB : memref<4x32xf32, #gpu.address_space<workgroup>>

      // Thread tile sizes.
      // Each thread will compute <4x4> elements of C tile.
      %TM = arith.constant 4 : index
      %TN = arith.constant 4 : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      %BM = arith.constant 32 : index
      %BN = arith.constant 32 : index
      %BK = arith.constant 4 : index

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %numSubTilesK = arith.ceildivsi %dimK, %BK : index

      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.

      // Accumulate C tile elements in thread registers.
      // Each thread computes <TMxTN> C tile elements.
      %regC = memref.alloca() : memref<4x4xf32, #gpu.address_space<private>>

      // Initialize accumulator registers.
      %threadRow = arith.muli %2, %TM : index
      %threadCol = arith.muli %3, %TN : index
      scf.for %tm = %c0 to %TM step %c1 {
        scf.for %tn = %c0 to %TN step %c1 {
          %cRow = arith.addi %threadRow, %tm : index
          %cCol = arith.addi %threadCol, %tn : index
          // C tile stride 4 elements -> serialized GMEM load [slow].
          // NOTE: These loads can be vectorized to increase performance.
          // NOTE: CUDA compiler might unroll this loop and vectorize the loads.
          %elemC = memref.load %subview_1[%cRow, %cCol] : memref<32x32xf32, strided<[64, 1], offset: ?>>
          memref.store %elemC, %regC[%tm, %tn] : memref<4x4xf32, #gpu.address_space<private>>
        }
      }

      // Thread tile single row (A tile) and col (B tile) register caches.
      %regA = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
      %regB = memref.alloca() : memref<4xf32, #gpu.address_space<private>>

      %tRowOffset = arith.muli %2, %bDimX : index
      %tId = arith.addi %tRowOffset, %3 : index

      %tRowA = arith.divui %tId, %BK : index
      %tColA = arith.remui %tId, %BK : index

      %tRowB = arith.divui %tId, %BN : index
      %tColB = arith.remui %tId, %BN : index

      scf.for %subtileIv = %c0 to %numSubTilesK step %c1 {
        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        // The sub-tiles are loaded cooperatively using all threads in a threadblock.
        // Find the start position of a sub-tile.
        %subtileOffset = arith.muli %subtileIv, %BK : index

        // Fetch data from GMEM to SMEM using all threads in a threadblock.
        // Each thread has to load 2 elements of A and B tiles.
        scf.for %rowOffset = %c0 to %BM step %c16 {
          %rowA = arith.addi %tRowA, %rowOffset : index
          %colA = arith.addi %subtileOffset, %tColA : index

          // A tile 4 consecutive elements -> coalesced GMEM load [medium].
          %elemA = memref.load %subview[%rowA, %colA] : memref<32x64xf32, strided<[64, 1], offset: ?>>
          memref.store %elemA, %smemA[%rowA, %tColA] : memref<32x4xf32, #gpu.address_space<workgroup>>
        }
        scf.for %rowOffset = %c0 to %BK step %c2 {
          %tileStart = arith.addi %tRowB, %subtileOffset : index
          %rowB = arith.addi %tileStart, %rowOffset : index
          %smemRowB = arith.addi %tRowB, %rowOffset : index

          // B tile 32 consecutive elements -> coalesced GMEM load [fast].
          %elemB = memref.load %subview_0[%rowB, %tColB] : memref<64x32xf32, strided<[64, 1], offset: ?>>
          memref.store %elemB, %smemB[%smemRowB, %tColB] : memref<4x32xf32, #gpu.address_space<workgroup>>
        }

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        gpu.barrier

        // GEMM computation.
        %outRowOffset = arith.muli %2, %TM : index
        %outColOffset = arith.muli %3, %TN : index
        scf.for %offset = %c0 to %BK step %c1 {
          scf.for %tm = %c0 to %TM step %c1 {
            %smemRowA = arith.addi %outRowOffset, %tm : index
            %elemA = memref.load %smemA[%smemRowA, %offset] : memref<32x4xf32, #gpu.address_space<workgroup>>
            memref.store %elemA, %regA[%tm] : memref<4xf32, #gpu.address_space<private>>
          }
          scf.for %tn = %c0 to %TN step %c1 {
            %smemColB = arith.addi %outColOffset, %tn : index
            %elemB = memref.load %smemB[%offset, %smemColB] : memref<4x32xf32, #gpu.address_space<workgroup>>
            memref.store %elemB, %regB[%tn] : memref<4xf32, #gpu.address_space<private>>
          }

          // Outer product on A and B register caches.
          scf.for %tm = %c0 to %TM step %c1 {
            scf.for %tn = %c0 to %TN step %c1 {
              %acc = memref.load %regC[%tm, %tn] : memref<4x4xf32, #gpu.address_space<private>>
              %elemA = memref.load %regA[%tm] : memref<4xf32, #gpu.address_space<private>>
              %elemB = memref.load %regB[%tn] : memref<4xf32, #gpu.address_space<private>>
              %mul = arith.mulf %elemA, %elemB : f32
              %partRes = arith.addf %acc, %mul : f32
              memref.store %partRes, %regC[%tm, %tn] : memref<4x4xf32, #gpu.address_space<private>>
            }
          }
        }

        // Synchronize all threads in a threadblock.
        // All current computations have to be finished before SMEM A and B tiles can be
        // replaced with new values (new tiles) from GMEM.
        gpu.barrier
      }

      // Store the final C tile element values.
      scf.for %tm = %c0 to %TM step %c1 {
        scf.for %tn = %c0 to %TN step %c1 {
          %cRow = arith.addi %threadRow, %tm : index
          %cCol = arith.addi %threadCol, %tn : index
          // C tile stride 4 elements -> serialized GMEM store [slow].
          // NOTE: These loads can be vectorized to increase performance.
          // NOTE: CUDA compiler might unroll this loop and vectorize the stores.
          %res = memref.load %regC[%tm, %tn] : memref<4x4xf32, #gpu.address_space<private>>
          memref.store %res, %subview_1[%cRow, %cCol] : memref<32x32xf32, strided<[64, 1], offset: ?>>
        }
      }

      gpu.return
    }
  }
}
