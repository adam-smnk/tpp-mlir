// Manual optimizations.

// Kernel 10.

// Compute C matrix tiles in a swizzle pattern.
//
// Access C tiles in a swizzle pattern to try to improve L2 cache hit rate.
// The chosen pattern assumes that threadblocks are launched linearly in
// order of increasing linearized threadblock ID.
//
// NOTE: Multiple threadblocks can run in parallel both within a single SM
//       and across multiple SMs. In general, the order in which blocks
//       are scheduled cannot be predicted. Thus, this optimization might
//       not be beneficial.
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
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      // Pick C matrix tile in swizzle pattern.
      //
      // Groups C matrix tiles as swizzle blocks which are accessed in
      // column-major fashion. The swizzle blocks are <2x2> and the swizzle
      // tiles are accessed in row-major fashion.
      //
      // AFTER BENCH NOTE:
      //   The applied swizzle seems to bring minimal performance improvement on A3000
      //   and A100 GPUs.
      //   By tuning the swizzle size, a minor performance boost in observed for GEMM
      //   M=N=K=1024 with '%swizzleSize = 4'. Performance appears to degrade by a large
      //   margin for other swizzle pattern sizes.
      //
      // Linearize threadblock ID for swizzle location calculation.
      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %bRowOffset = arith.muli %1, %gDimX : index
      %bId = arith.addi %bRowOffset, %0 : index

      // NOTE: Should be tuned for performance.
      %swizzleSize = arith.constant 2 : index
      // Assume rectangular access pattern:
      // For examplem, with a C matrix tiled as:
      //   | 0 | 1 | 2 | 3 |
      //   | 4 | 5 | 6 | 7 |
      // Swizzle blocks:
      //   |  S0   |  S2   |
      //   |  S1   |  S3   |
      // Swizzle tiles:
      //   | st(0,0) | st(0,1) | st(2,0) | st(2,1) |
      //   | st(1,0) | st(1,1) | st(3,0) | st(3,1) |
      // C matrix tile access in 2D representation:
      //   0 -> 1 | 2 -> 3
      //      /    ^   /
      //     v    /   v
      //   4 -> 5 | 6 -> 7
      // C matrix tile access in 1D representation:
      //   0 -> 1 -> 4 -> 5 -> 2 -> 3 -> 6 -> 7
      %swizzleBlockSize = arith.muli %swizzleSize, %swizzleSize : index
      // Swizzle <2x2> block ID.
      %swizzleBlockId = arith.divui %bId, %swizzleBlockSize : index
      // Swizzle tile ID within a swizzle block. 
      %swizzleId = arith.remui %bId, %swizzleBlockSize : index

      // Map swizzle block and tile to C matrix tile position.
      //
      // Compute C tile row position.
      %swizzleWrapSize = arith.divui %gDimY, %swizzleSize : index
      %swizzleblockRow = arith.remui %swizzleBlockId, %swizzleWrapSize : index
      %swizzleRow = arith.divui %swizzleId, %swizzleSize : index
      %rowOffsetC = arith.muli %swizzleblockRow, %swizzleSize : index
      %tileRowC = arith.addi %rowOffsetC, %swizzleRow : index // New C tile row ID

      // Compute C tile column position.
      %swizzleCol = arith.remui %swizzleId, %swizzleSize : index
      %colOffsetStepC = arith.divui %swizzleBlockId, %swizzleWrapSize : index
      %colOffsetC = arith.muli %colOffsetStepC, %swizzleSize : index
      %tileColC = arith.addi %colOffsetC, %swizzleCol : index // New C tile column ID

      // Apply mapping from the new C tile positions to the offsets pointing
      // to starting elements of a C tile within the C matrix.
      %4 = affine.apply #map(%tileRowC)
      %5 = affine.apply #map(%tileColC)

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x4xf32, #gpu.address_space<workgroup>>
      %smemB = memref.get_global @smemTileB : memref<4x32xf32, #gpu.address_space<workgroup>>

      // Thread tile sizes.
      %TM = arith.constant 4 : index
      %TN = arith.constant 4 : index

      // Block tile sizes.
      %BM = arith.constant 32 : index
      %BN = arith.constant 32 : index
      %BK = arith.constant 4 : index

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
        %subtileOffset = arith.muli %subtileIv, %BK : index

        scf.for %rowOffset = %c0 to %BM step %c16 {
          %rowA = arith.addi %tRowA, %rowOffset : index
          %colA = arith.addi %subtileOffset, %tColA : index

          %elemA = memref.load %subview[%rowA, %colA] : memref<32x64xf32, strided<[64, 1], offset: ?>>
          memref.store %elemA, %smemA[%rowA, %tColA] : memref<32x4xf32, #gpu.address_space<workgroup>>
        }
        scf.for %rowOffset = %c0 to %BK step %c2 {
          %tileStart = arith.addi %tRowB, %subtileOffset : index
          %rowB = arith.addi %tileStart, %rowOffset : index
          %smemRowB = arith.addi %tRowB, %rowOffset : index

          %elemB = memref.load %subview_0[%rowB, %tColB] : memref<64x32xf32, strided<[64, 1], offset: ?>>
          memref.store %elemB, %smemB[%smemRowB, %tColB] : memref<4x32xf32, #gpu.address_space<workgroup>>
        }

        // Synchronize all threads in a threadblock.
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
        gpu.barrier
      }

      // Store the final C tile element values.
      scf.for %tm = %c0 to %TM step %c1 {
        scf.for %tn = %c0 to %TN step %c1 {
          %cRow = arith.addi %threadRow, %tm : index
          %cCol = arith.addi %threadCol, %tn : index

          %res = memref.load %regC[%tm, %tn] : memref<4x4xf32, #gpu.address_space<private>>
          memref.store %res, %subview_1[%cRow, %cCol] : memref<32x32xf32, strided<[64, 1], offset: ?>>
        }
      }

      gpu.return
    }
  }
}
