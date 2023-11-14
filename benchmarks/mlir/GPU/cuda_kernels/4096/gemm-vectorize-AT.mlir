// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 137438953472

// Vectorize loads to enforce maximum size (128 byte) loads.
// Transpose A tile while loading from GMEM to SMEM.
#map = affine_map<(d0) -> (d0 * 32)>
module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c8 = arith.constant 8 : index
    // In this example, the matmul tile remains the same as before <32x32>.
    // However, each thread will compute <4x4> C tile elements.
    // Thus, the block size is reduced to <8x8> threads.
    %c128 = arith.constant 128 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c128, %c128, %c1) threads in (%c8, %c8, %c1)  args(%arg0 : memref<4096x4096xf32>, %arg1 : memref<4096x4096xf32>, %arg2 : memref<4096x4096xf32>, %c0 : index, %c4096 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    memref.global "private" @smemTileA : memref<32x4xf32, 3>
    memref.global "private" @smemTileB : memref<4x32xf32, 3>

    gpu.func @entry_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
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

      %subview = memref.subview %arg0[%4, 0] [32, 4096] [1, 1] : memref<4096x4096xf32> to memref<32x4096xf32, strided<[4096, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [4096, 32] [1, 1] : memref<4096x4096xf32> to memref<4096x32xf32, strided<[4096, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x4xf32, 3>
      %smemB = memref.get_global @smemTileB : memref<4x32xf32, 3>

      // Thread tile sizes.
      // Each thread will compute <4x4> elements of C tile.
      %TM = arith.constant 4 : index
      %TN = arith.constant 4 : index
      %sizeTT = arith.muli %TM, %TN : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      %BM = arith.constant 32 : index
      %BN = arith.constant 32 : index
      %BK = arith.constant 4 : index

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x4096xf32, strided<[4096, 1], offset: ?>>
      %numSubTilesK = arith.ceildivsi %dimK, %BK : index

      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.

      // Accumulate C tile elements in thread registers.
      // Each thread computes <TMxTN> C tile elements.
      %regC = memref.alloca() : memref<4x4xf32>

      // Initialize accumulator registers.
      %threadRow = arith.muli %2, %TM : index
      %threadCol = arith.muli %3, %TN : index
      scf.for %tm = %c0 to %TM step %c1 {
        %cRow = arith.addi %threadRow, %tm : index
        // C tile vector load 128B (4 elements) -> coalesced GMEM load [fast].
        %elemsC = vector.load %subview_1[%cRow, %threadCol]
                  : memref<32x32xf32, strided<[4096, 1], offset: ?>>, vector<4xf32>
        vector.store %elemsC, %regC[%tm, %c0]
            : memref<4x4xf32>, vector<4xf32>
      }

      // Thread tile single row (A tile) and col (B tile) register caches.
      %regA = memref.alloca() : memref<4xf32>
      %regB = memref.alloca() : memref<4xf32>

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
          %elemA = memref.load %subview[%rowA, %colA] : memref<32x4096xf32, strided<[4096, 1], offset: ?>>

          // Transpose A tiles.
          %tileRowStep = arith.remui %tId, %BK : index
          %tileStep = arith.divui %tId, %sizeTT : index
          %tileRowOffset = arith.muli %tileStep, %BK : index
          %tileRowPos = arith.addi %tileRowOffset, %tileRowStep : index
          %rowAT = arith.addi %rowOffset, %tileRowPos : index

          %tileColStep = arith.divui %tId, %BK : index
          %colAT = arith.remui %tileColStep, %BK : index

          memref.store %elemA, %smemA[%rowAT, %colAT] : memref<32x4xf32, 3>
        }
        scf.for %rowOffset = %c0 to %BK step %c2 {
          %tileStart = arith.addi %tRowB, %subtileOffset : index
          %rowB = arith.addi %tileStart, %rowOffset : index
          %smemRowB = arith.addi %tRowB, %rowOffset : index

          // B tile 32 consecutive elements -> coalesced GMEM load [fast].
          %elemB = memref.load %subview_0[%rowB, %tColB] : memref<4096x32xf32, strided<[4096, 1], offset: ?>>
          memref.store %elemB, %smemB[%smemRowB, %tColB] : memref<4x32xf32, 3>
        }

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        gpu.barrier

        // GEMM computation.
        %outRowOffset = arith.muli %2, %TM : index
        %outColOffset = arith.muli %3, %TN : index
        scf.for %offset = %c0 to %BK step %c1 {
          // A tile load 128B (4 elements) -> coalesced SMEM load [fast].
          %smemRowOffset = arith.addi %outRowOffset, %outColOffset : index
          %smemRowA = arith.addi %outRowOffset, %offset : index
          %elemsA = vector.load %smemA[%smemRowA, %c0]
                  : memref<32x4xf32, 3>, vector<4xf32>
          vector.store %elemsA, %regA[%c0]
              : memref<4xf32>, vector<4xf32>

          // B tile load 128B (4 elements) -> coalesced SMEM load [fast].
          %elemsB = vector.load %smemB[%offset, %outColOffset]
                  : memref<4x32xf32, 3>, vector<4xf32>
          vector.store %elemsB, %regB[%c0]
              : memref<4xf32>, vector<4xf32>

          // Outer product on A and B register caches.
          scf.for %tm = %c0 to %TM step %c1 {
            scf.for %tn = %c0 to %TN step %c1 {
              %acc = memref.load %regC[%tm, %tn] : memref<4x4xf32>
              %elemA = memref.load %regA[%tm] : memref<4xf32>
              %elemB = memref.load %regB[%tn] : memref<4xf32>
              %mul = arith.mulf %elemA, %elemB : f32
              %partRes = arith.addf %acc, %mul : f32
              memref.store %partRes, %regC[%tm, %tn] : memref<4x4xf32>
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
        %cRow = arith.addi %threadRow, %tm : index
        // C tile vector load 128B (4 elements) -> coalesced GMEM store [fast].
        %res = vector.load %regC[%tm, %c0] : memref<4x4xf32>, vector<4xf32>
        vector.store %res, %subview_1[%cRow, %threadCol]
          : memref<32x32xf32, strided<[4096, 1], offset: ?>>, vector<4xf32>
      }

      gpu.return
    }
  }
}
