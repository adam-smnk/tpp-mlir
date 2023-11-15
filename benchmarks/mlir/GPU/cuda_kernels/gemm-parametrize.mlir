// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2147483328

// Coarse threading kernel with tuned parameters.
//
// Tile size: 32
// PARAM: map step has to match GEMM tile size
#map = affine_map<(d0) -> (d0 * 32)>

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    // In this example, the matmul tile is <32x32>.
    // However, each thread will compute <8x8> C tile elements.
    // Thus, the block size is reduced to <16x16> threads.
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c32, %c32, %c1) threads in (%c8, %c8, %c1)  args(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>, %arg2 : memref<1024x1024xf32>, %c0 : index, %c32 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    // PARAM: SMEM caches have to match <(GEMM tile size)x(reduction dim step)>
    memref.global "private" @smemTileA : memref<32x4xf32, #gpu.address_space<workgroup>>
    memref.global "private" @smemTileB : memref<4x32xf32, #gpu.address_space<workgroup>>

    gpu.func @entry_kernel(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 16, 16, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.
      // gpu.printf "block: %lld %lld\n" %1, %0 : index, index
      // gpu.printf "thread: %lld %lld\n" %2, %3 : index, index

      // PARAM: sizes have to match new GEMM tile size
      %subview = memref.subview %arg0[%4, 0] [32, 1024] [1, 1] : memref<1024x1024xf32> to memref<32x1024xf32, strided<[1024, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [1024, 32] [1, 1] : memref<1024x1024xf32> to memref<1024x32xf32, strided<[1024, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<1024x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>

      %smemA = memref.get_global @smemTileA : memref<32x4xf32, #gpu.address_space<workgroup>>
      %smemB = memref.get_global @smemTileB : memref<4x32xf32, #gpu.address_space<workgroup>>

      // Thread tile sizes.
      // Each thread will compute <4x4> elements of C tile.
      %TM = arith.constant 4 : index
      %TN = arith.constant 4 : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      //
      // PARAM: block sizes BM and BN have to match GEMM tile size
      %BM = memref.dim %subview, %c0 : memref<32x1024xf32, strided<[1024, 1], offset: ?>>
      %BN = memref.dim %subview_0, %c1 : memref<1024x32xf32, strided<[1024, 1], offset: ?>>
      %BK = arith.constant 4 : index

      // Find size of the GEMM tiles reduction dimension.
      // Rectangular sub-tile shape is assumed for simplicity.
      %dimK = memref.dim %subview, %c1 : memref<32x1024xf32, strided<[1024, 1], offset: ?>>
      %numSubTilesK = arith.ceildivsi %dimK, %BK : index

      // Needs constant value for better optimizations.
      // %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      // %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      %bDimX = arith.constant 8 : index
      %bDimY = arith.constant 8 : index
      %blockSize = arith.muli %bDimX, %bDimY : index

      // Thread caches.
      // PARAM: sizes have to match TM and TN
      //
      // Accumulate C tile elements in thread registers.
      // Each thread computes <TMxTN> C tile elements.
      %regC = memref.alloca(%TM, %TN) : memref<?x?xf32, #gpu.address_space<private>>
      // Thread tile single row (A tile) and col (B tile) register caches.
      %regA = memref.alloca(%TM) : memref<?xf32, #gpu.address_space<private>>
      %regB = memref.alloca(%TN) : memref<?xf32, #gpu.address_space<private>>

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
          %elemC = memref.load %subview_1[%cRow, %cCol] : memref<32x32xf32, strided<[1024, 1], offset: ?>>
          memref.store %elemC, %regC[%tm, %tn] : memref<?x?xf32, #gpu.address_space<private>>
        }
      }

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
        //
        // PARAM: Step size here should be:
        // numStepsA = (SMEM cache size = BM * BK) / (threadblocksize = 'block dim x' * 'block dim y')
        %smemSizeA = arith.muli %BM, %BK : index
        %numStepsA = arith.divui %smemSizeA, %blockSize : index
        %numElemPerT = arith.constant 1 : index
        %stepSize = arith.muli %blockSize, %numElemPerT : index
        scf.for %subtileStep = %c0 to %numStepsA step %c1 {
          %numLoaded = arith.muli %subtileStep, %stepSize : index
          %stepRowOffset = arith.divui %numLoaded, %BK : index
          // %rowOffset = arith.muli %subtileStep, %stepRowOffset : index
          %rowA = arith.addi %tRowA, %stepRowOffset : index
          %colA = arith.addi %subtileOffset, %tColA : index

          // A tile 4 consecutive elements -> coalesced GMEM load [medium].
          %elemA = memref.load %subview[%rowA, %colA] : memref<32x1024xf32, strided<[1024, 1], offset: ?>>
          memref.store %elemA, %smemA[%rowA, %tColA] : memref<32x4xf32, #gpu.address_space<workgroup>>
        }
        // PARAM: Step size here:
        // numStepsB = (SMEM cache size = BN * BK) / (threadblocksize = 'block dim x' * 'block dim y')
        %smemSizeB = arith.muli %BK, %BN : index
        %numStepsB = arith.divui %smemSizeB, %blockSize : index
        scf.for %subtileStep = %c0 to %numStepsB step %c1 {
          // %rowOffset = arith.muli %subtileStep, %BK : index
          %numLoaded = arith.muli %subtileStep, %stepSize : index
          %stepRowOffset = arith.divui %numLoaded, %BN : index
          %tileStart = arith.addi %tRowB, %subtileOffset : index
          %rowB = arith.addi %tileStart, %stepRowOffset : index
          %smemRowB = arith.addi %tRowB, %stepRowOffset : index

          // B tile 32 consecutive elements -> coalesced GMEM load [fast].
          %elemB = memref.load %subview_0[%rowB, %tColB] : memref<1024x32xf32, strided<[1024, 1], offset: ?>>
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
            memref.store %elemA, %regA[%tm] : memref<?xf32, #gpu.address_space<private>>
          }
          // gpu.printf "threads: %lld %lld \n" %2, %3 : index, index
          scf.for %tn = %c0 to %TN step %c1 {
            %smemColB = arith.addi %outColOffset, %tn : index

            %cst = arith.constant 0 : index
            %t0 = arith.cmpi eq, %3, %cst : index
            %elemB = memref.load %smemB[%offset, %smemColB] : memref<4x32xf32, #gpu.address_space<workgroup>>
            memref.store %elemB, %regB[%tn] : memref<?xf32, #gpu.address_space<private>>
          }

          // Outer product on A and B register caches.
          scf.for %tm = %c0 to %TM step %c1 {
            scf.for %tn = %c0 to %TN step %c1 {
              %acc = memref.load %regC[%tm, %tn] : memref<?x?xf32, #gpu.address_space<private>>
              %elemA = memref.load %regA[%tm] : memref<?xf32, #gpu.address_space<private>>
              %elemB = memref.load %regB[%tn] : memref<?xf32, #gpu.address_space<private>>
              %mul = arith.mulf %elemA, %elemB : f32
              %partRes = arith.addf %acc, %mul : f32
              memref.store %partRes, %regC[%tm, %tn] : memref<?x?xf32, #gpu.address_space<private>>
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
          %res = memref.load %regC[%tm, %tn] : memref<?x?xf32, #gpu.address_space<private>>
          memref.store %res, %subview_1[%cRow, %cCol] : memref<32x32xf32, strided<[1024, 1], offset: ?>>
        }
      }

      gpu.return
    }
  }
}
