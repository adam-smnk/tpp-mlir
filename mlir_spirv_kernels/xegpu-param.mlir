// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2147483328

// Coarse threading kernel with tuned parameters.
//
// Tile size: <16x32> - 4 DPAS <8x16> tiles
// PARAM: map step has to match GEMM tile size
#mapRow = affine_map<(d0) -> (d0 * 16)>
#mapCol = affine_map<(d0) -> (d0 * 32)>

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<128x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<128x1024xf32>) -> memref<1024x1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index

    %tileSizeX = arith.constant 16 : index
    %tileSizeY = arith.constant 32 : index

    %dimM = memref.dim %arg2, %c0 : memref<128x1024xf32>
    %dimN = memref.dim %arg2, %c1 : memref<128x1024xf32>
    %numTilesX = arith.divui %dimM, %tileSizeX : index
    %numTilesY = arith.divui %dimN, %tileSizeY : index

    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%numTilesX, %numTilesY, %c1) threads in (%c2, %c2, %c1)  args(%arg0 : memref<128x1024xf16>, %arg1 : memref<1024x1024xf16>, %arg2 : memref<128x1024xf32>, %c0 : index, %c32 : index, %c1 : index)
    return %arg2 : memref<1024x1024xf32>
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<128x1024xf32>) kernel attributes {gpu.known_block_size = array<i32: 2, 2, 1>, gpu.known_grid_size = array<i32: 32, 32, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #mapRow(%0)
      %5 = affine.apply #mapCol(%1)
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y // Contiguous vectorizable dimension.

      // Block tiles.
      // Created by initial GEMM tiling.
      // Each thread block computes one C tile.
      //
      // PARAM: sizes have to match new GEMM tile size
      %blockA = memref.subview %A[%4, 0] [16, 1024] [1, 1] : memref<128x1024xf16> to memref<16x1024xf16, strided<[1024, 1], offset: ?>>
      %blockB = memref.subview %B[0, %5] [1024, 32] [1, 1] : memref<1024x1024xf16> to memref<1024x32xf16, strided<[1024, 1], offset: ?>>
      %blockC = memref.subview %C[%4, %5] [16, 32] [1, 1] : memref<128x1024xf32> to memref<16x32xf32, strided<[1024, 1], offset: ?>>

      // Thread tile sizes.
      // Each thread will compute <1x1> DPAS tile (<8x16> elements) of C tile.
      %TM = arith.constant 8 : index
      %TN = arith.constant 16 : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      //
      // PARAM: block sizes BM and BN have to match GEMM tile size
      %BM = memref.dim %blockC, %c0 : memref<16x32xf32, strided<[1024, 1], offset: ?>>
      %BN = memref.dim %blockC, %c1 : memref<16x32xf32, strided<[1024, 1], offset: ?>>
      %BK = arith.constant 32 : index // == %blockDimK - matches inner block tile dim size
      %numThreadTiles = arith.divui %BK, %TN : index

      // Find size of the GEMM tiles reduction dimension.
      %dimK = memref.dim %blockA, %c1 : memref<16x1024xf16, strided<[1024, 1], offset: ?>>
      %numSubTilesK = arith.ceildivsi %dimK, %BK : index

      // Needs constant value for better optimizations.
      // %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      // %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      %bDimX = arith.constant 2 : index
      %bDimY = arith.constant 2 : index
      %blockSize = arith.muli %bDimX, %bDimY : index

      // Initialize accumulator registers.
      //
      // Each thread loads C tiles it will compute.
      %tileC = xegpu.create_nd_tdesc %blockC[%2, %3] {mode = vc} : memref<16x32xf32, strided<[1024, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
      %vC = xegpu.load_nd %tileC {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      %res = scf.for %subtileIv = %c0 to %numSubTilesK step %c1 iter_args(%acc = %vC) -> (vector<8x16xf32>) {
        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        // The sub-tiles are loaded cooperatively using all threads in a threadblock.
        // Find the start position of a sub-tile.
        %subtileOffset = arith.muli %subtileIv, %BK : index
        %subA = memref.subview %blockA[%subtileOffset, 0] [16, 32] [1, 1] : memref<16x1024xf16, strided<[1024, 1], offset: ?>> to memref<16x32xf16, strided<[1024, 1], offset: ?>>
        %subB = memref.subview %blockB[0, %subtileOffset] [32, 32] [1, 1] : memref<1024x32xf16, strided<[1024, 1], offset: ?>> to memref<32x32xf16, strided<[1024, 1], offset: ?>>

        // Fetch data from GMEM to SMEM using all threads in a threadblock.
        // Each thread has to load 1 tile of A and B from their block tiles.
        %tileA = xegpu.create_nd_tdesc %subA[%2, %3] {mode = vc} : memref<16x32xf16, strided<[1024, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16>
        %tileB = xegpu.create_nd_tdesc %subB[%2, %3] {mode = vc} : memref<32x32xf16, strided<[1024, 1], offset: ?>> -> !xegpu.tensor_desc<16x16xf16>

        // Use prefetching to cache all the A and B sub-tiles.
        // They will be shared among threads within the block.
        xegpu.prefetch_nd %tileA {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %tileB {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<16x16xf16>

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        // TOOD: see if needed, HW might enforce cache coherency on its own
        gpu.barrier

        // GEMM computation.
        // TODO: unroll this loop
        %partRes = scf.for %tOffset = %c0 to %numThreadTiles step %c1 iter_args(%valC = %acc) -> (vector<8x16xf32>) {
          %tA = xegpu.create_nd_tdesc %subA[%2, %tOffset] {mode = vc} : memref<16x32xf16, strided<[1024, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16>
          %tB = xegpu.create_nd_tdesc %subB[%tOffset, %3] {mode = vc} : memref<32x32xf16, strided<[1024, 1], offset: ?>> -> !xegpu.tensor_desc<16x16xf16>
          %vA = xegpu.load_nd %tA {mode = vc, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
          %vB = xegpu.load_nd %tB {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %dpas = xegpu.dpas %vA, %vB, %valC {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          scf.yield %dpas : vector<8x16xf32>
        }

        // Synchronize all threads in a threadblock.
        // All current computations have to be finished before SMEM A and B tiles can be
        // replaced with new values (new tiles) from GMEM.
        // TODO: see if needed, cache might be large enough to allow prefetching of the next set of tiles
        gpu.barrier

        scf.yield %partRes : vector<8x16xf32>
      }

      // Store the final C tile element values.
      xegpu.store_nd %res, %tileC {mode = vc} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      gpu.return
    }
  }
}
