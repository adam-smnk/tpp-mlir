// Manual optimizations.

// Kernel 5.

// Cache A and B tiles in SMEM (shared memory) for better data reuse.
//
// L2 data/instruction cache is shared between all SM (streaming multiprocessor) units.
// L1 data and instruction caches resides in each SM and shared between threads and warps.
// L0 instruction caches reside within each SM.
// In newer architectures, L2 and L1 caching is enabled by default.
//
// Shared memory resides in L1 data cache (carveout). The same on-chip memory storage
// is split and used for both L1 data cache and shared memory.
// Split between L1 and shared memory is automatically enabled by default. The split ratio
// can be explicitly set when initializing a device.
//
// Shared memory acts as a manually managed L1 data cache. Kernel can allocate
// shared memory which is used together by all threads within the same threadblock.
// That is a shared memory buffer is allocated per block.
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
    // Allocate to SMEM buffers to manually cache A and B tiles:
    // loads from L1 data cache -> better memory bandwidth -> faster computation.
    //
    // The buffers are created separately in and shared within each threadblock.
    // The SMEM buffers are stored in part of L1 data cache which usually has
    // limited storage up to 150KB.
    // Exact size of available SMEM is defined by L1 data cache size and L1 split
    // (carveout) config.
    //
    // SMEM buffers are represented as a GPU kernel global memrefs with shared memory
    // attribute (kSharedMemorySpace -> 3 - see NVVMMemorySpace).
    // The tile sizes are known at compile time, thus, the SMEM tile buffers can be stored
    // in static shared memory buffers (fixed sizes opposed to dynamic shared memory).
    //
    // NOTE: reduction dimension (GEMM K dim) is not tiled, thus, the SMEM buffers grow in size
    //       with the input. This will quickly exhaust all available SMEM space and prevent
    //       kernel from running ('uses too much shared data' error).
    //       Therefore, reduction dim tiling will have to be applied first in order to run
    //       any larger GEMM computations.
    //       See: 'kernel 6'.
    memref.global "private" @smemTileA : memref<32x64xf32, 3>
    memref.global "private" @smemTileB : memref<64x32xf32, 3>

    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      // Memref alloc with shared memory space attr i.e., `memref.alloc() : memref<64x32xf32, 3>`,
      // has no default convertion path to a SMEM GPU buffer representation.
      // Therefore, the shared buffers are explicitly represented as kernel globals with SMEM attr.
      // NOTE: it seems possible to use memref.alloc together with NVGPU ops.
      %smemA = memref.get_global @smemTileA : memref<32x64xf32, 3>
      %smemB = memref.get_global @smemTileB : memref<64x32xf32, 3>

      // Find size of the GEMM tiles reduction dimension.
      %dimK = memref.dim %subview, %c1 : memref<32x64xf32, strided<[64, 1], offset: ?>>
      %bDimX = gpu.block_dim x // Threadblock size in X (first) dim.
      %bDimY = gpu.block_dim y // Threadblock size in Y (second) dim.
      // Threadblock has upper bound on its size (max 1024 threads). Therefore,
      // each thread in a threadblock might have to load multiple elements of A and B tiles
      // depending on the threadblock size and GEMM tile size.
      // In this case, every thread has to load two GEMM tile elements.
      //
      // Rectangular sub-tile shape is assumed for simplicity.
      // It is also assumed that GEMM tiles are equal or larger than the threadblock size.
      // 
      // Find number of sub-tiles to be loaded.
      %numSubTilesK = arith.ceildivsi %dimK, %bDimX : index

      // Load whole A and B tiles from GMEM to SMEM.
      // The tiles are loaded cooperatively using all threads in a threadblock.
      scf.for %arg6 = %c0 to %numSubTilesK step %c1 {
        // Find the start position of a sub-tile.
        %subTileStepRow = arith.muli %arg6, %bDimX : index
        %subTileStepCol = arith.muli %arg6, %bDimY : index
        %offsetA = arith.addi %3, %subTileStepRow : index
        %offsetB = arith.addi %2, %subTileStepCol : index

        // A tile consecutive elements (tID y, offset + tID x) -> coalesced GMEM read [fast].
        // B tile consecutive elements (offset + tID y, tID x) -> coalesced GMEM read [fast].
        %elemA = memref.load %subview[%2, %offsetA] : memref<32x64xf32, strided<[64, 1], offset: ?>>
        %elemB = memref.load %subview_0[%offsetB, %3] : memref<64x32xf32, strided<[64, 1], offset: ?>>

        memref.store %elemA, %smemA[%2, %offsetA] : memref<32x64xf32, 3>
        memref.store %elemB, %smemB[%offsetB, %3] : memref<64x32xf32, 3>
      }

      // Synchronize all threads in a threadblock.
      // Whole tiles are needed to perform computation.
      // Wait for all threads in a threadblock to finish loading A and B tile elements.
      gpu.barrier

      // C tile consecutive elements (tID y, tID x) -> coalesced GMEM read [fast].
      %6 = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      // SMEM is split into rows of 32x 4 byte banks -> 128 byte rows -> 32x 4 byte elements.
      //
      // Access pattern to SMEM have performance implication at the warp level:
      //  - When threads access different banks, the accesses happen in parallel [fast].
      //  - When some threads access the same bank, the accesses are serialized [slow].
      //  - When all threads read from the same bank, the result in broadcasted [fast].
      //
      // For example, assume a warp of two threads with each thread accessing
      // an element of a shared memory buffer 'smemBuf : memref<2x32xf32>':
      //   - T0: load smemBuf[0, 0], T1: load smemBuf[0, 1] -> different banks -> no conflict [fast]
      //   - T0: load smemBuf[0, 0], T1: load smemBuf[1, 1] -> different banks -> no conflict [fast]
      //   - T0: load smemBuf[0, 0], T1: load smemBuf[0, 29] -> different banks -> no conflict [fast]
      //   - T0: load smemBuf[0, 0], T1: load smemBuf[1, 0] -> same bank -> conflict [slow]
      //   - T0: load smemBuf[0, 0], T1: load smemBuf[0, 0] -> same element -> broadcast [fast]
      //
      // The above SMEM access patterns affect only threads in the same warp.
      // There is no concept of bank conflicts between threads in different warps.
      %7 = scf.for %arg6 = %arg3 to %arg4 step %arg5 iter_args(%arg7 = %6) -> (f32) {
        // A tile same element (tID y, iv) -> broadcast [fast] load from SMEM.
        // B tile consecutive elements (iv, tID x) -> no bank conflicts [fast] load from SMEM.
        %8 = memref.load %smemA[%2, %arg6] : memref<32x64xf32, 3>
        %9 = memref.load %smemB[%arg6, %3] : memref<64x32xf32, 3>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }
      // C tile consecutive elements (tID y, tID x) -> coalesced GMEM store [fast].
      memref.store %7, %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>
      gpu.return
    }
  }
}
