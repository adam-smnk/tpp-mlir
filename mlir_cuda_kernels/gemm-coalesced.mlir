// Manual optimizations.

// Kernel 4.

// Improve GMEM access with coalesced warp loads.
//
// Default kernel outlining uses thread ID x as row offset when accessing A tile.
// Due to thread ID increase (x + y Dx), each thread in a warp has different thread ID x
// and the same thread ID y.
// This causes threads in each warp to access elements from different rows of the A tile.
// For current GMEM load, it results in serialized individual loads which are slow.
// If SMEM (shared memory) was used to store A tile (see different optimization pass),
// it would leads to bank conflicts and again serialized data access.
//
// By swapping data access of thread ID x with thread ID y, threads in a warp now access
// consecutive elements of A tile. This allows for load coalescing (vectorized load)
// resulting in faster data access.
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
    gpu.func @entry_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      // Swap use of thread ID x and y.
      // Thread ID increase: (x + y Dx)
      // Use thread ID y dim (slower increasing) to iterate over rows.
      // Use thread ID x dim (faster increasing) to iterate over columns.
      // TODO: consider tweaking kernel outlining or lowering to CUDA/nvvm to prefer
      //       proper row-major access.
      //       Alternatively, add some separate pass or switch to custom kernel outlining.
      //       In case of multidimensional parallel loop:
      //         - use thread ID x as innermost loop iterator
      //         - use thread ID y for the middle loop
      //         - use thread ID z for the outermost loop
      %2 = gpu.thread_id  y // Fixed for each warp thread.
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.
      %subview = memref.subview %arg0[%4, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>

      // C tile consecutive elements -> coalesced [fast].
      %6 = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>

      %7 = scf.for %arg6 = %arg3 to %arg4 step %arg5 iter_args(%arg7 = %6) -> (f32) {
        // A tile same element.
        // '%subview' access:
        //   - row offset '%2' -> thread ID y -> same for every thread in
        //     a warp -> access to the same row in GMEM
        //   - col offset '%arg6' -> loop iterator -> same for every thread
        //     in a warp -> access to the same column in GMEM
        // -> broadcast [fast] load from GMEM.
        //
        // B tile consecutive elements.
        // '%subview_0' access:
        //   - row offset '%2' -> loop iterator -> same for every thread
        //     in a warp -> access to the same row in GMEM
        //   - col offset '%3' -> thread ID x -> consecutive for every
        //     thread in a warp -> access to different consecutive columns
        //     in GMEM
        // -> coalesced [fast] load from GMEM.
        %8 = memref.load %subview[%2, %arg6] : memref<32x64xf32, strided<[64, 1], offset: ?>>
        %9 = memref.load %subview_0[%arg6, %3] : memref<64x32xf32, strided<[64, 1], offset: ?>>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }

      // Each warp (32 threads) accesses one row of the C tile.
      // 
      // All warp threads write consecutive C tile elements.
      // '%subview_1' access:
      //   - row offset '%2' -> thread ID y -> same for every thread
      //     in a warp -> access to the same row in GMEM
      //   - col offset '%3' -> thread ID x -> consecutive for every
      //     thread in a warp -> access to different consecutive columns
      //     in GMEM
      // -> coalesced [fast] store to GMEM.
      memref.store %7, %subview_1[%2, %3] : memref<32x32xf32, strided<[64, 1], offset: ?>>
      gpu.return
    }
  }
}
