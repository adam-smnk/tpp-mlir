// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2147483648

// Improved GMEM access with coalesced warp loads.
#map = affine_map<(d0) -> (d0 * 32)>
module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c32, %c32, %c1) threads in (%c32, %c32, %c1)  args(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>, %arg2 : memref<1024x1024xf32>, %c0 : index, %c1024 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 32, 32, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // Swap use of thread ID x and y.
      // Thread ID increase: (x + y Dx)
      // Use thread ID y dim (slower increasing) to iterate over rows.
      // Use thread ID x dim (faster increasing) to iterate over columns.
      %2 = gpu.thread_id  y // Fixed for each warp thread
      %3 = gpu.thread_id  x // Consecutive increase within warp threads
      %4 = affine.apply #map(%0)
      %5 = affine.apply #map(%1)
      %subview = memref.subview %arg0[%4, 0] [32, 1024] [1, 1] : memref<1024x1024xf32> to memref<32x1024xf32, strided<[1024, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %5] [1024, 32] [1, 1] : memref<1024x1024xf32> to memref<1024x32xf32, strided<[1024, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%4, %5] [32, 32] [1, 1] : memref<1024x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
      %6 = memref.load %subview_1[%2, %3] : memref<32x32xf32, strided<[1024, 1], offset: ?>>
      %7 = scf.for %arg6 = %arg3 to %arg4 step %arg5 iter_args(%arg7 = %6) -> (f32) {
        // A tile same element -> broadcast [fast]
        // B tile consecutive elements -> coalesced [fast]
        %8 = memref.load %subview[%2, %arg6] : memref<32x1024xf32, strided<[1024, 1], offset: ?>>
        %9 = memref.load %subview_0[%arg6, %3] : memref<1024x32xf32, strided<[1024, 1], offset: ?>>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }
      // C tile consecutive elements -> coalesced [fast]
      memref.store %7, %subview_1[%2, %3] : memref<32x32xf32, strided<[1024, 1], offset: ?>>
      gpu.return
    }
  }
}
