// RUN: tpp-run %s -gpu=cuda -entry-point-result=void -e entry

module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global @data : memref<4x4xf32> = dense<[
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]
    ]>

    gpu.func @hello() kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id y
      %3 = gpu.thread_id x
      %data = memref.get_global @data : memref<4x4xf32>
      %4 = memref.load %data[%2, %3] : memref<4x4xf32>
      gpu.printf "Block: (%lld, %lld), Thread: (%lld, %lld), value: %f\n" %0, %1, %2, %3, %4 : index, index, index, index, f32
      gpu.return
    }
  }

  func.func @entry() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    gpu.launch_func @kernels::@hello
      blocks in (%c2, %c2, %c1)
      threads in (%c4, %c4, %c1)
    return
  }
}
