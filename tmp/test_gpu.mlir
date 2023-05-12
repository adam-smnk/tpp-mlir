// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

!A = memref<8x8xf32>
!B = memref<8x8xf32>
!C = memref<8x8xf32>
!Cv = vector<8x8xf32>

// GEMM packed with tile size: 64, 64, 64
func.func @entry() {
  %0 = memref.alloc() : !A
  %1 = memref.alloc() : !B
  %2 = memref.alloc() : !C

  %cast_a = memref.cast %0 : !A to memref<*xf32>
  gpu.host_register %cast_a : memref<*xf32>
  %cast_b = memref.cast %1 : !B to memref<*xf32>
  gpu.host_register %cast_b : memref<*xf32>
  %cast_c = memref.cast %2 :!C to memref<*xf32>
  gpu.host_register %cast_c : memref<*xf32>

  linalg.matmul ins(%0, %1 : !A, !B) outs(%2 : !C)

  call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(memref<*xf32>)
