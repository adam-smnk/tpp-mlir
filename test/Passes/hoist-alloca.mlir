// RUN: tpp-opt %s -hoist-statically-bound-allocations | FileCheck %s

func.func @nested_op_alloca(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  scf.for %iv = %c0 to %arg0 step %c5 {
    %1 = memref.alloca() : memref<64xi32>
  }
  return
}

// CHECK: func.func @nested_op_alloca(
// CHECK-DAG: %c0 = arith.constant 0
// CHECK-DAG: %c5 = arith.constant 5
// CHECK-DAG: %alloca = memref.alloca()
// CHECK: scf.for{{.*}}{
// CHECK-NOT: {{.*}}= memref.alloca()
// CHECK: }
