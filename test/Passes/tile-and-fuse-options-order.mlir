// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=16,16" -cse -split-input-file | FileCheck %s

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<"CPU"
    : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 64 : i32>>>
} {
  func.func @matmul_dlti(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048x2048xf32>)
      -> tensor<2048x2048xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<2048x2048xf32>, tensor<2048x2048xf32>)
                       outs(%arg2: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %0 : tensor<2048x2048xf32>
  }
}

// Verify that DLTI overrides manual pass options.
// CHECK-LABEL: @matmul_dlti
// CHECK-DAG: %[[TILE:.+]] = arith.constant 64 : index
// CHECK: scf.for %{{.+}} step %[[TILE]]
// CHECK-NEXT: scf.for %{{.+}} step %[[TILE]]
// CHECK: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<64x2048xf32>, tensor<2048x64xf32>)
// CHECK-SAME:                    outs(%{{.+}} : tensor<64x64xf32>)

// -----

module {
  func.func @matmul_only(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048x2048xf32>)
      -> tensor<2048x2048xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<2048x2048xf32>, tensor<2048x2048xf32>)
                       outs(%arg2: tensor<2048x2048xf32>)
      -> tensor<2048x2048xf32>
    return %0 : tensor<2048x2048xf32>
  }
}

// Verify that pass options are used.
// CHECK-LABEL: @matmul_dlti
// CHECK-DAG: %[[TILE:.+]] = arith.constant 16 : index
// CHECK: scf.for %{{.+}} step %[[TILE]]
// CHECK-NEXT: scf.for %{{.+}} step %[[TILE]]
// CHECK: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<16x2048xf32>, tensor<2048x16xf32>)
// CHECK-SAME:                    outs(%{{.+}} : tensor<16x16xf32>)
