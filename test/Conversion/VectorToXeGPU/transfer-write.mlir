// RUN: tpp-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @store_1D_vector(%arg0: vector<8xf32>,
    %arg1: memref<8x16x32xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2, %arg2]
    {in_bounds = [true]}
    : vector<8xf32>, memref<8x16x32xf32>
  return
}

// CHECK-LABEL: @store_1D_vector(
// CHECK-SAME:  %[[ARG0:.+]]: vector<8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[ARG1]][%[[ARG2]], %[[ARG2]], %[[ARG2]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       xegpu.store_nd %[[ARG0]], %[[DESC]] : vector<8xf32>

// -----

func.func @store_2D_vector(%arg0: vector<8x16xf32>,
    %arg1: memref<8x16x32xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2, %arg2]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<8x16x32xf32>
  return
}

// CHECK-LABEL: @store_2D_vector(
// CHECK-SAME:  %[[ARG0:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[ARG1]][%[[ARG2]], %[[ARG2]], %[[ARG2]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       xegpu.store_nd %[[ARG0]], %[[DESC]] : vector<8x16xf32>

// -----

func.func @store_dynamic_source(%arg0: vector<8x16xf32>,
    %arg1: memref<?x?x?xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2, %arg2]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<?x?x?xf32>
  return
}

// CHECK-LABEL: @store_dynamic_source(
// CHECK-SAME:  %[[ARG0:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[DIM_0:.+]] = memref.dim %[[ARG1]], %[[C0]]
// CHECK-DAG:   %[[DIM_1:.+]] = memref.dim %[[ARG1]], %[[C1]]
// CHECK-DAG:   %[[DIM_2:.+]] = memref.dim %[[ARG1]], %[[C2]]
// CHECK:       %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[ARG1]][%[[ARG2]], %[[ARG2]], %[[ARG2]]]
// CHECK-SAME:    [%[[DIM_0]], %[[DIM_1]], %[[DIM_2]]], [%[[DIM_0_STRIDE]], %[[DIM_2]], 1]
// CHECK-SAME:    memref<?x?x?xf32> -> !xegpu.tensor_desc<8x16xf32
// CHECK:       xegpu.store_nd %[[ARG0]], %[[DESC]] : vector<8x16xf32>

// -----

func.func @no_store_transposed(%arg0: vector<8x16xf32>,
    %arg1: memref<32x64xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2]
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]}
    : vector<8x16xf32>, memref<32x64xf32>
  return
}

// CHECK-LABEL: @no_store_transposed(
// CHECK:       vector.transfer_write

// -----

func.func @no_store_out_of_bounds(%arg0: vector<8x16xf32>,
    %arg1: memref<32x64xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2]
    {in_bounds = [false, true]}
    : vector<8x16xf32>, memref<32x64xf32>
  return
}

// CHECK-LABEL:   @no_store_out_of_bounds(
// CHECK:         vector.transfer_write

// -----

func.func @no_store_masked(%arg0: vector<4xf32>,
    %arg1: memref<4xf32>, %arg2: index) {
  %mask = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  vector.transfer_write %arg0, %arg1[%arg2], %mask
    {in_bounds = [true]}
    : vector<4xf32>, memref<4xf32>
  return
}

// CHECK-LABEL: @no_store_masked(
// CHECK:       vector.transfer_write

// -----

func.func @no_store_tensor(%arg0: vector<8x16xf32>,
    %arg1: tensor<32x64xf32>, %arg2: index) -> tensor<32x64xf32> {
  %0 = vector.transfer_write %arg0, %arg1[%arg2, %arg2]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: @no_store_tensor(
// CHECK:       vector.transfer_write

// -----

func.func @no_store_high_dim_vector(%arg0: vector<8x16x32xf32>,
    %arg1: memref<16x32x64xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2, %arg2]
    {in_bounds = [true, true, true]}
    : vector<8x16x32xf32>, memref<16x32x64xf32>
  return
}

// CHECK-LABEL: @no_store_high_dim_vector(
// CHECK:       vector.transfer_write

// -----

func.func @no_store_non_unit_inner_stride(%arg0: vector<8xf32>,
    %arg1: memref<32xf32, strided<[?], offset: ?>>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2]
    {in_bounds = [true]}
    : vector<8xf32>, memref<32xf32, strided<[?], offset: ?>>
  return
}

// CHECK-LABEL: @no_store_non_unit_inner_stride(
// CHECK:       vector.transfer_write

// -----

func.func @no_store_unsupported_map(%arg0: vector<8x16xf32>,
    %arg1: memref<16x32x64xf32>, %arg2: index) {
  vector.transfer_write %arg0, %arg1[%arg2, %arg2, %arg2]
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>,
    in_bounds = [true, true]}
    : vector<8x16xf32>, memref<16x32x64xf32>
  return
}

// CHECK-LABEL: @no_store_unsupported_map(
// CHECK:       vector.transfer_write
