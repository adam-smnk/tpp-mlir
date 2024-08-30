// RUN: tpp-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @load_1D_vector(%arg0: memref<32xf32>, %arg1: index) -> vector<8xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1], %cst
    {in_bounds = [true]} : memref<32xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @load_1D_vector(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[ARG0]][%[[ARG1]]]
// CHECK-SAME:    memref<32xf32> -> !xegpu.tensor_desc<8xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_2D_vector(%arg0: memref<32x64xf32>,
    %arg1: index, %arg2: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg2], %cst
    {in_bounds = [true, true]} : memref<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_2D_vector(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: index,
// CHECK-SAME:  %[[ARG2:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[ARG0]][%[[ARG1]], %[[ARG2]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<8x16xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_transposed(%arg0: memref<32x64xf32>,
    %arg1: index, %arg2: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg2], %cst
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_transposed(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: index,
// CHECK-SAME:  %[[ARG2:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[ARG0]][%[[ARG1]], %[[ARG2]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<16x8xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]] <{transpose = array<i64: 1, 0>}>
// CHECK-SAME:    -> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_dynamic_source(%arg0: memref<?x?x?xf32>,
    %arg1: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg1, %arg1], %cst
    {in_bounds = [true, true]} : memref<?x?x?xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_dynamic_source(
// CHECK-SAME:  %[[ARG0:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[DIM_0:.+]] = memref.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:   %[[DIM_1:.+]] = memref.dim %[[ARG0]], %[[C1]]
// CHECK-DAG:   %[[DIM_2:.+]] = memref.dim %[[ARG0]], %[[C2]]
// CHECK:       %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[ARG0]][%[[ARG1]], %[[ARG1]], %[[ARG1]]]
// CHECK-SAME:    [%[[DIM_0]], %[[DIM_1]], %[[DIM_2]]], [%[[DIM_0_STRIDE]], %[[DIM_2]], 1]
// CHECK-SAME:    memref<?x?x?xf32> -> !xegpu.tensor_desc<8x16xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @no_load_out_of_bounds(%arg0: memref<32x64xf32>,
    %arg1: index, %arg2: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg2], %cst
    {in_bounds = [true, false]} : memref<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @no_load_out_of_bounds(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_masked(%arg0 : memref<4xf32>,
    %arg1 : index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %mask = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  %0 = vector.transfer_read %arg0[%arg1], %cst, %mask
    {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @no_load_masked(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_tensor(%arg0: tensor<32x64xf32>,
    %arg1: index, %arg2: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg2], %cst
    {in_bounds = [true, true]} : tensor<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @no_load_tensor(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_high_dim_vector(%arg0: memref<16x32x64xf32>,
    %arg1: index, %arg2: index) -> vector<8x16x32xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg2, %arg1], %cst
    {in_bounds = [true, true, true]} : memref<16x32x64xf32>, vector<8x16x32xf32>
  return %0 : vector<8x16x32xf32>
}

// CHECK-LABEL: @no_load_high_dim_vector(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_non_unit_inner_stride(
    %arg0: memref<32xf32, strided<[?], offset: ?>>,
    %arg1: index) -> vector<8xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1], %cst {in_bounds = [true]}
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @no_load_non_unit_inner_stride(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_unsupported_map(%arg0: memref<16x32x64xf32>,
    %arg1: index) -> vector<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1, %arg1, %arg1], %cst
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>,
    in_bounds = [true, true]} : memref<16x32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @no_load_unsupported_map(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_transpose_unsupported_data_type(%arg0: memref<32x64xf16>,
    %arg1: index, %arg2: index) -> vector<8x16xf16> {
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %arg0[%arg1, %arg2], %cst
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf16>, vector<8x16xf16>
  return %0 : vector<8x16xf16>
}

// CHECK-LABEL: @no_load_transpose_unsupported_data_type(
// CHECK:       vector.transfer_read
