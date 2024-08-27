// RUN: tpp-opt %s -gpu-vector-tile -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @vector_tile_contract(%arg0: vector<32x16xf32>, %arg1: vector<16x32xf32>,
    %arg2: vector<32x32xf32>, %arg3: memref<32x32xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%tx, %ty, %tz) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    %0 = vector.contract {indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %arg0, %arg1, %arg2
      : vector<32x16xf32>, vector<16x32xf32> into vector<32x32xf32>
    vector.transfer_write %0, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vector_tile_contract(
// CHECK:         gpu.launch
// CHECK-COUNT-8:   vector.contract{{.*}}: vector<8x16xf32>, vector<16x16xf32> into vector<8x16xf32>

// -----

func.func @vector_tile_eltwise_binary(%arg0: vector<12x64xf32>, %arg1: vector<12x64xf32>, %arg2: memref<12x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%tx, %ty, %tz) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    %0 = arith.subf %arg0, %arg1 : vector<12x64xf32>
    vector.transfer_write %0, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<12x64xf32>, memref<12x64xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vector_tile_eltwise_binary(
// CHECK:         gpu.launch
// CHECK-COUNT-3:   arith.subf{{.*}}: vector<4x64xf32>

// -----

func.func @vector_tile_eltwise_unary(%arg0: vector<12x64xf32>, %arg1: memref<12x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%tx, %ty, %tz) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %0 = math.absf %arg0 : vector<12x64xf32>
    vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<12x64xf32>, memref<12x64xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vector_tile_eltwise_unary(
// CHECK:         gpu.launch
// CHECK-COUNT-3:   math.absf{{.*}}: vector<4x64xf32>

// -----

func.func @vector_tile_read_write_f32(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%tx, %ty, %tz) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst
      {in_bounds = [true, true]} : memref<64x64xf32>, vector<64x64xf32>
    vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, memref<64x64xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vector_tile_read_write_f32(
// CHECK:         gpu.launch
// CHECK-COUNT-8:   vector.transfer_read{{.*}}: memref<64x64xf32>, vector<32x16xf32>
// CHECK-COUNT-8:   vector.transfer_write{{.*}}: vector<32x16xf32>, memref<64x64xf32>

// -----

func.func @vector_tile_read_write_f16(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%tx, %ty, %tz) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst
      {in_bounds = [true, true]} : memref<64x64xf16>, vector<64x64xf16>
    vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf16>, memref<64x64xf16>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vector_tile_read_write_f16(
// CHECK:         gpu.launch
// CHECK-COUNT-4:   vector.transfer_read{{.*}}: memref<64x64xf16>, vector<32x32xf16>
// CHECK-COUNT-4:   vector.transfer_write{{.*}}: vector<32x32xf16>, memref<64x64xf16>
