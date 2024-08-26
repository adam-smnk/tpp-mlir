//===- GpuVectorTile.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUVECTORTILE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

static bool isVectorGemm(vector::ContractionOp op) {
  auto *ctx = op.getContext();

  // Check if it is a simple GEMM.
  auto iteratorTypes = op.getIteratorTypes().getValue();
  if (iteratorTypes.size() != 3 ||
      !vector::isParallelIterator(iteratorTypes[0]) ||
      !vector::isParallelIterator(iteratorTypes[1]) ||
      !vector::isReductionIterator(iteratorTypes[2]))
    return false;

  // GEMM variants with transpositions are not supported for now.
  AffineExpr m;
  AffineExpr n;
  AffineExpr k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) { return AffineMap::inferFromExprList(m, ctx); };
  if (maps != infer({{m, k}, {k, n}, {m, n}}))
    return false;

  return true;
}

// Control vector unrolling.
static LogicalResult shouldUnrollOp(Operation *op) {
  // Only process operations within GPU kernel launch.
  if (!op->getParentOfType<gpu::LaunchOp>())
    return failure();

  // Only process simple GEMM contraction.
  auto contraction = dyn_cast<vector::ContractionOp>(op);
  if (contraction && !isVectorGemm(contraction))
    return failure();

  return success();
}

// Get data movement sizes for Intel GPU.
static std::optional<SmallVector<int64_t>> get2DLoadStoreShape(VectorType vec) {
  SmallVector<int64_t> dataTransferShape{vec.getShape()};
  // Can handle at most 2D shape.
  if (dataTransferShape.size() > 2)
    return std::nullopt;
  // Pad with unit size in case of 1D shape.
  if (dataTransferShape.size() == 1)
    dataTransferShape.push_back(1);

  auto elemByteWidth = vec.getElementType().getIntOrFloatBitWidth() / 8;
  // TODO: Fetch actual list of supported load configs.
  // TODO: Take into consideration need for VNNI.
  int64_t maxHeight = 32;
  int64_t maxWidth = 64 / elemByteWidth;

  dataTransferShape[0] = std::min(dataTransferShape[0], maxHeight);
  dataTransferShape[1] = std::min(dataTransferShape[1], maxWidth);

  return dataTransferShape;
}

// Get elementwise SIMD operations sizes for Intel GPU.
static std::optional<SmallVector<int64_t>> getEltwiseShape(VectorType vec) {
  // Can handle at most 2D shape.
  ArrayRef<int64_t> shape = vec.getShape();
  if (shape.size() > 2)
    return std::nullopt;

  // Extract SIMD sized sub-tiles from loaded tiles.
  // TODO: Fetch SIMD sizes from target descriptor.
  int64_t maxSizeSIMD = 256;

  // Thus, split the registers into contiguous smaller slices. The current
  // hardware load restrictions ensure that the loaded tile width will not
  // exceed SIMD size.
  //
  // Take as wide lane as possible first.
  int64_t cols = std::min(shape.back(), maxSizeSIMD);

  SmallVector<int64_t> eltwiseShape;
  if (shape.size() == 2) {
    // In case of 2D shape, add rows if possible to fill up SIMD lane.
    int64_t rows = std::min(shape[0], maxSizeSIMD / cols);
    eltwiseShape.push_back(rows);
  }
  eltwiseShape.push_back(cols);

  return eltwiseShape;
}

// Control vector unrolling shapes.
static std::optional<SmallVector<int64_t>> getVectorOpShape(Operation *op) {
  // TODO: Run a separate analysis pass that assigns and propagates
  //       vector layouts throughout the graph.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = dyn_cast<VectorType>(op->getResult(0).getType()))
      return getEltwiseShape(vecType);
  }
  // TODO: Only unroll DPAS.
  if (isa<vector::ContractionOp>(op))
    return SmallVector<int64_t>{8, 16, 16};
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op))
    return get2DLoadStoreShape(readOp.getVector().getType());
  if (auto writeOp = dyn_cast<vector::TransferReadOp>(op))
    return get2DLoadStoreShape(writeOp.getVector().getType());

  return std::nullopt;
}

// Transfer data from host to a GPU device.
struct GpuVectorTile : public tpp::impl::GpuVectorTileBase<GpuVectorTile> {
  using GpuVectorTileBase::GpuVectorTileBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);

    vector::UnrollVectorOptions unrollOptions;
    unrollOptions.setFilterConstraint(shouldUnrollOp);
    unrollOptions.setNativeShapeFn(getVectorOpShape);
    vector::populateVectorUnrollPatterns(patterns, unrollOptions);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateBubbleVectorBitCastOpPatterns(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};

} // namespace
