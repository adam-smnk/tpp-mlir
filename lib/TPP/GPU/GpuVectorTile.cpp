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

static std::optional<SmallVector<int64_t>> getVectorOpShape(Operation *op) {
  // return std::nullopt;
  // return SmallVector<int64_t>{8, 8};
  if (isa<arith::AddFOp, arith::SelectOp, arith::CmpFOp>(op))
    return SmallVector<int64_t>{4, 4};
  if (isa<vector::ContractionOp>(op))
    return SmallVector<int64_t>{4, 4, 4};
  // For transfer ops, just propagate the shape coming from
  // InsertStridedSlices/ExtractStridedSlices.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    VectorType dstVec;
    for (Operation *users : readOp->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract)
        return std::nullopt;
      auto vecType = cast<VectorType>(extract.getResult().getType());
      if (dstVec && dstVec != vecType)
        return std::nullopt;
      dstVec = vecType;
    }
    // return SmallVector<int64_t>(dstVec.getShape().begin(),
    //                             dstVec.getShape().end());
    return SmallVector<int64_t>(dstVec.getShape().size(), 8);
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    auto insert =
        writeOp.getVector().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insert)
      return std::nullopt;
    ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
    // return SmallVector<int64_t>(shape.begin(), shape.end());
    return SmallVector<int64_t>(shape.size(), 8);
  }
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
