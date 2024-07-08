//===- GpuVectorTile.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
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

static LogicalResult shouldUnrollOp(Operation *op) {
  // Only process operations within GPU kernel launch.
  if (!op->getParentOfType<gpu::LaunchOp>())
    return failure();

  return success();
}

static std::optional<SmallVector<int64_t>> getVectorOpShape(Operation *op) {
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

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
