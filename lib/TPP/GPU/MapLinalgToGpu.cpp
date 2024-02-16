//===- MapLinalgToGpu.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_MAPLINALGTOGPU
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Creates an outermost parallel loop wrapper around an operation to represent
// number of GPU blocks.
// If there is already a parallel loop present, no operation is created and
// a nullopt is returned instead.
static std::optional<scf::ParallelOp>
createGpuBlocksWrapper(Operation *op, ArrayRef<int64_t> blockDims,
                       IRRewriter &rewriter) {
  assert(blockDims.size() <= 3 && "Too many GPU blocks dimensions");

  auto loc = op->getLoc();

  auto *parentOp = op->getParentOp();
  if (isa<scf::ParallelOp>(parentOp))
    return std::nullopt;

  OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointAfter(op);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> gpuBlocks;
  SmallVector<Value> lbs;
  SmallVector<Value> steps;
  for (auto blockDim : blockDims) {
    auto blockSize = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);
    gpuBlocks.push_back(blockSize);
    // Add matching number of lbs and steps.
    lbs.push_back(zero);
    steps.push_back(one);
  }

  return rewriter.create<scf::ParallelOp>(loc, lbs, gpuBlocks, steps);
}

static LogicalResult wrapLinalgInParallelLoop(linalg::LinalgOp linalgOp,
                                              IRRewriter &rewriter) {
  if (!linalgOp.hasPureBufferSemantics()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect linalg on memrefs for GPU mapping");
  }

  auto parallelOp = linalgOp->getParentOfType<scf::ParallelOp>();
  if (parallelOp) {
    return rewriter.notifyMatchFailure(
        linalgOp,
        "Operation must not be already nested within a parallel loop");
  }

  // Create a simple parallel loop to represent a GPU grid with
  // only a single block.
  // This will allow for naive mapping to GPU kernel without any
  // workload parallelization.
  auto wrapperLoop = createGpuBlocksWrapper(linalgOp, {1}, rewriter);
  if (!wrapperLoop)
    return failure();

  // Move the operation into the parallel loop.
  auto &loopBody = wrapperLoop->getRegion().front();
  rewriter.moveOpBefore(linalgOp, &loopBody, loopBody.begin());

  return success();
}

// Map and lower operations to generic GPU ops.
struct MapLinalgToGpu : public tpp::impl::MapLinalgToGpuBase<MapLinalgToGpu> {
  using MapLinalgToGpuBase::MapLinalgToGpuBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](linalg::LinalgOp linalgOp) {
      (void)wrapLinalgInParallelLoop(linalgOp, rewriter);
    });
  }
};

} // namespace
