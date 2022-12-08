//===- ConvertPerfToLoops.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::perf;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct ConvertBenchToLoops : public OpRewritePattern<perf::BenchOp> {
  using OpRewritePattern<perf::BenchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::BenchOp benchOp,
                                PatternRewriter &rewriter) const override {
    auto loc = benchOp.getLoc();
    auto benchYield = benchOp.getRegion().front().getTerminator();

    // TODO: add support for yielding values - requires either iter_args
    //       approach like in scf.for or automatic memory allocation
    //       outside of the benchmarking loop scope
    // TODO: replace perf.bench yielded values with loop yielded values
    if (benchYield->getNumOperands() != 0)
      return benchOp.emitOpError(
          "lowering with yielded values is not supported");

    auto numIters = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), benchOp.getNumIters());

    // Allocate memory to store iteration results
    auto allocType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    auto resultMem =
        rewriter.create<memref::AllocOp>(loc, allocType, ValueRange{numIters});
    benchOp.getDeltas().replaceAllUsesWith(resultMem.getMemref());

    // Create benchmark loop up to perf.bench numIters
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, zero, numIters, one);
    auto loopYield = loop.getRegion().front().getTerminator();

    // Move perf.bench region inside the loop
    rewriter.mergeBlockBefore(&benchOp.getRegion().front(), loopYield);

    // Wrap the benchmark kernel in timer calls
    rewriter.setInsertionPointToStart(loop.getBody());
    auto timer =
        rewriter.create<perf::StartTimerOp>(loc, rewriter.getI64Type());
    rewriter.setInsertionPoint(loopYield);
    auto delta = rewriter.create<perf::StopTimerOp>(loc, rewriter.getF64Type(),
                                                    timer.getTimer());

    // Move all perf.do_not_opt ops after the timer to prevent influencing
    // measurement
    for (auto &op : loop.getRegion().getOps()) {
      if (isa<perf::DoNotOptOp>(op))
        op.moveAfter(delta.getOperation());
    }

    // Store measured time delta
    rewriter.create<memref::StoreOp>(loc, delta.getDelta(), resultMem,
                                     loop.getInductionVar());

    // Forward perf.yield to the scf.yield
    rewriter.replaceOpWithNewOp<scf::YieldOp>(loopYield,
                                              benchYield->getResults());
    rewriter.eraseOp(benchYield);

    rewriter.eraseOp(benchOp);
    return success();
  }
};

void populatePerfToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBenchToLoops>(patterns.getContext());
}

struct ConvertPerfToLoops : public ConvertPerfToLoopsBase<ConvertPerfToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertPerfToLoopsPass() {
  return std::make_unique<ConvertPerfToLoops>();
}
