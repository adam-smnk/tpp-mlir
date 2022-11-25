//===- ConvertPerfToFunc.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::perf;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct ConvertStartTimerOp : public OpRewritePattern<perf::StartTimerOp> {
  using OpRewritePattern<perf::StartTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StartTimerOp startTimerOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

struct ConvertStopTimerOp : public OpRewritePattern<perf::StopTimerOp> {
  using OpRewritePattern<perf::StopTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StopTimerOp stopTimerOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

struct ConvertMeanOp : public OpRewritePattern<perf::MeanOp> {
  using OpRewritePattern<perf::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::MeanOp meanOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

struct ConvertStdevOp : public OpRewritePattern<perf::StdevOp> {
  using OpRewritePattern<perf::StdevOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StdevOp stdevOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

void populatePerfToFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertStartTimerOp, ConvertStopTimerOp, ConvertMeanOp,
               ConvertStdevOp>(patterns.getContext());
}

struct ConvertPerfToFunc : public ConvertPerfToFuncBase<ConvertPerfToFunc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToFuncPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertPerfToFuncPass() {
  return std::make_unique<ConvertPerfToFunc>();
}
