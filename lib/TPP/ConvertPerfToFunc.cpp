//===- ConvertPerfToFunc.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::perf;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

static LogicalResult buildPerfFuncCall(Location loc, std::string funcName,
                                       Operation *op,
                                       PatternRewriter &rewriter) {
  if (op->getNumResults() > 1)
    return op->emitError(
               "expected operation to have 0 or 1 result, but provided ")
           << op->getNumResults();

  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto libFnType =
      rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(rewriter.getContext()));
    funcOp.setPrivate();
  }

  auto funcCall = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), libFnType.getResults(), op->getOperands());
  op->replaceAllUsesWith(funcCall.getResults());

  return success();
}

struct ConvertStartTimerOp : public OpRewritePattern<perf::StartTimerOp> {
  using OpRewritePattern<perf::StartTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StartTimerOp startTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(startTimerOp.getLoc(), "timer_start",
                                 startTimerOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(startTimerOp);
    return res;
  }
};

struct ConvertStopTimerOp : public OpRewritePattern<perf::StopTimerOp> {
  using OpRewritePattern<perf::StopTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StopTimerOp stopTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(stopTimerOp.getLoc(), "timer_stop",
                                 stopTimerOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(stopTimerOp);
    return res;
  }
};

struct ConvertMeanOp : public OpRewritePattern<perf::MeanOp> {
  using OpRewritePattern<perf::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::MeanOp meanOp,
                                PatternRewriter &rewriter) const override {
    auto res =
        buildPerfFuncCall(meanOp.getLoc(), "timer_average", meanOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(meanOp);
    return res;
  }
};

struct ConvertStdevOp : public OpRewritePattern<perf::StdevOp> {
  using OpRewritePattern<perf::StdevOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StdevOp stdevOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(stdevOp.getLoc(), "timer_deviation", stdevOp,
                                 rewriter);
    if (succeeded(res))
      rewriter.eraseOp(stdevOp);
    return res;
  }
};

static LogicalResult buildDoNotOptCall(std::string funcName,
                                       perf::DoNotOptOp &doNotOptOp,
                                       PatternRewriter &rewriter) {
  auto loc = doNotOptOp.getLoc();
  auto ctx = doNotOptOp.getContext();

  auto ptrType = emitc::PointerType::get(ctx, doNotOptOp.getInput().getType());
  auto inputPtr =
      rewriter.create<emitc::ApplyOp>(loc, ptrType, "&", doNotOptOp.getInput());
  auto opaquePtr = rewriter.create<emitc::CastOp>(
      loc, emitc::OpaqueType::get(ctx, "void*"), inputPtr);
  rewriter.create<emitc::CallOp>(
      loc, TypeRange(), funcName, rewriter.getArrayAttr({}),
      rewriter.getArrayAttr({}), opaquePtr.getResult());

  return success();
}

struct ConvertDoNotOptOp : public OpRewritePattern<perf::DoNotOptOp> {
  using OpRewritePattern<perf::DoNotOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::DoNotOptOp doNotOptOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(doNotOptOp.getLoc(), "perf_do_not_opt",
                                 doNotOptOp, rewriter);
    // auto res = buildDoNotOptCall("perf_do_not_opt", doNotOptOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(doNotOptOp);
    return res;
  }
};

void populatePerfToFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertStartTimerOp, ConvertStopTimerOp, ConvertMeanOp,
               ConvertStdevOp, ConvertDoNotOptOp>(patterns.getContext());
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
