//===- SplitGenericToTpp.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

bool isAddOp(Operation *bodyOp) {
  return TypeSwitch<Operation *, bool>(bodyOp)
      .Case<arith::AddFOp, arith::AddIOp>([&](auto op) { return true; })
      .Default([&](Operation *) { return false; });
}

bool isMaxOp(Operation *bodyOp) {
  return TypeSwitch<Operation *, bool>(bodyOp)
      .Case<arith::MaxFOp, arith::MaxSIOp, arith::MaxUIOp>(
          [&](auto op) { return true; })
      .Default([&](Operation *) { return false; });
}

bool isTppAddMappable(Operation *op) {
  if (llvm::any_of(op->getResultTypes(),
                   [](Type type) { return !type.isIntOrFloat(); }))
    return false;

  auto genOp = dyn_cast_or_null<linalg::GenericOp>(op->getParentOp());
  if (!genOp)
    return false;

  return isAddOp(op) &&
         tpp::utils::allOperandsHaveSameShapeAndStrides(
             genOp->getOperands().getTypes()) &&
         tpp::utils::allIndexingsAreProjectedPermutation(genOp);
}

bool isTppReluMappable(Operation *op) {
  if (llvm::any_of(op->getResultTypes(),
                   [](Type type) { return !type.isIntOrFloat(); }))
    return false;

  auto genOp = dyn_cast_or_null<linalg::GenericOp>(op->getParentOp());
  if (!genOp)
    return false;

  return tpp::utils::allIndexingsAreProjectedPermutation(genOp) &&
         tpp::utils::isMaxfZeroOp(op);
}

SmallVector<linalg::GenericOp> splitGenericOp(linalg::GenericOp genericOp,
                                              PatternRewriter &rewriter) {
  SmallVector<linalg::GenericOp> splitOps;
  Location loc = genericOp.getLoc();
  Block *body = genericOp.getBody();
  auto *yieldOp = genericOp.getBody()->getTerminator();
  const int numBodyOps = body->getOperations().size();

  for (int i = 0; i < numBodyOps; ++i) {
    Operation *bodyOp = &(*body->begin());
    // llvm::dbgs() << "bodyop: " << *bodyOp << "\n";
    if (isa<linalg::YieldOp>(bodyOp)) {
      // rewriter.eraseOp(bodyOp);
      continue;
    }

    SmallVector<Value> inputs = genericOp.getInputs();
    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    if (!splitOps.empty()) {
      auto prevSplitOp = std::prev(splitOps.end());
      for (auto out : prevSplitOp->getOutputs())
        inputs.push_back(out);
      for (OpOperand *outOperand : prevSplitOp->getDpsInitOperands())
        indexingMaps.push_back(prevSplitOp->getMatchingIndexingMap(outOperand));
    }

    auto peeledGenericOp = rewriter.create<linalg::GenericOp>(
        loc, genericOp.getResultTypes(), inputs, genericOp.getOutputs(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        genericOp.getIteratorTypes(),
        /*doc=*/nullptr,
        /*libraryCall=*/nullptr, [](OpBuilder, Location, ValueRange) {});

    Block *peeledGenericOpBody = peeledGenericOp.getBody();
    peeledGenericOpBody->getOperations().splice(peeledGenericOpBody->begin(),
                                                body->getOperations(), bodyOp);
    Operation *peeledOp = &(*peeledGenericOpBody->begin());
    {
      // Yield all the result of the peeled scalar operation.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(peeledGenericOpBody);
      SmallVector<Value> yieldedVals;
      yieldedVals.append(llvm::to_vector(llvm::map_range(
          peeledOp->getResults(), [](OpResult opr) -> Value { return opr; })));
      rewriter.create<linalg::YieldOp>(loc, yieldedVals);
    }

    for (const auto &inputBlockArg :
         llvm::enumerate(genericOp.getBody()->getArguments())) {
      Value peeledOpReplacementArg =
          peeledGenericOpBody->getArgument(inputBlockArg.index());
      inputBlockArg.value().replaceUsesWithIf(
          peeledOpReplacementArg, [&](OpOperand &use) {
            return use.getOwner()->getBlock() == peeledGenericOpBody;
          });
    }

    if (!splitOps.empty()) {
      auto prevSplitOp = std::prev(splitOps.end());
      unsigned origNumInputs = genericOp.getNumDpsInputs();
      Operation *prevBodyOp = &(*prevSplitOp->getBody()->begin());
      unsigned prevOpNumResults = prevBodyOp->getNumResults();
      SmallVector<Value> scalarReplacements;
      scalarReplacements.reserve(prevOpNumResults);
      for (auto num : llvm::seq<unsigned>(0, prevOpNumResults))
        scalarReplacements.push_back(
            peeledGenericOpBody->getArgument(num + origNumInputs));
      bool allUsesReplaced = false;
      rewriter.replaceOpWithinBlock(prevBodyOp, scalarReplacements,
                                    peeledGenericOpBody, &allUsesReplaced);
      assert(!allUsesReplaced &&
             "peeled scalar operation is erased when it wasnt expected to be");
    }

    splitOps.push_back(peeledGenericOp);
  }

  return splitOps;
}

struct GenericOpTppFission : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // For now only match on operations where the iterator types are all
    // parallel
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
      return rewriter.notifyMatchFailure(
          genericOp,
          "unhandled split of generic with non-parallel iterator types");
    }

    auto &region = genericOp.getRegion();
    if (!region.hasOneBlock())
      return rewriter.notifyMatchFailure(
          genericOp, "Expect generic op with a single block");

    // // Check general TPP-specific mapping conditions.
    // if (!tpp::utils::hasMappingToTppConditions(genericOp))
    //   return failure();

    // Check if all individual operations within the generic can be mapped
    // to TPP operations.
    unsigned int numTppMappableOps = 0;
    for (auto &op : region.front()) {
      // Yield terminator is assumed to be mappable by default for
      // simplicity.
      if (isa<linalg::YieldOp>(op)) {
        ++numTppMappableOps;
        continue;
      }
      if (isTppAddMappable(&op)) {
        // llvm::dbgs() << op << " is add mappable\n";
        ++numTppMappableOps;
        continue;
      }
      if (isTppReluMappable(&op)) {
        // llvm::dbgs() << op << " is relu mappable\n";
        ++numTppMappableOps;
        continue;
      }
    }

    // Avoid partial splits - only decompose generics that are fully
    // TPP-mappable.
    if (numTppMappableOps != region.front().getOperations().size())
      return rewriter.notifyMatchFailure(
          genericOp, "Expect all generic body ops to be TPP mappable");

    RewritePatternSet patterns(getContext());
    linalgx::populateDecomposeLinalgOpsPattern(patterns, true);
    if (failed(applyOpPatternsAndFold(ArrayRef{genericOp.getOperation()},
                                      std::move(patterns)))) {
      return failure();
    }

    // auto splitOps = splitGenericOp(genericOp, rewriter);
    // genericOp.getOperation()->getParentOp()->dump();

    // rewriter.eraseOp(genericOp);

    return success();
  }
};

void populateSplitGenericPatterns(RewritePatternSet &patterns) {
  patterns.add<GenericOpTppFission>(patterns.getContext());
}

struct SplitGenericToTpp : public SplitGenericToTppBase<SplitGenericToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSplitGenericPatterns(patterns);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createSplitGenericToTpp() {
  return std::make_unique<SplitGenericToTpp>();
}
