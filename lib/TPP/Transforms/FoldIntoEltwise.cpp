//===- FoldIntoEltwise.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_FOLDINTOELTWISE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Create affine map between a producer operand and a consumer op indexing.
// Assumes that the provided maps are composable.
static AffineMap
reindexProducerOperandIntoConsumer(AffineMap producerOperandMap,
                                   AffineMap producerResultMap,
                                   AffineMap consumerMap) {
  // Index producer result dimensions to its loops.
  AffineMap invProducerResultMap = inversePermutation(producerResultMap);
  // Index producer operand with respect to the producer result dimensions.
  AffineMap operandToResultMap =
      producerOperandMap.compose(invProducerResultMap);
  // Remap producer operand into consumer indexing.
  return operandToResultMap.compose(consumerMap);
}

// Fold linalg unary producer op into a linalg elementwise operation.
struct UnaryProducerIntoEltwise
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(linalgOp))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "not an elementwise operation");

    if (!linalgOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(linalgOp, "expects tensor semantics");

    auto isFoldableProducer = [](Operation *op) {
      return isa<linalg::BroadcastOp, linalg::FillOp>(op);
    };

    // Look for foldable producers within inputs.
    // Reshaping output might be less beneficial and it is not considered now.
    if (llvm::none_of(linalgOp.getDpsInputs(), [&](Value input) {
          auto op = input.getDefiningOp();
          return op && isFoldableProducer(op);
        }))
      return rewriter.notifyMatchFailure(linalgOp, "no foldable producers");

    SmallVector<Value> inputs = linalgOp.getDpsInputs();
    ValueRange outputs = linalgOp.getDpsInits();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iterators =
        linalgOp.getIteratorTypesArray();
    SmallVector<Type> resultTypes = TypeRange(ValueRange{outputs});

    for (auto [idx, input] : llvm::enumerate(linalgOp.getDpsInputs())) {
      auto producerOp = input.getDefiningOp<linalg::LinalgOp>();
      if (!producerOp || !isFoldableProducer(producerOp))
        continue;

      // Update indexing maps.
      // The producer input operand can be captured by indexing maps alone w.r.t
      // producer input and consumer iteration domain.
      indexingMaps[idx] = reindexProducerOperandIntoConsumer(
          producerOp.getMatchingIndexingMap(producerOp.getDpsInputOperand(0)),
          producerOp.getMatchingIndexingMap(producerOp.getDpsInitOperand(0)),
          indexingMaps[idx]);
      // Use the producer input directly instead of the producer result.
      inputs[idx] = producerOp.getDpsInputs()[0];
    }

    // All Linalg ops have a region attached that can be inlined.
    assert(linalgOp->getNumRegions() == 1 &&
           "expect op to have one region attached");
    // Replace the original op with a generic with broadcast folded in.
    auto genericOp = rewriter.create<linalg::GenericOp>(
        linalgOp.getLoc(), resultTypes, inputs, outputs, indexingMaps,
        iterators);
    rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                genericOp.getRegion().begin());
    rewriter.replaceOp(linalgOp, genericOp->getResults());

    return success();
  }
};

struct FoldIntoEltwise : tpp::impl::FoldIntoEltwiseBase<FoldIntoEltwise> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<UnaryProducerIntoEltwise>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
