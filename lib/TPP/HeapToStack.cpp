//===- HeapToStack.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert buffers from heap to stack allocation.
struct HeapToStackAllocation : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  HeapToStackAllocation(MLIRContext *context, unsigned maxAllocSizeInBytes,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<memref::AllocOp>(context, benefit),
        maxAllocSizeInBytes(maxAllocSizeInBytes) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    auto type = alloc.getType().dyn_cast<ShapedType>();

    // Ignore dynamically sized buffers as their total size is unknown.
    if (!type.hasStaticShape())
      return rewriter.notifyMatchFailure(alloc,
                                         "Expected statically sized buffer");

    // Check allocation size. Only move small buffers to stack.
    unsigned bitwidth = mlir::DataLayout::closest(alloc).getTypeSizeInBits(
        type.getElementType());
    unsigned size = type.getNumElements() * bitwidth;
    if (size > (maxAllocSizeInBytes * 8))
      return rewriter.notifyMatchFailure(
          alloc, "Buffer exceeds maximum convertion size");

    // Find matching deallocation operation.
    Operation *deallocOp = nullptr;
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        deallocOp = user;
        break;
      }
    }
    if (!deallocOp)
      return rewriter.notifyMatchFailure(
          alloc, "Expected to find matching deallocator");

    // Ensure that the memory allocation and deallocation happens within same
    // scope.
    auto region = alloc->getParentRegion();
    if (region != deallocOp->getParentRegion())
      return rewriter.notifyMatchFailure(
          alloc, "Expected deallocator to be in the same scope");

    SmallVector<Operation *> regionOps;
    SmallVector<Type> resultTypes;
    bool isNewScope = false;
    for (auto &op : region->getOps()) {
      if (&op == deallocOp)
        break;
      if (isNewScope) {
        regionOps.push_back(&op);
        for (Type resType : op.getResultTypes()) {
          resultTypes.push_back(resType);
        }
      }
      if (&op == alloc.getOperation())
        isNewScope = true;
    }

    // Remove the deallocator as stack lifetime is managed automatically.
    rewriter.eraseOp(deallocOp);

    auto loc = alloc.getLoc();
    auto scope = rewriter.create<memref::AllocaScopeOp>(loc, resultTypes);
    Block *scopeBlock = rewriter.createBlock(&scope.getBodyRegion());

    // Replace the original buffer with an equivalent stack allocation.
    auto alloca = rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        alloc, alloc.getMemref().getType(), alloc.getAlignmentAttr());

    SmallVector<Value> results;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(scopeBlock);

      alloca->moveBefore(scopeBlock, scopeBlock->end());

      for (auto *op : regionOps) {
        op->moveBefore(scopeBlock, scopeBlock->end());
        for (Value result : op->getResults()) {
          results.push_back(result);
        }
      }

      rewriter.create<memref::AllocaScopeReturnOp>(loc, results);
    }

    // Swap bench results with loop results.
    for (auto [res, scopeRes] : llvm::zip_equal(results, scope.getResults()))
      res.replaceUsesWithIf(scopeRes, [&](OpOperand &use) {
        return use.getOwner()->getBlock() != scopeBlock;
      });

    SmallVector<int> newResultsPos;
    SmallVector<Type> newResultTypes;
    for (auto [idx, res] : llvm::enumerate(scope.getResults())) {
      llvm::SmallVector<Operation *, 4> users(res.getUsers().begin(),
                                              res.getUsers().end());
      if (users.size() != 0) {
        newResultsPos.push_back(idx);
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPointAfter(scope);
    auto newScope = rewriter.create<memref::AllocaScopeOp>(loc, newResultTypes);
    Block *newScopeBlock = rewriter.createBlock(&newScope.getBodyRegion());

    SmallVector<Value> newResults;
    auto yield = scope.getBodyRegion().front().getTerminator();
    for (auto idx : newResultsPos)
      newResults.push_back(yield->getOperand(idx));
    rewriter.eraseOp(yield);

    rewriter.mergeBlocks(&scope.getBodyRegion().front(),
                         &newScope.getBodyRegion().front(), {});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newScopeBlock);

      rewriter.create<memref::AllocaScopeReturnOp>(loc, newResults);
    }

    auto oldScopeResults = scope.getResults();
    auto newScopeResults = newScope.getResults();
    for (auto [idx, pos] : llvm::enumerate(newResultsPos)) {
      oldScopeResults[pos].replaceUsesWithIf(
          newScopeResults[idx], [&](OpOperand &use) {
            return use.getOwner()->getBlock() != newScopeBlock;
          });
    }

    rewriter.eraseOp(scope);

    return success();
  }

private:
  unsigned maxAllocSizeInBytes;
};

struct HeapToStack : public HeapToStackBase<HeapToStack> {
  HeapToStack() = default;
  HeapToStack(unsigned maxAllocSizeInBytes) {
    this->maxAllocSizeInBytes = maxAllocSizeInBytes;
  }

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<HeapToStackAllocation>(patterns.getContext(),
                                        maxAllocSizeInBytes);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createHeapToStackPass(unsigned maxAllocSizeInBytes) {
  return std::make_unique<HeapToStack>(maxAllocSizeInBytes);
}
