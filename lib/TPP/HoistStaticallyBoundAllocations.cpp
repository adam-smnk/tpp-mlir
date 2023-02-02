// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "tpp-hoist-statically-bound-allocations"

namespace {

struct HoistStaticallyBoundAllocationsPass
    : HoistStaticallyBoundAllocationsBase<HoistStaticallyBoundAllocationsPass> {
  void runOnOperation() override;
};

std::optional<Value>
hoistStaticallyBoundAllocations(func::FuncOp funcOp, OpBuilder &builder,
                                Location loc, MemRefType allocaType,
                                ValueRange dynamicSizes,
                                std::optional<uint64_t> alignment) {
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;
  // For static case just create a new allocation in the entry block of the
  // same size. No need to insert a subview.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Value allocation =
        builder.create<memref::AllocaOp>(loc, allocaType, alignmentAttr);
    return allocation;
  }

  /// For the dynamic but bounded case, insert an allocation
  /// of the shape of the bounds, and a subview of the
  /// required size to be used as a replacement.
  SmallVector<int64_t> staticShape;
  SmallVector<OpFoldResult> subviewSizes;
  staticShape.reserve(allocaType.getRank());
  subviewSizes.reserve(allocaType.getRank());

  int index = 0;
  for (auto dimSize : allocaType.getShape()) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticShape.push_back(dimSize);
      subviewSizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dynamicSize = dynamicSizes[index++];
    auto ub = linalg::getConstantUpperBoundForIndex(dynamicSize);
    if (failed(ub)) {
      return std::nullopt;
    }
    staticShape.push_back(ub.value());
    subviewSizes.push_back(dynamicSize);
  }
  SmallVector<OpFoldResult> offsets(allocaType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocaType.getRank(),
                                    builder.getIndexAttr(1));

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto allocationType =
        MemRefType::get(staticShape, allocaType.getElementType());
    allocation =
        builder.create<memref::AllocaOp>(loc, allocationType, alignmentAttr);
  }

  Value subviewOp = builder.create<memref::SubViewOp>(loc, allocation, offsets,
                                                      subviewSizes, strides);
  return subviewOp;
}
std::optional<Value>
hoistStaticallyBoundAllocations(func::FuncOp funcOp, OpBuilder &builder,
                                memref::AllocaOp allocaOp) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(allocaOp);
  return hoistStaticallyBoundAllocations(
      funcOp, builder, allocaOp.getLoc(), allocaOp.getType(),
      allocaOp.getDynamicSizes(), allocaOp.getAlignment());
}

/// Some uses of a `memref.alloca` can be replaced with a `memref.subview`
/// easily. Other uses (like a use in a `scf.yield` or `func.return`) are
/// non-trivial because of compatibility between types of different SSA values.
static bool isUseReplacableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  return isa<linalg::LinalgOp, memref::StoreOp, memref::SubViewOp>(user);
}

void HoistStaticallyBoundAllocationsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<memref::AllocaOp> allocaOps;

  // Collect all allocas that are hoistable.
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    if (allocaOp->getBlock() == &funcOp.getBody().front())
      return;
    if (allocaOp.getDynamicSizes().empty()) {
      allocaOps.push_back(allocaOp);
      return;
    }
    if (llvm::all_of(allocaOp->getUses(), [](OpOperand &use) {
          return isUseReplacableWithSubview(use);
        })) {
      allocaOps.push_back(allocaOp);
      return;
    }
  });

  // Hoist the allocas and replace all uses.
  OpBuilder builder(&getContext());
  for (auto allocaOp : allocaOps) {
    LLVM_DEBUG({
      llvm::dbgs() << "Alloca Op : ";
      allocaOp->dump();
      int numUses = std::distance(allocaOp.getResult().use_begin(),
                                  allocaOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement =
        hoistStaticallyBoundAllocations(funcOp, builder, allocaOp);
    if (!replacement)
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    allocaOp.getResult().replaceAllUsesWith(replacementVal);
    allocaOp->erase();
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createHoistStaticallyBoundAllocationsPass() {
  return std::make_unique<HoistStaticallyBoundAllocationsPass>();
}
