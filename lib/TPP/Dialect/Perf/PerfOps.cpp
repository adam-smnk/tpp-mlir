//===- PerfOps.cpp - Perf dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Perf/PerfOps.cpp.inc"

using namespace mlir;
using namespace mlir::perf;

void BenchOp::build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTypes, ValueRange operands,
                    BodyBuilderFn bodyBuilder) {
  result.addOperands(operands);

  // First result is always collection of time deltas, and anything else being
  // yielded from the body
  MLIRContext *ctx = result.getContext();
  result.addTypes(
      {MemRefType::get({ShapedType::kDynamic}, FloatType::getF64(ctx))});
  for (Type type : resultTypes)
    result.addTypes(type);

  // Add a body region
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return
  Block &bodyBlock = bodyRegion->front();

  if (resultTypes.empty() && !bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<perf::YieldOp>(result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

void BenchOp::print(OpAsmPrinter &p) {
  // Print input argument
  // (%n)
  p << " (" << getNumIters() << ")";

  // Print the op body
  // { ... }
  p.printRegion(getBodyRegion());

  // Print attr-dict
  p.printOptionalAttrDict((*this)->getAttrs());

  // Print return types - measured deltas and yielded values if present
  // -> memref<>, ...
  p.printOptionalArrowTypeList(getResultTypes());
}

ParseResult BenchOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Parse input arguments
  SmallVector<OpAsmParser::UnresolvedOperand, 1> rawOperands;
  if (parser.parseOperandList(rawOperands) || rawOperands.size() != 1 ||
      parser.resolveOperand(rawOperands[0], IndexType::get(ctx),
                            result.operands))
    // return failure();
    void();

  // Parse the op body
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    // return failure();
    void();

  // Parse attr-dict
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    // return failure();
    void();
  result.addAttributes(attrs);

  // Parse the types of results returned from the op
  SmallVector<Type, 4> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes) ||
      parser.addTypeToList(
          MemRefType::get({ShapedType::kDynamic}, FloatType::getF64(ctx)),
          result.types) ||
      parser.addTypesToList(resultTypes, result.types))
    // return failure();
    void();

  return success();
}

LogicalResult StopTimerOp::verify() {
  auto timerSrc = getTimer().getDefiningOp();
  if (!timerSrc || !isa<StartTimerOp>(timerSrc))
    return emitOpError("invalid timer input");

  int numStopTimers = 0;
  for (auto user : timerSrc->getUsers()) {
    if (isa<StopTimerOp>(*user))
      ++numStopTimers;
  }
  if (numStopTimers != 1)
    return emitOpError("timer stopped multiple times");

  return success();
}
