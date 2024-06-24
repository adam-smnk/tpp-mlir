//===- CheckDialect.cpp - Perf dialect --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Check/CheckOps.h"

using namespace mlir;
using namespace mlir::check;

//===----------------------------------------------------------------------===//
// Check dialect.
//===----------------------------------------------------------------------===//

void CheckDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/Check/CheckOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TPP/Dialect/Check/CheckOpsTypes.cpp.inc"
      >();
}

#include "TPP/Dialect/Check/CheckOpsDialect.cpp.inc"
