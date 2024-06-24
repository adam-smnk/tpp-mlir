//===- CheckDialect.td - Perf dialect ---------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_CHECK_DIALECT
#define TPP_CHECK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Check dialect definition.
//===----------------------------------------------------------------------===//

def Check_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::check";
  let summary = [{
    A dialect implementing test assertions.
  }];
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Check_Op<string mnemonic, list<Trait> traits = []> :
        Op<Check_Dialect, mnemonic, traits>;

#endif // TPP_CHECK_DIALECT
