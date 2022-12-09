// RUN: tpp-opt %s -convert-perf-to-func -split-input-file | FileCheck %s

// CHECK-DAG: func.func private @timer_start() -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-LABEL: @perf_start_timer
func.func @perf_start_timer() {
  // CHECK: call @timer_start()
  %t = perf.start_timer : i64
  return
}

// -----

// CHECK-DAG: func.func private @timer_start() -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @timer_stop({{.*}}) -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-LABEL: @perf_stop_timer
func.func @perf_stop_timer() {
  // CHECK: %[[timer:.*]] = call @timer_start()
  %t = perf.start_timer : i64
  // CHECK: call @timer_stop(%[[timer]])
  %delta = perf.stop_timer(%t : i64) : f64
  return
}

// -----

// CHECK-DAG: func.func private @timer_average(memref<*xf64>) -> f64 attributes {llvm.emit_c_interface}
// CHECK-LABEL: @perf_mean
func.func @perf_mean(%arg0: memref<?xf64>) {
  // CHECK: call @timer_average({{.*}})
  %mean = perf.mean(%arg0 : memref<?xf64>) : f64
  return
}

// -----

// CHECK-DAG: func.func private @timer_deviation(memref<*xf64>, f64) -> f64 attributes {llvm.emit_c_interface}
// CHECK-LABEL: @perf_stdev
func.func @perf_stdev(%arg0: memref<?xf64>, %mean: f64) {
  // CHECK: call @timer_deviation({{.*}})
  %stdev = perf.stdev(%arg0 : memref<?xf64>, %mean : f64) : f64
  return
}

// -----

// CHECK-DAG: func.func private @perf_do_not_opt_memref(memref<*xf64>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_do_not_opt_tensor(tensor<?xf64>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_do_not_opt_i32(i32) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_do_not_opt_f64(f64) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_do_not_opt_index(index) attributes {llvm.emit_c_interface}
// CHECK-LABEL: @perf_disable_opt
func.func @perf_disable_opt(%arg0: memref<?xf64>, %arg1: tensor<?xf64>, %arg2: i32, %arg3: f64, %arg4: index) {
  // CHECK: call @perf_do_not_opt_memref({{.*}})
  perf.do_not_opt(%arg0) : memref<?xf64>
  // CHECK: call @perf_do_not_opt_tensor({{.*}})
  perf.do_not_opt(%arg1) : tensor<?xf64>
  // CHECK: call @perf_do_not_opt_i32({{.*}})
  perf.do_not_opt(%arg2) : i32
  // CHECK: call @perf_do_not_opt_f64({{.*}})
  perf.do_not_opt(%arg3) : f64
  // CHECK: call @perf_do_not_opt_index({{.*}})
  perf.do_not_opt(%arg4) : index

  return
}
