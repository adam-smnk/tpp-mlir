// RUN: tpp-opt %s -split-input-file -verify-diagnostics

func.func @perf_timer_multi_stop() {
  %t = perf.start_timer : index
  // expected-error @below {{'perf.stop_timer' op timer stopped multiple times}}
  %del = perf.stop_timer(%t : index) : f64
  %del1 = perf.stop_timer(%t : index) : f64
  return
}

// -----

func.func @perf_invalid_timer(%n: index) {
  // expected-error @below {{'perf.stop_timer' op invalid timer input}}
  %del = perf.stop_timer(%n : index) : f64
  return
}

// -----

func.func @perf_invalid_timer_1() {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'perf.stop_timer' op invalid timer input}}
  %del = perf.stop_timer(%c0 : index) : f64
  return
}
