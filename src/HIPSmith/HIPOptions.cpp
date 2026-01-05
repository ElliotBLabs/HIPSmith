#include "HIPSmith/HIPOptions.h"

#include <iostream>

#include "CGOptions.h"

namespace HIPSmith {

// Eww macro, used here just to make the flags easier to write.
#define DEFINE_HIPFLAG(name, type, init)      \
  type HIPOptions::name##_ = init;            \
  type HIPOptions::name() { return name##_; } \
  void HIPOptions::name(type x) { name##_ = x; }
DEFINE_HIPFLAG(output, const char*, "HIPProg.cc")
DEFINE_HIPFLAG(safe_math, bool, true)
DEFINE_HIPFLAG(small, bool, false)
#undef DEFINE_CLFLAG

void HIPOptions::set_default_settings() {
  output_ = "HIPProg.cc";
  safe_math_ = true;
  small_ = false;
}

void HIPOptions::ResolveCGOptions() {
  // General settings for normal HIP programs.
  // No static in OpenCL.
  CGOptions::force_globals_static(false);

  // Maybe enable in future. Has a different syntax.
  CGOptions::packed_struct(false);
  // No printf in OpenCL.
  CGOptions::hash_value_printf(false);
  // The way we currently handle globals means we need to disable consts.
  CGOptions::consts(false);
  // Volatiles were playing up so turned off
  CGOptions::volatiles(false);
  // cpp options
  CGOptions::lang_cpp(true);
  CGOptions::cpp11(true);

  // Setting for small programs.
  if (small_) {
    // Limit number of functions to no more than 3.
    CGOptions::max_funcs(3);
    CGOptions::max_blk_depth(3);
    CGOptions::max_expr_depth(5);
    CGOptions::max_block_size(20);
    CGOptions::max_array_dimensions(3);
    CGOptions::max_array_length_per_dimension(5);
    CGOptions::max_array_length(7);
  }
}

}  // namespace HIPSmith
