#include <cstdint>

#include "safe_math_macros.h"

#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif

// Correct definition
FORCE_INLINE void transparent_crc_no_string(uint64_t *crc64_context,
                                            uint64_t val) {
  *crc64_context += val;
}

#define transparent_crc_(A, B, C, D) transparent_crc_no_string(A, B)