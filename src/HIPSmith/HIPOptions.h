// Options for HIPSmith. Like CGOptions

namespace HIPSmith {

// Encapsulates all the options used throughout the program such that they can
// easily be accessed anywhere.
class HIPOptions {
 public:
  HIPOptions() = delete;

  // Flag meanings
  // output                   - filename of where to output hip code
  // safe_math                - should safe math methods be used
  // small                    - restrict the output to be a smaller program
  // is_emitting_device_code  - are currently writing hip code not cpp code
  // hip_consts               - generate global read only HIP constants
  // hip_shared               - generate local HIP shared memory variables

#define DEFINE_HIPFLAG(name, type) \
 private:                          \
  static type name##_;             \
                                   \
 public:                           \
  static type name();              \
  static void name(type x);

  DEFINE_HIPFLAG(output, const char*)
  DEFINE_HIPFLAG(safe_math, bool)
  DEFINE_HIPFLAG(small, bool)
  DEFINE_HIPFLAG(is_emitting_device_code, bool)
  DEFINE_HIPFLAG(vectors, bool)
  DEFINE_HIPFLAG(hip_consts, bool)
  DEFINE_HIPFLAG(hip_shared, bool)
  DEFINE_HIPFLAG(hip_managed, bool)
  DEFINE_HIPFLAG(hip_device, bool)
  DEFINE_HIPFLAG(hip_builtins, bool)

#undef DEFINE_HIPFLAG

  static void set_default_settings();

  // Automagically sets flags in CGOptions
  static void ResolveCGOptions();
};

}  // namespace HIPSmith
