// Options for HIPSmith. Like CGOptions

namespace HIPSmith {

// Encapsulates all the options used throughout the program such that they can
// easily be accessed anywhere.
class HIPOptions {
 public:
  HIPOptions() = delete;

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
#undef DEFINE_HIPFLAG

  static void set_default_settings();

  // Automagically sets flags in CGOptions
  static void ResolveCGOptions();
};

}  // namespace HIPSmith
