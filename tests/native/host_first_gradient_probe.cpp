#include "integrals/s_integrals.hpp"

extern "C" int host_packaged_gradient(const unsigned* angular, unsigned component,
                                      const double* record, double* output) {
  using namespace generativeqc::integrals;
  switch (packaged_eri_primitive_gradient(angular, component, record, output)) {
    case PackagedEriGradientStatus::Success:
      return 0;
    case PackagedEriGradientStatus::Unavailable:
      return 1;
    case PackagedEriGradientStatus::EvaluationFailed:
      return -1;
  }
  return -2;
}

extern "C" bool host_packaged_gradient_available() {
  return generativeqc::integrals::has_packaged_cpu_eri_shell_derivatives();
}
