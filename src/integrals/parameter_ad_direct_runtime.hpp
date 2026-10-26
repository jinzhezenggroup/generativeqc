// CPU storage-independent consumers of the canonical integral/normalization
// schedule. Generated method code supplies every stationary and J/K weight.
namespace integral_ad {
template <class Scalar>
struct DirectConsumer {
  const Scalar* state;
  Scalar* output;
  const double* da;
  const double* db;
  std::size_t size() const { return da ? 3 * n2 : 2 * n2 + 1; }
  bool accept(std::size_t index) const {
    if (state) return true;
    const bool pair = index >= 2 * n2 && index < value_count - 1;
    return da ? pair : !pair;
  }
  Scalar weight(std::size_t index) const { return streamed_weight(index, state); }
  void add(std::size_t index, Scalar value) const {
    if (da)
      streamed_contract(index, value, da, db, output);
    else
      output[index == value_count - 1 ? 2 * n2 : index] += value;
  }
};

template <class Scalar>
static int direct_execute(const Scalar* coordinates, const Scalar* alpha, const Scalar* raw,
                          const Scalar* state, Scalar* output, const double* da = nullptr,
                          const double* db = nullptr) {
  std::vector<Scalar> scratch(scratch_count);
  return execute_consumer(coordinates, alpha, raw, state, output, scratch.data(), 0, 1,
                          DirectConsumer<Scalar>{state, output, da, db});
}
}  // namespace integral_ad

extern "C" int integral_ad_direct_execute(const double* coordinates, const double* alpha,
                                          const double* raw, const double* state, double* output) {
  try {
    return integral_ad::direct_execute(coordinates, alpha, raw, state, output);
  } catch (...) {
    return 2;
  }
}

#ifdef GQC_INTEGRAL_AD_SECOND
static int integral_ad_direct_motion(const double* coordinates, const double* alpha,
                                     const double* raw, const double* direction,
                                     const double* state, const double* state_direction,
                                     double* output, const double* da = nullptr,
                                     const double* db = nullptr) {
  using namespace integral_ad;
  try {
    std::vector<Directional> parameters(parameter_count), weights(state ? 4 * n2 : 0);
    for (int i = 0; i < 3 * nat; ++i) parameters[i] = {coordinates[i], direction[i]};
    for (int i = 0; i < nprim; ++i) {
      parameters[3 * nat + i] = {alpha[i], direction[3 * nat + i]};
      parameters[3 * nat + nprim + i] = {raw[i], direction[3 * nat + nprim + i]};
    }
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = {state[i], state_direction ? state_direction[i] : 0};
    const auto size = state ? parameter_count : (da ? 3 * n2 : 2 * n2 + 1);
    std::vector<Directional> result(size);
    const int status = direct_execute(parameters.data(), parameters.data() + 3 * nat,
                                      parameters.data() + 3 * nat + nprim,
                                      state ? weights.data() : nullptr, result.data(), da, db);
    if (status) return status;
    for (std::size_t i = 0; i < size; ++i) output[i] = result[i].tangent;
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int integral_ad_direct_directional(const double* coordinates, const double* alpha,
                                              const double* raw, const double* direction,
                                              const double* state, const double* state_direction,
                                              double* output) {
  return integral_ad_direct_motion(coordinates, alpha, raw, direction, state, state_direction,
                                   output);
}
#endif

/** Native SCF callback: parameters are owned snapshots and density is borrowed
 * only for this synchronous action. No Python callback enters the hot loop. */
extern "C" int integral_ad_direct_fock(const double* parameters, const double* direction,
                                       const double* da, const double* db, double* output) {
  using namespace integral_ad;
  try {
    if (direction) {
#ifdef GQC_INTEGRAL_AD_SECOND
      return integral_ad_direct_motion(parameters, parameters + 3 * nat,
                                       parameters + 3 * nat + nprim, direction, nullptr, nullptr,
                                       output, da, db);
#else
      return 1;
#endif
    }
    return direct_execute<double>(parameters, parameters + 3 * nat, parameters + 3 * nat + nprim,
                                  nullptr, output, da, db);
  } catch (...) {
    return 2;
  }
}
