#include <generativeqc/generativeqc.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

void require(bool condition, const char* detail) {
  if (!condition) throw std::runtime_error(detail);
}
int main(int argc, char** argv) {
  generativeqc_context* context = nullptr;
  generativeqc_d4_batch* batch = nullptr;
  try {
    const auto backend = argc > 1 && std::string(argv[1]) == "cuda"
                             ? GENERATIVEQC_BACKEND_CUDA
                             : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    generativeqc_context_descriptor desc{sizeof(desc), GENERATIVEQC_ABI_VERSION, 0, backend};
    require(generativeqc_context_create(&desc, &context) == GENERATIVEQC_STATUS_SUCCESS, "context");
    std::array<int32_t, 3> numbers{8, 1, 1};
    std::array<double, 9> xyz{0,
                              0,
                              0,
                              1.4304281900354767,
                              0,
                              1.0999887901572898,
                              -1.4304281900354767,
                              0,
                              1.0999887901572898};
    const std::array<double, 9> reference{5.2939559203393771e-23, 0, 9.2079299545713894e-6,
                                          -3.7885512553180557e-6, 0, -4.6039649772856955e-6,
                                          3.7885512553180561e-6,  0, -4.6039649772856955e-6};
    generativeqc_d4_system_descriptor system{
        sizeof(system), GENERATIVEQC_ABI_VERSION, numbers.data(), xyz.data(), 3, 0};
    generativeqc_d4_bj_eeq_descriptor model{sizeof(model),
                                            GENERATIVEQC_ABI_VERSION,
                                            GENERATIVEQC_D4_PROFILE_R2SCAN3C_EEQ,
                                            1,
                                            0,
                                            2,
                                            0.42,
                                            5.65,
                                            2,
                                            1,
                                            30,
                                            60,
                                            40,
                                            64u << 20};
    require(generativeqc_d4_batch_prepare(context, &system, 1, &model, &batch) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "prepare");
    std::array<double, 9> gradient{};
    std::array<double, 3> charges{};
    generativeqc_d4_batch_item_result_descriptor result{};
    result.struct_size = sizeof(result);
    result.abi_version = GENERATIVEQC_ABI_VERSION;
    result.gradient = gradient.data();
    result.gradient_count = gradient.size();
    result.charges = charges.data();
    result.charge_count = charges.size();
    for (int replay = 0; replay < 2; ++replay) {
      require(generativeqc_d4_batch_execute(batch, nullptr, 0, &result, 1) ==
                      GENERATIVEQC_STATUS_SUCCESS &&
                  result.status == GENERATIVEQC_STATUS_SUCCESS,
              "execute");
      require(result.executed_backend == backend, "backend");
      require(std::abs(result.energy + 8.5663634618552597e-5) < 2e-13, "energy oracle");
      for (std::size_t i = 0; i < gradient.size(); ++i)
        require(std::abs(gradient[i] - reference[i]) < 2e-12, "gradient oracle");
      require(std::abs(charges[0] + 0.60153994622485873) < 1e-10, "charge oracle");
    }
    generativeqc_d4_batch_destroy(batch);
    batch = nullptr;
    generativeqc_context_destroy(context);
    context = nullptr;
    std::puts("installed D4 SDK independent E/gradient/charge replay passed");
    return 0;
  } catch (const std::exception& error) {
    generativeqc_d4_batch_destroy(batch);
    generativeqc_context_destroy(context);
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
