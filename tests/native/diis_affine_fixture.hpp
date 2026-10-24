#pragma once

#include <array>

// Final three residuals from a cold Direct r2SCAN-3c/96 endpoint. The original
// pivoted FP64 solve accepts this nearly dependent Gram while losing sum(c)=1
// by about 1e-5 (CUDA FMA) or 3e-5 (unfused host arithmetic). An independent
// 80-decimal solve satisfies the constraint to <1e-75. Keep the raw metric so
// normalization and chronological retirement are exercised by each consumer.
inline constexpr std::array<double, 9> ill_conditioned_diis_gram{
    4.2551308937401342e-07,  -1.8408448414771539e-08, -1.8408448417904224e-08,
    -1.8408448414771539e-08, 1.6392064817049214e-08,  1.6392064817374055e-08,
    -1.8408448417904224e-08, 1.6392064817374055e-08,  1.6392064817699365e-08};
