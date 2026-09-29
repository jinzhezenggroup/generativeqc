#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "tensor/cpu_linalg.hpp"

namespace generativeqc::posthf {

[[nodiscard]] std::size_t rank2_transform_workspace_bytes(std::size_t n);

[[nodiscard]] std::vector<double> rank2_ao_to_mo(
    std::span<const double> coefficients, std::span<const double> ao, std::size_t n,
    const tensor::CpuLinalgPlan& plan = {});

[[nodiscard]] std::vector<double> rank2_mo_to_ao(
    std::span<const double> coefficients, std::span<const double> mo, std::size_t n,
    const tensor::CpuLinalgPlan& plan = {});

}  // namespace generativeqc::posthf
