#pragma once

#include <cstddef>
#include <cstdint>

#include "dft/ao_grid_work.hpp"
#include "dft/grid_task_view.cuh"
#include "tensor/metrics.hpp"

/** Private typed declarations for the existing native grid owner ABI. */
extern "C" {
int grid_cuda_create_v3(int, int, int, const std::size_t*, const double*, std::size_t, unsigned,
                        std::size_t, std::size_t, const std::size_t*, std::size_t, unsigned, void**,
                        char*, std::size_t);
void grid_cuda_destroy_v1(void*);
int grid_cuda_density_device_v1(void*, const double*, const double*, std::size_t, unsigned, void*,
                                char*, std::size_t);
int grid_cuda_run_selected_device_deferred_v1(void*, const double*, std::size_t, int,
                                              const std::size_t*, std::size_t, double*, double*,
                                              char*, std::size_t);
int grid_cuda_view_v1(void*, generativeqc::dft::GridTaskView*, char*, std::size_t);
int grid_cuda_density_jets_v2(void*, std::uint64_t, unsigned, const double**, std::uint64_t*, char*,
                              std::size_t);
int grid_cuda_metrics_v1(void*, generativeqc_tensor::Metrics*, int*, char*, std::size_t);
int grid_cuda_work_metrics_v1(void*, generativeqc::dft::AoGridWork*, char*, std::size_t);
}
