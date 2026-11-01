#pragma once

/** Optional bounded spatial XC queue. Each slot copies only the current tile,
 * so later AO production cannot overwrite a queued point or its error witness.
 * GridPlan owns these buffers and destroys them before its stream/context. */
struct SpatialXcBatch {
  generativeqc::runtime::OwnedCudaBuffer<double> numeric;
  generativeqc::runtime::OwnedCudaBuffer<size_t> maps;
  generativeqc::runtime::OwnedCudaBuffer<CudaXcCompactTile> descriptors;
  generativeqc::runtime::OwnedCudaBuffer<int> errors;
  std::vector<CudaXcCompactTile> host;
  size_t capacity{}, tiles{}, active{}, jets{}, queued{}, bytes{};
  bool fp32{};
  int evaluator = -1;
  bool restricted{};
  double *ao{}, *work{}, *features{}, *values{}, *weights{}, *point_totals{}, *local{}, *totals{};

  void allocate(Context& ctx, size_t slots, size_t points, size_t aos, size_t ao_jets, bool mixed,
                size_t expected_bytes) {
    using generativeqc::runtime::size_add;
    using generativeqc::runtime::size_mul;
    capacity = points;
    tiles = slots;
    active = aos;
    jets = std::min(size_t(4), ao_jets);
    fp32 = mixed;
    const auto panel = size_mul(size_mul(tiles, capacity), active);
    const auto points_all = size_mul(tiles, capacity);
    const auto matrices = size_mul(size_mul(2, tiles), size_mul(active, active));
    const auto elements = size_add(size_add(size_mul(jets + 8, panel), size_mul(28, points_all)),
                                   size_add(matrices, 6));
    bytes = size_add(size_mul(8, elements),
                     size_add(size_mul(sizeof(size_t), size_mul(tiles, active)),
                              size_add(size_mul(sizeof(CudaXcCompactTile), tiles),
                                       size_mul(sizeof(int), tiles + 1))));
    if (bytes != expected_bytes) throw std::invalid_argument("spatial XC batch storage mismatch");
    host.resize(tiles);
    numeric.allocate(ctx.device, elements, ctx.stream);
    maps.allocate(ctx.device, size_mul(tiles, active), ctx.stream);
    descriptors.allocate(ctx.device, tiles, ctx.stream);
    errors.allocate(ctx.device, tiles + 1, ctx.stream);
    ao = numeric.get();
    work = ao + jets * panel;
    features = work + 8 * panel;
    values = features + 13 * points_all;
    weights = values + 11 * points_all;
    point_totals = weights + points_all;
    local = point_totals + 3 * points_all;
    totals = local + matrices;
  }

  void flush(Context& ctx, int evaluator, bool restricted, unsigned ingredients, size_t nao,
             double* potential) {
    if (!queued) return;
    int* error = errors.get() + tiles;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(descriptors.get(), host.data(), queued * sizeof(CudaXcCompactTile),
                                 cudaMemcpyHostToDevice, ctx.stream));
    });
    ctx.section(true, ctx.metrics.kernel_ms, [&] {
      spatial_batch_points<<<dim3(blocks(capacity, 128), queued), 128, 0, ctx.stream>>>(
          descriptors.get(), capacity, evaluator, restricted, features, weights, values,
          point_totals, errors.get(), error);
      cuda_check(cudaGetLastError());
      spatial_batch_panels<<<dim3(blocks(2 * capacity * active, 128), queued), 128, 0,
                             ctx.stream>>>(descriptors.get(), ingredients, values, ao, weights,
                                           work, error);
      cuda_check(cudaGetLastError());
      const size_t columns = (active + 15) / 16;
      const dim3 domain(columns * (columns + 1) / 2, 1, 2 * queued);
      const I work_jets = ingredients & 8U ? 4 : 1;
      if (fp32)
        spatial_batch_potential<float><<<domain, dim3(16, 16), 0, ctx.stream>>>(
            descriptors.get(), ao, work, weights, local, work_jets, error);
      else
        spatial_batch_potential<double><<<domain, dim3(16, 16), 0, ctx.stream>>>(
            descriptors.get(), ao, work, weights, local, work_jets, error);
      cuda_check(cudaGetLastError());
      batch_ordered_scatter<true, double><<<dim3(2, (nao + 15) / 16), 128, 0, ctx.stream>>>(
          descriptors.get(), queued, local, nullptr, 0, maps.get(), nao, potential, totals, error);
      cuda_check(cudaGetLastError());
      accumulate_compensated_xc_totals<<<1, 32, 0, ctx.stream>>>(point_totals, queued * capacity,
                                                                 capacity, totals, error);
      cuda_check(cudaGetLastError());
    });
    queued = 0;
  }
};
