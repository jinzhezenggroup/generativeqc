// Verbatim host bodies from 6a02e4b184a27301e06fe36762ec0c0c6902fdbf.
// Kernel launches are substituted mechanically by the Python fixture.
Gfn2EigensolverLaunchResult triangular_solve(cublasHandle_t blas, cublasSideMode_t side,
                                             cublasOperation_t operation,
                                             const Gfn2EigensolverBucket& bucket,
                                             double** factor_pointers,
                                             double** matrix_pointers) noexcept {
  const auto factors = reinterpret_cast<const double* const*>(factor_pointers);
  const cublasStatus_t status = cublasDtrsmBatched(
      blas, side, CUBLAS_FILL_MODE_LOWER, operation, CUBLAS_DIAG_NON_UNIT, bucket.orbital_count,
      bucket.orbital_count, &kOne, factors, bucket.orbital_count, matrix_pointers,
      bucket.orbital_count, bucket.system_count);
  return status == CUBLAS_STATUS_SUCCESS ? launch_success() : cublas_failure(status);
}

static Gfn2EigensolverLaunchResult solve_eigensystems_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) || !valid_cache(batch, cache) ||
      ((!dynamic_epoch && scalar_generation == 0u) ||
       (dynamic_epoch &&
        (scalar_generation != 0u || geometry_epoch->value == nullptr ||
         geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token ||
         !is_aligned(geometry_epoch->value, alignof(std::uint64_t))))) ||
      solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < batch.total_orbitals ||
      results.coefficient_elements < batch.total_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_solve_ranges(batch, cache, hamiltonians,
                          dynamic_epoch ? geometry_epoch->value : nullptr, workspace, results,
                          system_errors, device_error)) {
    return invalid_argument();
  }
  Gfn2EigensolverLaunchResult result = configure_solver(solver, stream);
  if (!result.success()) {
    return result;
  }
  result = prepare_launch_sequence(batch, workspace, device_error, stream);
  if (!result.success()) {
    return result;
  }
  result = configure_blas(blas, stream, workspace, options.deterministic_debug);
  if (!result.success()) {
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const std::int64_t matrix_begin = bucket.matrix_scratch_offset;
    const std::int64_t orbital_begin = bucket.orbital_scratch_offset;
    const std::int64_t info_begin = bucket.system_index_offset;
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, stream>>>(
        batch, bucket, cache, scalar_generation, dynamic_epoch ? geometry_epoch->value : nullptr,
        hamiltonians, options.symmetry_tolerance, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    symmetrize_transformed_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                           kThreadsPerSystem, 0, stream>>>(
        batch, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, bucket, bucket,
                                  workspace.matrix_scratch_b + matrix_begin,
                                  workspace.eigenvalue_scratch + orbital_begin, options, workspace,
                                  workspace.info_a + info_begin, stream);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    publish_eigensystems_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                         kThreadsPerSystem, 0, stream>>>(
        batch, bucket, workspace, results, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
}

static Gfn2EigensolverLaunchResult solve_spin_eigensystems_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, std::uint64_t scalar_generation,
    const Gfn2GeometryEpochDevice* geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if (!valid_spin_bucket_plan(batch, layout, buckets, bucket_count) || !valid_options(options) ||
      !valid_spin_workspace(batch, layout, workspace) || !valid_cache(batch, cache) ||
      ((!dynamic_epoch && scalar_generation == 0u) ||
       (dynamic_epoch &&
        (scalar_generation != 0u || geometry_epoch->value == nullptr ||
         geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token ||
         !is_aligned(geometry_epoch->value, alignof(std::uint64_t))))) ||
      solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < layout.total_spin_orbitals ||
      results.coefficient_elements < layout.total_spin_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_spin_solve_ranges(batch, layout, cache, hamiltonians,
                               dynamic_epoch ? geometry_epoch->value : nullptr, workspace, results,
                               system_errors, device_error)) {
    return invalid_argument();
  }
  Gfn2EigensolverLaunchResult result = configure_solver(solver, stream);
  if (!result.success()) {
    return result;
  }
  result = prepare_launch_sequence(batch, workspace, device_error, stream);
  if (!result.success()) {
    return result;
  }
  cudaError_t clear_status = cudaMemsetAsync(
      workspace.info_a, 0,
      static_cast<std::size_t>(layout.total_spin_channels) * sizeof(*workspace.info_a), stream);
  if (clear_status != cudaSuccess) {
    return cuda_failure(clear_status);
  }
  result = configure_blas(blas, stream, workspace, options.deterministic_debug);
  if (!result.success()) {
    return result;
  }

  /* Validate every bucket before launching any publication stage so a device-
   * resident topology mismatch is whole-call fail-closed. */
  result = validate_spin_solve_buckets(batch, layout, buckets, bucket_count, workspace,
                                       device_error, stream);
  if (!result.success()) {
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const Gfn2EigensolverBucket submission{
        bucket.orbital_count, bucket.solve_count, bucket.solve_index_offset,
        bucket.spin_matrix_scratch_offset, bucket.spin_orbital_scratch_offset};
    result =
        prepare_spin_solve_bucket(batch, layout, bucket, cache, scalar_generation,
                                  dynamic_epoch ? geometry_epoch->value : nullptr, hamiltonians,
                                  options, workspace, system_errors, device_error, stream);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    symmetrize_spin_transformed_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                                kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, bucket, submission,
                                  workspace.matrix_scratch_b + bucket.spin_matrix_scratch_offset,
                                  workspace.eigenvalue_scratch + bucket.spin_orbital_scratch_offset,
                                  options, workspace, workspace.info_a + bucket.solve_index_offset,
                                  stream);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    validate_spin_eigenpairs_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                             kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    publish_spin_eigensystems_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                              kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, results, system_errors);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
}

Gfn2EigensolverLaunchResult enqueue_capacity_eigensolver_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, const Gfn2EigensolverOptions& options) noexcept {
  const Gfn2EigensolverBucket submission{bucket.orbital_count, static_cast<std::int32_t>(capacity),
                                         bucket.system_index_offset, bucket.matrix_scratch_offset,
                                         bucket.orbital_scratch_offset};
  gather_compacted_solve_bucket_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
      batch, bucket, bucket_index, capacity, cache, hamiltonians, workspace);
  Gfn2EigensolverLaunchResult result = check_kernel_launch();
  if (result.success()) {
    result = configure_blas(blas, capture_stream, workspace, options.deterministic_debug);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    validate_compacted_transformed_bucket_kernel<<<capacity, 1, 0, capture_stream>>>(
        bucket, capacity, workspace, system_errors, device_error);
    result = check_kernel_launch();
  }
  if (result.success()) {
    symmetrize_compacted_bucket_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
        bucket, capacity, workspace, system_errors);
    result = check_kernel_launch();
  }
  if (result.success()) {
    result = configure_solver(solver, capture_stream);
  }
  if (result.success()) {
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, bucket, submission,
                                  workspace.matrix_scratch_b + bucket.matrix_scratch_offset,
                                  workspace.eigenvalue_scratch + bucket.orbital_scratch_offset,
                                  options, workspace, workspace.info_a + bucket.system_index_offset,
                                  capture_stream);
  }
  return result;
}

Gfn2EigensolverLaunchResult enqueue_capacity_backtransform_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, bool deterministic_debug) noexcept {
  const Gfn2EigensolverBucket submission{bucket.orbital_count, static_cast<std::int32_t>(capacity),
                                         bucket.system_index_offset, bucket.matrix_scratch_offset,
                                         bucket.orbital_scratch_offset};
  mark_backtransform_submission_kernel<<<1, 1, 0, capture_stream>>>(bucket_index, capacity,
                                                                    workspace);
  Gfn2EigensolverLaunchResult result = check_kernel_launch();
  if (result.success()) {
    result = configure_blas(blas, capture_stream, workspace, deterministic_debug);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    scatter_compacted_eigensystems_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
        batch, bucket, capacity, workspace, results, system_errors, device_error);
    result = check_kernel_launch();
  }
  return result;
}
