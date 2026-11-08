"""Native resource adapter for the existing density portfolio."""


def emit_density_provider() -> str:
    """Bind shared GEMM/resources; scientific materialization is emitted separately."""
    return r"""
#if defined(GENERATIVEQC_TEST_HOOKS)
extern "C" void xc_density_provider_for_test(bool library, bool unavailable) {
  tensor::panel_product_library_for_test = library;
  tensor::panel_product_unavailable_for_test = unavailable;
}
extern "C" void xc_density_indexed_provider_for_test(bool library) {
  tensor::panel_product_bounded_library_for_test = library;
}
#endif
std::unique_ptr<tensor::PreparedPanelProduct> prepare_density_provider(
    const CudaXcLayout& l, cudaStream_t stream, std::size_t provider_budget,
    std::size_t automatic_budget, std::uint64_t expected_replays) {
  if (!expected_replays) throw std::invalid_argument("density reuse must be positive");
  // Automatic storage is a separate optional quote, not permission to expand
  // the caller's ordinary generated/provider reservation after a profile miss.
  const bool indexed_gemm=indexed_fp64_profile_rejection(l,automatic_budget).empty();
  const auto budget=indexed_gemm ? automatic_budget : provider_budget;
  bool qualified_indexed_gemm=indexed_gemm;
#if defined(GENERATIVEQC_TEST_HOOKS)
  // Small oracle fixtures exercise the exact production FP64 math policy,
  // without exposing a public recipe bypass or expanding automatic admission.
  qualified_indexed_gemm=qualified_indexed_gemm ||
      (l.local_ao && tensor::panel_product_bounded_library_for_test);
#endif
  return std::make_unique<tensor::CudaPanelProduct>(
      density_lowering_request, density_lowering_candidates, density_lowering_target,
      density_lowering_compilation, l.nao, tensor::contraction_product(l.work_jets,l.tile_points),
      l.spins, stream, budget, tiled_xc_admitted(l.nao,l.tile_points,l.spins,l.work_jets) ? 1 : 0,
      l.local_ao ? 5 : 4, !l.response, qualified_indexed_gemm, expected_replays);
}
#if defined(GENERATIVEQC_TEST_HOOKS)
extern "C" void xc_density_materialize_for_test(cudaStream_t stream, const double* density,
    std::size_t n, std::size_t batches, double* output, int* error) {
  materialize_density_factor<<<blocks(n*n*batches,128),128,0,stream>>>(density,n,batches,output,error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}
extern "C" void xc_density_gather_for_test(cudaStream_t stream, const double* density,
    std::size_t global_n, std::size_t active, std::size_t batches, const std::size_t* ids,
    double* output, int* error) {
  gather_density_factor<<<blocks(active*active*batches,128),128,0,stream>>>(
      density,global_n,active,batches,ids,output,error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}
#endif
"""
