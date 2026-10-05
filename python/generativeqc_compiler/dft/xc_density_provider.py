"""Native resource adapter for the existing density portfolio."""


def emit_density_provider() -> str:
    """Bind shared GEMM/resources; scientific materialization is emitted separately."""
    return r"""
#if defined(GENERATIVEQC_TEST_HOOKS)
extern "C" void xc_density_provider_for_test(bool library, bool unavailable) {
  tensor::panel_product_library_for_test = library;
  tensor::panel_product_unavailable_for_test = unavailable;
}
#endif
std::unique_ptr<tensor::PreparedPanelProduct> prepare_density_provider(
    const CudaXcLayout& l, cudaStream_t stream, std::size_t budget) {
  return std::make_unique<tensor::CudaPanelProduct>(
      density_lowering_request, density_lowering_candidates, density_lowering_target,
      density_lowering_compilation, l.nao, tensor::contraction_product(l.work_jets,l.tile_points),
      l.spins, stream, budget, tiled_xc_admitted(l.nao,l.tile_points,l.spins,l.work_jets) ? 1 : 0,
      4, !l.local_ao && !l.response);
}
#if defined(GENERATIVEQC_TEST_HOOKS)
extern "C" void xc_density_materialize_for_test(cudaStream_t stream, const double* density,
    std::size_t n, std::size_t batches, double* output, int* error) {
  materialize_density_factor<<<blocks(n*n*batches,128),128,0,stream>>>(density,n,batches,output,error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}
#endif
"""
