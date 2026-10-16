"""Resident AO-jet traversal over the authoritative normalized packed basis."""

from .ao_cuda import emit_grid_policy


def emit_resident_ao() -> str:
    """Bind the existing through-f/order-three Gaussian AD factors to device IO."""
    return (
        emit_grid_policy()
        + r"""
#include <cuda_runtime.h>
__global__ void resident_ao(const double* basis, size_t natom, size_t nprimitive,
    size_t nao, const double* points, size_t npoint, size_t jets, double* output, int* status) {
  const double* primitives=basis+3*natom;
  const double* records=primitives+2*nprimitive;
  for (size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x; index<jets*npoint*nao;
       index+=size_t(blockDim.x)*gridDim.x) {
    const size_t orbital=index%nao, point=index/nao%npoint, jet=index/(nao*npoint);
    const double* record=records+16*orbital;
    const size_t atom=static_cast<size_t>(record[0]);
    const double x=points[3*point]-basis[3*atom];
    const double y=points[3*point+1]-basis[3*atom+1];
    const double z=points[3*point+2]-basis[3*atom+2];
    const double radius=x*x+y*y+z*z;
    const size_t first=static_cast<size_t>(record[1]), end=first+static_cast<size_t>(record[2]);
    double value=0;
    for (size_t radial=first; radial<end; ++radial) {
      const double alpha=primitives[2*radial];
      const double weight=primitives[2*radial+1]*exp(-alpha*radius);
      if (weight==0) continue;
      for (int term=0; term<static_cast<int>(record[3]); ++term) {
        value+=weight*record[7+4*term]*
          generativeqc_grid_policy::axis_jet(static_cast<int>(record[4+4*term]),generativeqc_grid_policy::derivatives[jet][0],alpha,x)*
          generativeqc_grid_policy::axis_jet(static_cast<int>(record[5+4*term]),generativeqc_grid_policy::derivatives[jet][1],alpha,y)*
          generativeqc_grid_policy::axis_jet(static_cast<int>(record[6+4*term]),generativeqc_grid_policy::derivatives[jet][2],alpha,z);
      }
    }
    if (!isfinite(value)) atomicExch(status,1);
    output[index]=value;
  }
}
extern "C" int generativeqc_resident_ao(const double* basis, size_t natom,
    size_t nprimitive, size_t nao, const double* points, size_t npoint,
    size_t jets, double* output, int* status, cudaStream_t stream) {
  const size_t size=jets*npoint*nao;
  if (size) resident_ao<<<(size+127)/128,128,0,stream>>>(basis,natom,nprimitive,nao,points,npoint,jets,output,status);
  return static_cast<int>(cudaGetLastError());
}
"""
    )
