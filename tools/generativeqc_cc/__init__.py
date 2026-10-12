"""Internal conventional coupled-cluster equations, state and validation APIs.

The facade accepts validated closed-shell RHF snapshots and conventional CPU
integral providers. RCCSD supports CPU, ordinary-stream CUDA and resident CUDA;
RCCSD(T) composes CPU or resident CUDA RCCSD with audited bounded triples tiles
and offers isolated homogeneous Python batch helpers. #155 additionally binds
the validated complete RCCSD(T) analytic-gradient owner as an internal force API.
These APIs still do not register Calculator methods or a native CCSD(T) prepared
owner; the C ABI remains fail-closed until the generated response graph has that
owner.

PreparedUCCSD separately accepts validated UHF snapshots with conventional
spin-resolved providers and shared native CPU/CUDA TensorIR contractions. Its
explicit host-controlled iteration contract does not register Calculator
UCCSD/UCCSD(T) methods. PreparedUCCSDT
qualifies a converged UCCSD state and adds all four canonical triples channels
through bounded shared CPU/CUDA tiles with complete numeric/work admission.
PreparedUCCSDLambda supplies native CPU/CUDA Lambda and fixed-orbital input
cotangents in explicit physical excitation-dual coordinates, reusing the existing
GMRES acceptance contract. PreparedUCCSDPhysicalResponse binds the raw spin
Hamiltonian and shared UHF adjoint; PreparedUCCSDGradient streams complete
conventional shell derivatives. CPU one-electron derivatives explicitly use the
native compatibility recurrence; CUDA derivatives do not fall back to CPU.
Neither owner supplies corrected-(T) response or public method registration.
"""

from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.equations import amplitude_layouts, build_program

from .api import (
    BatchItemResult,
    BatchRCCSDResult,
    Capabilities,
    RCCSDResult,
    batch_energy,
    energy,
    method_capabilities,
)
from .ccsd_t_api import (
    BatchRCCSDTForceResult,
    BatchRCCSDTResult,
    PreparedRCCSDTBatch,
    PreparedRCCSDTForceBatch,
    RCCSDTBatchItemResult,
    RCCSDTCapabilities,
    RCCSDTForceBatchItemResult,
    RCCSDTResult,
    rccsd_t_batch_energy,
    rccsd_t_batch_forces,
    rccsd_t_energy,
    rccsd_t_force,
    rccsd_t_method_capabilities,
)
from .complete_gradient import (
    BoundCCSDGradient,
    CCSDGradientCapabilities,
    CCSDGradientOptions,
    CCSDGradientResult,
    complete_gradient_validation,
    gradient_capabilities,
)
from .df_factorized import (
    DFCCSDResult,
    FactorizedDFIntegralState,
    PreparedDFCCSD,
    solve_df_ccsd,
    virtual_correction_workspace_bytes,
    virtual_corrections,
)
from .df_gradient import (
    DFThreeIndexCotangent,
    DFThreeIndexPullback,
    pullback_df_three_index,
)
from .df_triples import (
    DFCCSDTResult,
    factorized_triples_energy,
    factorized_triples_workspace_bytes,
    solve_df_ccsdt,
)
from .evaluate import evaluate
from .lambda_cuda import PreparedCUDALambda
from .lambda_equations import (
    CCSDLambdaPrograms,
    build_lambda_programs,
    build_parameter_vjp,
)
from .lambda_response import BoundCCSDResponse, CCSDParameterWeight
from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, LambdaOptions
from .native_tensor_cuda import CudaCCTensorExecutor
from .resident_solver import PreparedResidentCCSD, solve_gpu_resident
from .solver import CCSDResult, PreparedCCSD, SolverOptions, solve
from .state_transport import (
    OrbitalFrameDiagnostics,
    StateIdentity,
    StateTransport,
    StateTransportPolicy,
    StateTransportRequest,
    TransportCompatibility,
)
from .triples import (
    build_triples_program,
    triples_energy,
    triples_energy_tensorir,
    triples_fullsum,
)
from .triples_complete_gradient import (
    BoundCCSDTGradient,
    complete_ccsdt_cuda_response_gradient_validation,
    complete_ccsdt_gradient_validation,
)
from .triples_cuda import (
    CudaTriplesResult,
    CudaTriplesTiles,
    TriplesTileConfig,
    cpu_triples_tiles,
)
from .triples_lambda_response import (
    BoundCCSDTResponse,
    CCSDTParameterWeight,
    CorrectedLambdaResult,
    solve_corrected_lambda,
)
from .triples_orbital_response import BoundCCSDTOrbitalResponse
from .triples_response import (
    TRIPLES_RESPONSE_INPUTS,
    accumulate_tile_triples_vjp,
    build_full_triples_vjp,
    build_tile_triples_vjp,
    full_triples_vjp,
    tile_triples_vjp,
)
from .triples_response_cuda import (
    CudaTriplesResponseResult,
    CudaTriplesResponseTiles,
    solve_corrected_lambda_cuda,
)
from .triples_tiles import (
    TileSpec,
    TriplesTileEnumerator,
    build_tile_triples_program,
    tile_triples_energy,
    tile_triples_energy_masked,
    tile_triples_energy_tensorir,
)
from .uccsd_gradient import (
    PreparedUCCSDGradient,
    UCCSDGradientOptions,
    UCCSDGradientResult,
)
from .uccsd_lambda import (
    PreparedUCCSDLambda,
    UCCSDLambdaOptions,
    UCCSDLambdaResult,
    UCCSDParameterCotangent,
)
from .uccsd_physical_response import (
    PreparedUCCSDPhysicalResponse,
    UCCSDPhysicalResponseOptions,
    UCCSDRelaxedCotangents,
)
from .uccsd_solver import PreparedUCCSD, UCCSDIteration, UCCSDResult
from .uccsd_t import PreparedUCCSDT, UCCSDTOptions, UCCSDTResult
from .uccsd_t_lambda import (
    PreparedUCCSDTLambda,
    UCCSDTLambdaOptions,
    UCCSDTLambdaResult,
)
from .uccsd_triples_fock_response import (
    PreparedUCCSDTriplesFockResponse,
    UCCSDTriplesFockOptions,
    UCCSDTriplesFockResponseResult,
)
from .uccsd_triples_response import (
    PreparedUCCSDTriplesResponse,
    UCCSDTriplesResponseResult,
)

__all__ = [
    "TRIPLES_RESPONSE_INPUTS",
    "BatchItemResult",
    "BatchRCCSDResult",
    "BatchRCCSDTForceResult",
    "BatchRCCSDTResult",
    "BoundCCSDGradient",
    "BoundCCSDLambda",
    "BoundCCSDResponse",
    "BoundCCSDTGradient",
    "BoundCCSDTOrbitalResponse",
    "BoundCCSDTResponse",
    "CCSDGradientCapabilities",
    "CCSDGradientOptions",
    "CCSDGradientResult",
    "CCSDLambdaPrograms",
    "CCSDLambdaResult",
    "CCSDParameterWeight",
    "CCSDResult",
    "CCSDTParameterWeight",
    "Capabilities",
    "CorrectedLambdaResult",
    "CudaCCTensorExecutor",
    "CudaTriplesResponseResult",
    "CudaTriplesResponseTiles",
    "CudaTriplesResult",
    "CudaTriplesTiles",
    "DFCCSDResult",
    "DFCCSDTResult",
    "DFThreeIndexCotangent",
    "DFThreeIndexPullback",
    "FactorizedDFIntegralState",
    "LambdaOptions",
    "OrbitalFrameDiagnostics",
    "PreparedCCSD",
    "PreparedCUDALambda",
    "PreparedDFCCSD",
    "PreparedRCCSDTBatch",
    "PreparedRCCSDTForceBatch",
    "PreparedResidentCCSD",
    "PreparedUCCSD",
    "PreparedUCCSDGradient",
    "PreparedUCCSDLambda",
    "PreparedUCCSDPhysicalResponse",
    "PreparedUCCSDT",
    "PreparedUCCSDTLambda",
    "PreparedUCCSDTriplesFockResponse",
    "PreparedUCCSDTriplesResponse",
    "RCCSDResult",
    "RCCSDTBatchItemResult",
    "RCCSDTCapabilities",
    "RCCSDTForceBatchItemResult",
    "RCCSDTResult",
    "SolverOptions",
    "StateIdentity",
    "StateTransport",
    "StateTransportPolicy",
    "StateTransportRequest",
    "TileSpec",
    "TransportCompatibility",
    "TriplesTileConfig",
    "TriplesTileEnumerator",
    "UCCSDGradientOptions",
    "UCCSDGradientResult",
    "UCCSDIteration",
    "UCCSDLambdaOptions",
    "UCCSDLambdaResult",
    "UCCSDParameterCotangent",
    "UCCSDPhysicalResponseOptions",
    "UCCSDRelaxedCotangents",
    "UCCSDResult",
    "UCCSDTLambdaOptions",
    "UCCSDTLambdaResult",
    "UCCSDTOptions",
    "UCCSDTResult",
    "UCCSDTriplesFockOptions",
    "UCCSDTriplesFockResponseResult",
    "UCCSDTriplesResponseResult",
    "accumulate_tile_triples_vjp",
    "amplitude_layouts",
    "batch_energy",
    "build_ccsd_program",
    "build_full_triples_vjp",
    "build_lambda_programs",
    "build_parameter_vjp",
    "build_program",
    "build_tile_triples_program",
    "build_tile_triples_vjp",
    "build_triples_program",
    "complete_ccsdt_cuda_response_gradient_validation",
    "complete_ccsdt_gradient_validation",
    "complete_gradient_validation",
    "cpu_triples_tiles",
    "energy",
    "evaluate",
    "factorized_triples_energy",
    "factorized_triples_workspace_bytes",
    "full_triples_vjp",
    "gradient_capabilities",
    "method_capabilities",
    "pullback_df_three_index",
    "rccsd_t_batch_energy",
    "rccsd_t_batch_forces",
    "rccsd_t_energy",
    "rccsd_t_force",
    "rccsd_t_method_capabilities",
    "solve",
    "solve_corrected_lambda",
    "solve_corrected_lambda_cuda",
    "solve_df_ccsd",
    "solve_df_ccsdt",
    "solve_gpu_resident",
    "tile_triples_energy",
    "tile_triples_energy_masked",
    "tile_triples_energy_tensorir",
    "tile_triples_vjp",
    "triples_energy",
    "triples_energy_tensorir",
    "triples_fullsum",
    "virtual_correction_workspace_bytes",
    "virtual_corrections",
]
