# Decision: bind COSX matrix sites through a shared prepared owner

Status: implemented
Date: 2026-10-05

## Problem

COSX value assembly had native scalar inner loops for `AO * D` and
`raw += AO^T * potential`. Replacing them with method-local library calls would
violate #1886, while separate per-operation owners would duplicate provider
handles and resource reservations. Full/tail shapes are known before execution.

## Decision

Express the two existing equations in TensorIR. The second graph contains the
previous raw matrix as an explicit input with exclusive last-use donation.
CPU and CUDA requests share scientific/semantic identity; the matrix descriptor
retains the complete update identity and `beta=1`. No symmetry of the input
density is assumed. Weight application, ESP integrals, final symmetrization and
RHF/spin-resolved energy conventions stay with their existing owners.

Prepare four concrete sites in a shared `PreparedContractionSites` resource
owner: full projection/update and tail projection/update. Each site independently
selects from the same canonical portfolio and retains its resolved descriptor,
selected candidate, rejection and physical work counts. One context serves all
sites on the existing grid stream. The enclosing plan destroys that context
before destroying the borrowed stream, including constructor failure paths.

The common matrix executor supplies generated ordered FP64 and pedantic library
execution with an output finite audit. Generated matrix evaluation uses its
established explicit RN multiplication/addition and applies the seed after the
tile product; bitwise equivalence to the former compiler-fused scalar loop is
not claimed. Independent numerical gates cover the change. Nonfinite flags are
sticky before downstream operations can mask invalid values. Capture is rejected
because this table's host counters have no physical graph replay publisher.

The host binding reserves 128 KiB, including bounded descriptor/selection
temporaries. All sites share one optional 96 MiB device allowance with zero
explicit numeric packing. The pure staging estimate reports host capacity;
actual staging diagnostics add admitted provider storage to the existing device
request. Budget shortfalls and optional provider allocation rejection retain
generated execution. Host ownership failures and execution errors propagate.
An exception fence drains uploaded caller spans before returning a provider
failure, even if the plan survives. A separate younger fence protects local
download targets before vector destruction. Fault injection checks both borrows
and a successful subsequent replay.

Production has no promoted library endpoint profile. Test-only qualification
scores exercise projection and accumulation independently, both together,
provider unavailability and missing provider budget. Scores are not performance
estimates. Complete method endpoints must qualify any later promotion.

## Validation contract

The native `--contractions` gate compares full/tail products and asymmetric seed
updates with an independent long-double oracle. It exercises alias rejection,
explicit capture rejection and device-injected nonfinite/sticky errors on both
implementations. Actual COSX endpoints use independent CPU raw/symmetric K and
energy, spherical s/d/f functions, asymmetric changed densities, signed weights,
RHF/spin-resolved conventions and three tile shapes across six admission routes.
Per-site counters must match physical full/tail work and one provider reservation.
The existing full COSX value/derivative/lifetime regression remains required.

Reproduction, source manifests, ccache commands/statistics, native qualification
and sanitizer evidence are retained under ignored `.artifacts/1884-cosx-lowering/`.
Native compilation uses n1; all real-device execution uses finite Slurm
`main/gpu:5090:1` allocations on n1/node1.

## Follow-up scope

Point-batched ESP application and derivative variants need the same semantic
boundary in separate small PRs. The shared owner is independent of COSX and can
host their concrete sites; no COSX-local handle/selector should be added.
Complete cold/warm/changed-geometry COSX/SCF timing and matched RI-K crossover
remain required under #1884/#246. This slice does not complete #1886.
