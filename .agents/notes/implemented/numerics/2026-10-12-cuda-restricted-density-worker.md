# Decision: explicit CUDA restricted density work, not P2 folding

Status: implemented
Date: 2026-10-12

## Problem

The automatic CUDA RKS sibling folded equal spin coordinates only after
constructing polarized derivatives. For total density between one and two
worker floors this evaluates a different worker from Libxc's unpolarized
normalization. Prepared U1 tile results did not exercise the actual automatic
provider. The provider also advertised legacy domain version 4 for density work.

## Decision

Reuse `build_bulk_runtime_program(spin="unpolarized", domain=DENSITY_WORK_DOMAIN)`
and the incumbent scalar Graph emitter for the density-only RKS body. The
shared physical-spin traversal remains unchanged: its paired half-density
transport is restored to total density before worker projection. The U1 body
rejects unequal transport channels and duplicates its scalar potential for the
incumbent half-channel pullback. This is explicit producer spin selection, not
an equality-based guess about UKS semantics.

Both real launcher schedules select this U1 body for density-only RKS even when
the equal-spin optimization knob is off. UKS retains P2, including equal inputs.
Other ingredient families retain their existing P2 equal-spin continuation.
No functional-name scientific branch, new mathematical formula or runtime
oracle is introduced.

The compiler point-source contract records spin, domain, domain version,
features, derivative roots and normalization. CUDA descriptors retain their
native ABI, with a combined mathematical identity binding both layouts. The
provider source identity advances to v2 and binds both point sources and exact
traversal/launch bytes. Density-only metadata is exactly version 5; legacy
semilocal metadata remains version 4. Old source receipts are conservatively
invalidated, not relabeled as new qualification.

## Rejected alternatives

- Disabling scalar sharing: the knob-off path still evaluated P2 work floors.
- Labeling the P2 body U1 or changing only domain metadata: neither changes
  the active-tail worker or derivative normalization.
- Replacing the common traversal, extending the native descriptor or adding
  named LDA implementations: none is necessary for the producer-bound fix.
- Treating a prepared U1 tile or point fixture as a CUDA KS/force endpoint:
  those products do not exercise the complete automatic method.

## Invariants

Keep independent Libxc spin-0 and spin-1 gates, the complete original
11-functional point matrix, both launcher modes, knob-off/on, compact partial
tiles, and invalid binding/input gates. Source changes must invalidate receipts.
Keep physical empty-spin/birth and complete endpoint/backend promotion gates
separate; ordinary force success does not clear retained boundary failures.

## Evidence

The predecessor diagnosis ran actual single/batched providers with knob 0/1
for three LDAs in finite Slurm job 7815 on node1, preserving assigned device
visibility 4. All 12 rows showed wrong active-tail potentials, with absolute
errors 1.56e-6 to 2.04e-6, and advertised domain version 4. Raw source, binaries,
logs and JSON remain in the owned ignored runtime evidence.

`tests/python/test_automatic_restricted_cuda.py` retains explicit cross-backend
U1 identity, strict host execution of CUDA scalar bytes against independent
Libxc 7.0.0, stale-layout/domain negatives, and opt-in actual CUDA-provider
tests. GPU tests require `GENERATIVEQC_XC_CUDA_TEST=1` inside finite `srun`, keep
Slurm visibility, compile with the verified existing cache, and retain exact
source/object/binary identities, resource reports and per-row raw values.
The original polarized scalar and registry-wide sharing tests remain.

The GPU wrapper retains the original 2e-10 absolute/relative energy and
coefficient gates for GGA/meta-GGA, while density-only energy has an additional
1e-30 absolute active-tail gate and physical spin counts retain their strict
gate. An initial blanket 1e-30 absolute energy test was inappropriate for
near-zero cancellation: the APBE fixture with large gradient had independently
compiled/oracle energies differing by at most 1.91e-21, although all original
point gates and coefficients passed. The first two failed harness runs remain
retained; they are not promoted as qualification. This does not waive the
separate retained APBE/GVT4/TPSS production-domain failures.

## Revisit when

Additional ingredient families gain independently qualified explicit U1
worker contracts, or the native ABI deliberately adopts per-layout descriptor
fields after complete matched endpoint qualification.

## References

- Issues #1118, #1119, #1121 and #1124.
- [Stationary point layout](2026-10-12-stationary-point-layout.md).
- [Density work contract](../../../../docs/developer/libxc_bulk_capabilities.md).
