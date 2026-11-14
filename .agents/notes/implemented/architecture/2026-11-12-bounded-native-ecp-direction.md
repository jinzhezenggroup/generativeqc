# Decision: preserve full ECP coordinate gates in a bounded directional source

Status: implemented (internal source; not a complete stationary endpoint)
Date: 2026-11-12

## Problem

The host integral nuclear source includes ionic attraction but not residual
local/nonlocal ECP operators. Contracting the legacy first-derivative tensor
would retain coordinate-times-AO-squared storage. Testing only a projected
direction's coarse/fine agreement would silently weaken the existing ECP
quadrature contract: a zero or cancelling direction could hide an unacceptable
physical-coordinate derivative.

## Decision

Add a synchronous `checked_ecp_direction` in the ordinary native core. Keep
the old value/derivative ABI unchanged. Analytic power-two residuals with
local/s/p/d projectors reuse the existing Gaussian pair/primitive jets,
compensated sums and bounded-work preflight, retaining two directional matrices
rather than coordinate tensors. Other legal powers/projectors retain the
checked-grid fallback. Shared `ecp_grid_cpu` metadata and radial layers keep
the raw and directional consumers on the same generated scientific authority.
The compiler's existing scalar Graph owns the relative-center projection.

For any fixed public AO pair, derivatives on its A and B atoms collect all
ECP sites. A coordinate distinct from both AO atoms receives only the
contribution of its own ECP site. Retain A/B sums until the traversal ends,
but check and discard every other completed physical coordinate per site.
Scatter coincident A/B/site identities before checking. Retain the total
primal matrix across sites rather than imposing a different per-site primal
gate. The resulting 28 global, 12 current-site and two output matrix panels
cost 42 `nao^2` doubles, not a coordinate tensor. AO/projection scratch is
shared across all pairs in each radial layer; no coordinate-page source
replay is introduced.

The query includes owned numeric/grid/AO staging with conservative capacity
allowances and checked arithmetic. Mandatory backend-specific work caps count
all radial frontiers (including zero-potential layers), primitive-term
AO-point visits and contraction work. Borrowed inputs, allocator overhead,
stack and total RSS/instructions/time are not certified by that query.
Owner validation is mandatory during traversal and at final publication.
Both local/nonlocal matrices return together; failure returns neither.

## Rejected alternatives

- Contract the full raw derivative tensor: unbounded coordinate storage.
- Gate only the projected direction or primal matrix: admits derivative-only
  grid failure for zero/translation directions.
- Gate every ECP site's primal separately: changes the existing total-matrix
  convergence contract in the presence of cancelling residual sites.
- Recompute all sources once per coordinate page: bounded memory but amplified
  work; the physical-coordinate sparsity makes that unnecessary here.
- Introduce another derivative recurrence or production finite differences:
  loses shared scientific authority and independent validation boundaries.

## Evidence and boundaries

`tests/python/test_native_ecp_direction.py` uses independent libcint primal
local/nonlocal matrices at three difference steps, Cartesian/spherical s--f,
and synthetic analytic/general residuals. A sharp-AO fixture constructs an
actual derivative-only coarse/fine failure while the primal gate passes;
legacy and directional zero-projection consumers must both reject. Other
controls cover raw-coordinate contraction, translation, linearity, limits,
owner failure/recovery, and basis-free all-electron atoms not expanding source
workspace/work. Native execution does not require Python on PATH.

These tests do not certify representative atomic ECPs or full endpoints.
Composition with the host integral and moving-XC sources, second-source
derivatives, stationary Hessian assembly, Python-free CPU TS and complete
representative endpoint work/time/resource qualification remain separate work.
Source-matched local receipts are retained under
`build/qualification/host-ecp-direction-20261112/`; earlier accepted integral
receipts describe their own source cohort and must not be relabeled.

## Revisit when

The ECP projector domain or acceptance contract changes, batching introduces
source reuse across multiple nuclear directions, or complete endpoint evidence
shows this source dominates. Preserve independent reference checks, full
physical-coordinate gates and explicit finite work limits in any replacement.
