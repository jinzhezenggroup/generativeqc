# Decision: expose whole-request native CPU derivative work admission

Status: implemented
Date: 2026-11-12

## Problem

The native CPU C ABI and CLI already carry a finite whole-request nuclear ERI
primitive cap. The Python bridge hardcoded 100,000,000 visits. Canonical
spherical def2-SVP IH2 and NiH3 matrices require respectively 191,635,164 and
626,946,120 first-plus-curvature visits even after exact symmetry reuse. A
single HVP fits the old cap, but assembling a matrix by separately requesting
columns would conceal the complete request's admission and cost.

## Decision

Keep the existing finite default. Add `max_eri_primitives=None` to the public
Calculator matrix/HVP and molecular saddle interfaces and the retained native
batch bridge. `None` selects the incumbent 100,000,000 cap; an explicit value
must be a positive int64 count. The native owner still checks first plus
curvature work across every requested column before derivative execution.
It never admits an incomplete matrix or resets the allowance per column.
Other derivative owners reject the override rather than silently ignoring it.
The saddle allowance applies separately to each complete requested Hessian,
not to a column or the whole optimization. Actual work remains cumulative.

No numerical formula, backend, SCF root, approximation or default cap changes.
The count is not a byte bound, endpoint-time limit, induced-Fock integral count
or practical-scaling claim. Explicitly accepting more finite work does not
qualify a scientific domain; ECP public/TS admission remains separately gated.

Only the public Calculator runner labels a result as a public Calculator
endpoint. A retained-batch action is also used by development qualification and
the molecular optimizer, so it must not unconditionally claim public admission.
The scalar receipt now retains the selected primitive cap next to actual visits.

## Rejected alternatives

- Raise the default silently: removes useful bounded admission for every caller.
- Split a large matrix into individually admitted HVPs: hides whole-request
  work and defeats the caller's declared resource contract.
- Treat a larger cap as a performance fix or public heavy-element promotion:
  neither complete execution time nor scientific qualification follows from it.
- Ignore the new keyword on other owners: makes a promised limit ineffective.

## Evidence

`test_native_cpu_hessian.py` executes per-column versus whole-matrix refusal,
recovery on the same prepared batch, exact-bound acceptance and public receipt
propagation. Invalid/bool/overflow inputs fail before SCF; other owners refuse
the control. Actual molecular failure injection verifies that a primitive-work
refusal does not fall back to Python response or publish a successful saddle.
These controls and the molecular admission suite pass 69 tests in the local
`host-canonical-heavy-20261112` cohort.

A separately rebuilt/repaired wheel passed isolated installed public RKS and
UKS matrix/HVP, whole-matrix cap refusal, exact-cap recovery and molecular-saddle
work-refusal controls (`installed-api-controls-v3.json` in the same cohort).
The native source identity matches the frozen source core. The minimal CPU venv
contains no PySCF, dftd3 or CuPy; execution uses an empty environment with
compiler/search paths unavailable. This validates keyword propagation and native
execution, not canonical heavy-element science or a positive saddle. Earlier
local driver serialization failures are retained separately, not accepted.

`test_native_canonical_heavy_hessian.py` is a separately opt-in real-element
qualification procedure with a frozen manifest. It uses actual native
SCF/forces, complete raw matrices and fresh HVPs, matched independent central
states and three independent displaced-gradient stencil sizes. It does not
change public domain gates, call an external optimizer or use displaced SCFs in
production. A skipped or partially completed run is not acceptance.
Driver v2 additionally retains every displaced oracle's actual SCF state and
checks occupied cross-geometry subspaces separately for each spin. Earlier v1
gradient-only displaced receipts are not relabeled as having this evidence.

## Revisit when

A qualified schedule reduces actual primitive work, or a new resource contract
can bound complete endpoint cost more usefully. Preserve whole-request
admission, independent scientific gates and explicit failure/recovery semantics.
