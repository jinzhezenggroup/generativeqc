# Decision: compose UMP2 nuclear forces from spin cotangents and native UHF response

Status: implemented
Date: 2026-10-19

## Problem

Issue #1823 had energy and fixed-orbital MO cotangents but no complete nuclear
force. Reusing RHF relaxed weights would silently introduce factors of two and
lose opposite-spin coupling. A reference-package gradient or displaced SCF energy
would not establish a production analytic endpoint.

## Decision

Generate native VJPs from the existing three spin-labelled UMP2 TensorIR programs.
The small legacy CPU emitter does not implement all nodes produced by transpose;
use the existing bounded CPU TensorIR emitter, including range-safe bilinear
quotients and transactional finite outputs. Do not lower those quotients to a
new hand-written denominator derivative or add an independent UMP2 math owner.

Retain three natural ijab cotangents plus spin rank-two Fock multipliers. Enforce
canonical same-space stationarity, solve one coupled spin Z-vector with the native
prepared UHF J/K owner and shared GMRES, and derive overlap weights from the full
relaxed coefficient gradient. The native response adapter extends the shared
response layer; it does not implement another integral or SCF equation. The
response layer borrows a method-neutral electronic reference view and physical
Fock/owner-validation callbacks. Only the method adapter includes HF/SCF owners;
post-HF consumes a reference-bound spin response problem. Exact frame binding
rejects a same-dimension response from another reference without adding forbidden
reverse subsystem dependencies.

Pull alpha/beta weights back separately and consume the shared analytic
one-electron/ERI derivative contracts. Stage one shell slot at a time. Preserve
the default energy-only Python behavior; forces require an explicit property
request. CUDA/DF/ECP/frozen-core/complex/mixed-precision boundaries remain explicit.

## Rejected alternatives

- Restricted response/occupation multipliers: incorrect for unequal occupations
  and for broken-symmetry equal occupations with different orbital frames.
- Production finite differences or PySCF: neither qualifies native analytic force.
- A molecular AO-rank-four or coordinate-major derivative output: unnecessary for
  the existing shell-local derivative sink.
- Declaring planned capacity to be measured RSS or a speedup: no such evidence is
  produced by this implementation.

## Invariants and validation

Energy and force share exact source/reference and denominator admission. All three
spin channels contribute, exchange bars scatter into their own physical slots,
and epsilon bars accumulate repeated appearances. Actual unshifted UHF stationarity,
true response residual and final Lagrangian stationarity gate publication. Resource
and scientific failures preserve no partial force/diagnostic and do not commit a
failed warm-state candidate. Empty orbital-response spaces need no fake solve.

The molecular gate is independent PySCF 2.14.0 analytic gradients with exact native
shell inputs, plus multistep reconverged finite differences, translations,
restricted limit and alpha/beta exchange. Native controls cover generated VJP
signs/factors, inclusive capacities, nonstationary/stale owners and injected source
failure/recovery. See `docs/developer/ump2_adjoints.md` for reproducible commands
and numerical acceptance tolerances.

## Consequences and revisit

The initial CPU implementation retains dense ijab weights and streams O(N^3) MO
blocks. Its checked conservative numeric plan is finite; it is not a work-bound,
large-molecule profitability or CUDA residency claim. Revisit canonical same-space
division with a qualified noncanonical amplitude-response representation if a
stationary degenerate reference exposes unresolved numerical conditioning.


## Independent-reference diagnosis

At the initial OH/STO-3G geometry `(O,0,0,0); (H,0.2,0,1.8)` Bohr, native and
PySCF 2.14.0 UMP2 energies differed by only 2.6e-13 Hartree. The stock PySCF
analytic gradient differed by 4.75e-7 Hartree/Bohr, failing the retained 2e-7 gate.
Both independently reconverged energy differences agreed and converged toward the
native gradient as steps decreased from 1e-3 to 3e-4 to 1e-4 Bohr. The stock
CPHF result had a true physical maximum residual of 2.45e-5; requesting tol=1e-13
and 200 cycles did not change its termination/result.

The 13-dimensional independent physical response has rank 12 and a symmetry
zero mode. A NumPy SVD solve of PySCF's own J/K action yielded a fresh residual
5.8e-17 and an analytic gradient matching the native force. The permanent small
oracle therefore certifies its true residual with an independent bounded dense
solve; it does not weaken the force gate or add a production fallback. Do not
repeat the failed approach of tightening only the reference iteration tolerance
or raising the molecular force tolerance to hide an uncertified reference.
