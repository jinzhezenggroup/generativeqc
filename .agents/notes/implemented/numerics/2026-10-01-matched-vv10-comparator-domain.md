# Decision: match both GPU4PySCF VV10 density masks to the native contract

Status: implemented
Date: 2026-10-01

## Problem

Correcting the through-f public endpoint exposes a 2.46e-8 Eh difference on
def2-TZVP water, outside the unchanged 1e-8 Eh independent acceptance gate.
Changing native mathematics or weakening the gate would conceal the cause.

GPU4PySCF 1.8.1 uses `NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-10` in SCF and
copies this constant into its RKS gradient module at import. Native MolecularV1
VV10 and PySCF 2.14.0 use rho >= 1e-8 in both pair domains. A shared integration
grid does not establish shared nonlocal work or quadrature semantics.

## Diagnosis

At one fixed independently converged density, native AO jets agree with PySCF
to 5.7e-14; the integrated production semilocal energy differs by 2.1e-15 Eh.
GPU4PySCF Hartree, semilocal and SR/LR exchange components agree with the CPU
oracle to FP64 roundoff. Only VV10 differs:

| Evaluator | VV10 energy (Eh) |
| --- | ---: |
| Native, rho >= 1e-8 | 0.04256683725774054 |
| PySCF CPU, rho >= 1e-8 | 0.042566837257740596 |
| GPU4PySCF default rho >= 1e-10 | 0.0425668620313594 |

The complete independent CPU energy is -76.42645718763048 Eh, agreeing with
native -76.42645718763046 Eh. Tightening integral or AO screening does not
remove the GPU4PySCF difference. Aligning its SCF and force masks gives
-76.4264571878114 Eh under the diagnostic reference's default SCF settings,
within 1.81e-10 Eh of native. Libxc is 7.0.0 in both CPU and CUDA comparators.

## Decision

Scope the comparator's two imported constants together for the complete SCF
and force call, restoring defaults on success or failure. GPU4PySCF still owns
its entire SCF, integrals, XC, VV10 kernel and analytic moving-grid response.
This is explicit shared screening configuration, not a replacement oracle or
a native production dependency.

Record density policy, threshold and the comparator's separate hard-coded
absolute-weight threshold (1e-14) in the scientific protocol. Recheck the
unchanged all-call 1e-8 Eh / 1e-7 Eh/Bohr gates. Old default-domain reference
timings must not be reused as matched-domain evidence; rerun both geometries
and all frozen-density repeats.

## Invariants and rejected alternatives

Do not modify the existing native 1e-8 domain, functional graph, basis shells,
range integrals or force tolerances to compensate for a different comparator
domain. Configuring SCF alone leaves the gradient mask stale. Concurrent
reference engines with different global policies require separate processes.
The benchmark's schema/protocol changes prevent mixing historical defaults
with the matched rerun.

## Evidence and references

Host regressions check both constants and failure-safe restoration. Opt-in CUDA
tests exercise the complete public f-shell endpoint and finite differences.
Ignored diagnostics are under `.artifacts/f-energy-diagnosis.log`,
`.artifacts/reference-component-diagnosis.log`,
`.artifacts/reference-screening-diagnosis.json`, and
`.artifacts/matched-reference-gate.json`. Fresh retained measurements belong in
`benchmarks/results/omol25-wb97mv-20261001/`, not the old default-domain curve.

See the [public endpoint correction](../compatibility/2026-10-01-wb97mv-through-f-public-forces.md)
and [historical benchmark diagnosis](../performance/2026-10-01-omol25-matched-grid-benchmark.md).

## Appended independent acceptance

With both reference masks scoped together, named def2-TZVP RKS water (Slurm
11938), full local def2-TZVPD RKS water (11940), and named def2-TZVP UKS NH₂
(11947) pass the unchanged complete energy/force, translation, replay and three
finite-difference-step gates. The initial OH comparator did not converge and
is not accepted; its replacement changes the test fixture, not thresholds,
native mathematics or the OMol25 water benchmark's 48×16×32 grid/protocol.
Accepted case outcomes and enclosing job outcomes are retained separately in
[force-qualification.json](../../../../benchmarks/results/omol25-wb97mv-20261001/force-qualification.json).

The schema-v3 endpoint clock additionally includes force-array host export,
matching HF's actual `cp.asnumpy(-gradient.kernel())` timing scope. Schema-v2
reference calls stopped before that transfer; despite passing numerical gates,
their timings and paired native records are excluded from the retained curve.
Both engines rerun from separate fresh processes. Host regressions now pass
110 tests with 16 opt-in skips, including timer-order and old-schema rejection.
