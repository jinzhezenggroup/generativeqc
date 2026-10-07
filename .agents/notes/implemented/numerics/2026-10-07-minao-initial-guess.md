# Decision: opt-in occupied-ANO MINAO cold seed

Status: implemented; default promotion and CUDA performance qualification pending
Date: 2026-10-07

## Problem

The native restricted KS cold path starts from the core-Hamiltonian density when
no explicit/imported/warm density is available. The independent GPU4PySCF
reference used by the PBE0 cold comparison calls its ordinary molecular SCF
entry point without `dm0`, so PySCF/GPU4PySCF uses its default `minao` seed.
The two engines therefore did not perform comparable cold-start initialization.

The existing HF and coarse-LDA preliminary providers can reduce target
iterations, but both execute an additional SCF and Fock sequence. Issue #2052
requires a lower-cost SAD/MINAO-style candidate before changing any production
default.

## Decision

Add an explicit `InitialGuessSpec("minao")` provider. It constructs the occupied
H-Ar ANO/MINAO atomic reference used by PySCF, evaluates the native rectangular
cross overlap, applies the target metric projection

```
C_target = S_target^-1 S_target,source
D_raw = C_target occ C_target^T
```

and then passes the proposal through the existing native restricted-density
seed boundary. That boundary restores exact target electron count and symmetry
before strict ensemble-density validation. The target Hamiltonian, basis, grid,
functional, precision, screening and convergence controls are unchanged.

The source table is derived from the occupied contractions of
`pyscf/gto/basis/ano.dat` at PySCF blob
`a2e714f9e4eb395904b115cb1153aa3715a35fc7`. Fractional occupations follow
PySCF's `NRSRHF_CONFIGURATION` / `atom_hf.frac_occ`. Zero-occupation ANO
contractions are omitted because their contribution to `D_raw` is identically
zero.

The first admitted domain is:

- all-electron H-Ar only;
- restricted singlets;
- FP64 exact two-electron target providers;
- CPU RHF/KS and CUDA KS;
- no density fitting, ECP, unrestricted target or second-order endpoint.

HF/LDA keep their previous CPU energy-only domain. MINAO performs zero
preliminary Fock builds and zero preliminary SCF iterations.

## Ownership and fallback

An explicit/imported/retained density remains authoritative and skips MINAO.
Cold MINAO preparation is bounded by `maximum_numeric_bytes`; the public
resource plan separately charges all retained cold seed matrices and the
largest serialized projection workspace. The native narrow numeric bound
includes the simultaneous target orthogonalizer, raw projected density,
normalized output density, both rectangular projection buffers, occupations,
source coordinates and primitive pairs.

Preparation rejection or budget exhaustion falls back to the existing Hcore
cold path. If a MINAO-seeded target fails to converge or reports a numerical
failure, exactly one fresh Hcore target attempt is permitted. Allocation
failures propagate. A successful warm state is published only from the target
solve, never from the source projection.

## Preserved invariants

- `initial_guess=None` remains the public default.
- No runtime PySCF dependency is introduced.
- MINAO owns no target J/K, XC, DIIS or final-state object.
- Warm/checkpoint density precedence is unchanged.
- Unsupported elements/ECP/spin/provider combinations fail closed.
- No tolerance, iteration limit or target method is relaxed.
- Complete cold endpoint time must include MINAO preparation.

## Rejected alternatives

- Calling PySCF at runtime: violates the native dependency and resource model.
- Making HF or LDA preliminary SCF the default: adds a second Fock/SCF workload
  before evidence that it wins complete CUDA PBE0 cold endpoints.
- Importing the full ANO virtual inventory: zero-occupation contractions add
  cross-overlap/storage work but exactly zero density.
- Immediate default promotion: issue #2052 requires complete endpoint evidence,
  including non-water holdouts and comparison against the incumbent Hcore path.

## Evidence and remaining qualification

Focused native/API tests cover descriptor parsing, zero preliminary Fock work,
resource bounds, warm-density precedence, bounded Hcore fallback and unchanged
final target energy/density on the existing water fixture. The source table is
pinned to the PySCF blob above.

This note does **not** claim CUDA speedup or default qualification. Before
promotion, measure complete PBE0 energy/force cold endpoints on the retained
48/96-atom cases and non-water holdouts, recording MINAO construction time,
target SCF iterations/Fock builds, independent final-state gates and peak
resources. Compare both default-vs-default and matched-initialization controls.

Refs #2052

Agent: ChatGPT
Model: GPT-5.6 Sol

The original admission and numeric-cap discussion above is superseded by
[explicit MINAO ensemble admission](2026-10-07-minao-ensemble-admission.md),
which preserves the raw projection and strict global seed gate.
