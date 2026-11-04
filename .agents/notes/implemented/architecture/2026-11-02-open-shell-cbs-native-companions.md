# Decision: open-shell CBS uses native same-UHF-reference energy groups

Status: implemented
Date: 2026-11-02

The initial energy-only scope is superseded by
`../numerics/2026-11-03-unrestricted-cbs-analytic-cuda-df-response.md`.

## Problem

CBS orchestration rejected every non-singlet although native conventional CPU
UMP2 and UCCSD(T) energy owners already existed. Their public companion API
admitted only restricted methods. Removing the singlet guard alone would either
run RHF for a high-spin state or fail later, and separate UHF/UMP2/CC endpoints
would repeat SCF and could select inconsistent reference branches.

## Decision

Multiplicity greater than one selects UHF/UMP2/UCCSD(T). Singlets retain their
existing restricted behavior, and one workflow can alternate multiplicities.
The existing native component ABI now admits energy-only UHF companions for
UMP2 and UHF/UMP2 companions for UCCSD(T), without changing descriptor layouts
or broadening any method's force capability. Python preserves spin-specific
component names, including the default HF companion.

Each group owns one physical UHF frame and exact interaction source. UCCSD(T)
evaluates the optional UMP2 companion before allocating CC workspace, using the
same UMP2 denominator threshold and generated equations as standalone UMP2.
The MP2 transform workspace does not overlap CC solver workspace. Endpoint
capacity diagnostics include the larger phase capacity. The MP2 diagnostic
and every output remain unpublished until the complete group succeeds.

Open-shell CBS rejects forces, CUDA and DF rather than substituting a restricted
or CPU calculation, finite differences, or a Python/reference implementation.
Native correction-basis admission remains all-electron s/p/d through twelve
AOs, with existing memory and scalar-work gates. This is useful bounded energy
support, not unrestricted force or GPU qualification.

The existing relaxed-convergence transport is closed-shell-only. Open-shell
calls strip that private transport and clamp common SCF/CC controls to native
strict ceilings, preserving tighter inputs. This is per-call state so a later
restricted call still uses the workflow's requested policy. Convergence metadata
reports this strict selection instead of pretending Standard/Fast relaxation ran.

## Rejected alternatives

- Relabel restricted MP2/CCSD(T): different spin equations and reference state.
- Independent same-basis calculators: duplicate SCF and risk branch mismatch.
- Synthesize Python HF results from correlation diagnostics: loses the native
  grouped transaction and overstates metadata available to Python.
- Advertise prototype unrestricted forces: the public native UCCSD(T) contract
  remains energy-only.

## Evidence

`tests/python/test_ccsdt_cbs.py` checks spin/charge transport, unchanged weights,
same-basis role reuse, explicit refusals, owner cleanup, and alternating spin
calls. `tests/python/test_cbs_open_shell_native.py` compares source-matched
independent PySCF UHF/UMP2/UCCSD(T) components and complete composites for a
neutral doublet and a charged empty-beta triplet. Its gates are 1e-8 Hartree
for Tight and 1e-7 Hartree for Standard. The deliberately small synthetic basis
pair tests algebra and backend behavior, not predictive CBS accuracy. Native
failure injection verifies untouched output descriptors and revoked companion
diagnostics on CC nonconvergence or an invalid MP2 denominator.

## Revisit when

The fixed twelve-AO cutoff was subsequently removed; see
[resource-based admission](../compatibility/2026-11-03-unrestricted-cc-resource-admission.md).
The energy-only, CPU/conventional restrictions below remain unresolved.

Native UCCSD(T) adds independently qualified forces, DF/GPU ownership, larger
domains, or reusable unrestricted MO blocks. Extend admission and tests together;
do not infer these capabilities from the restricted implementation.
