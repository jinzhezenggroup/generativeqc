# DF density preconvergence for an exact RHF reference

The native `df-ccsd(t)` owner automatically prepares a CUDA DF-JK density
guess behind compatibility, source-path, work and resource guards. The **accepted reference remains
conventional, unscreened FP64 RHF**, with correlation-only density fitting.
This does not select RI-HF, change the final Hamiltonian, or alter response
and gradient equations.

## Default admission and fallback

Automatic preparation is limited to neutral, closed-shell H/C systems with
spherical orbital AOs, through-f orbital shells and no ECP. There is no fixed
200–400 AO interval. These are compatibility limits, not a claim that every
admitted geometry/basis has demonstrated a speedup. The
independently bundled `aug-cc-pvtz-jkfit` H/C auxiliary records are normalized
by the native basis owner on the current geometry. They are not the caller's
correlation RI basis, and no Python/PySCF computation or lookup occurs in
production.

The shared exact-reference source policy first excludes cache-eligible
references: adding DF to a potentially resident four-center ERI path is not
assumed profitable. For the remaining Direct source domain, a conservative
**dense-volume heuristic** requires
`N_cartesian^2 / (32 * N_JK_auxiliary) >= 1`. It compares full four-center
Cartesian volume with the maximum provisional three-center sweep volume;
32 is the existing DF iteration cap, not a fitted AO threshold. This is not
an actual screened/symmetry-reduced integral count, hardware timing model,
or proof of a universal performance crossover. It ignores contraction lengths
and shell-class cost differences, so measured complete endpoints remain the
authority for profitability. Both dimensions and the ratio are exported by
the benchmark; changing either basis can change admission at the same AO count.

The preliminary solve uses FP64, one compact CUDA SCF attempt with at most
32 iterations, and `1e-4` energy and density tolerances. The ordinary DF
solver's additional host-orchestrated SCF retry is disabled for this guess;
nonconvergence returns to cold Direct RHF. Physical final-state validation
after convergence remains required and is separate from the SCF cycle count.
Its numeric budget is capped at 512 MiB, after
charging live correlation auxiliary metadata and any retained response cache;
less than 256 MiB available skips preparation. The shared DF preparation
storage estimator must also admit the actual Cartesian/public matrices,
primitive/shell metadata and JK auxiliary metadata under that cap. The native
source-backed dense/packed tile planner must first select full retained
three-center storage under the same metadata and lazy-DIIS reservations as
native SCF. A streamed proposal is skipped before factor generation: bounded
memory alone does not imply bounded work or profitable initialization. The native
DF tile/SCF planner remains authoritative for subsequent reservations and may
refuse the guess; preflight does not replace its complete
live-owner accounting. Its source and SCF owners are
destroyed before Direct SCF. Only the detached density remains, and its
**capacity** is reserved in every subsequent phase.

Cache-eligible, unamortized-work or streamed-plan cases, unsupported topology and tight budgets retain cold
Direct RHF. An explicit density seed takes precedence. Refused/nonconverged
DF preparation retains cold Direct; a failed seeded Direct solve invokes the
existing bounded cold retry, never acceptance after one exact Fock build.
Both successful and refused DF preparation are included in the public
reference timer and complete endpoint timer.

Set `GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS=direct` before execution for the
historical all-Direct control; unset it or select `auto` to restore the guarded
default. Other values are errors. This selector does not enable full DF-HF.
Prepared DF-CCSD(T) does not retain a warm density or support prepared batches;
re-execution and changed geometry rebuild the guess rather than reusing an
identity-sensitive DF owner.

Changing a guess does not prove uniqueness of an RHF stationary root. The
qualification compares final density and orbital energies as well as residuals.
Use the Direct opt-out for explicit root-continuity controls near degeneracies.

## Scientific boundary

`benchmarks/df_hf_preconvergence.cpp` can run native CUDA DF-JK RHF using a separate
JK-fit auxiliary basis and transfers **only its detached AO density** into a
fresh native Direct RHF solve. DF Focks, orbitals, orbital energies and DIIS
history are not transferred. The Direct solver retains energy tolerance
`1e-12 Eh`, density RMS tolerance `1e-11`, zero screening and FP64 precision.
It must converge normally and export its validated physical reference before
CCSD, triples, Lambda, Z-vector or nuclear response can execute.

DF preparation/nonconvergence refusal falls back to a cold Direct solve. A
failed seeded Direct solve likewise retains the existing cold retry. The
seed's retained capacity is charged beside the exact reference and downstream
owners. Reusing the internal benchmark diagnostic object requires releasing
its previous output reference. Production uses a call-local owner and retains
no fitted frame across calls.

An all-DF-HF reference is not implemented. It would require
its own method definition and consistent DF SCF, orbital response and nuclear
derivatives, including three-center and metric contributions.

## Reproduction

Build the Release native CUDA library with a verified compiler-cache launcher.
Then compile the experiment against that same checkout and library:

```bash
ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 \
  -Iinclude -Isrc -I"$CUDA_HOME/include" \
  -c benchmarks/df_hf_preconvergence.cpp -o build/probe.o
g++ build/probe.o build/libgenerativeqc.so \
  -Wl,-rpath,"$PWD/build" -o build/df-hf-preconvergence
```

Prepare JK-fit **basis metadata**, not an oracle density:

```bash
export PYTHONPATH=python:.
python -m benchmarks.df_hf_preconvergence prepare-jk \
  --input benchmarks/results/df-lambda-gemm-20261004/ethane230.input \
  --basis aug-cc-pvtz-jkfit --output .artifacts/df-hf/jk.shells
```

The retained geometry has 230 spherical orbital AOs, 488 correlation auxiliary
functions and 484 JK-fit auxiliary functions. Run both commands inside the
same finite Slurm allocation, preserving Slurm's `CUDA_VISIBLE_DEVICES`:

```bash
build/df-hf-preconvergence \
  benchmarks/results/df-lambda-gemm-20261004/ethane230.input \
  .artifacts/df-hf/jk.shells .artifacts/df-hf/direct.json direct forces
build/df-hf-preconvergence \
  benchmarks/results/df-lambda-gemm-20261004/ethane230.input \
  .artifacts/df-hf/jk.shells .artifacts/df-hf/seeded.json df-direct forces 1e-4
build/df-hf-preconvergence \
  benchmarks/results/df-lambda-gemm-20261004/ethane230.input \
  - .artifacts/df-hf/default.json auto-direct forces
```

Endpoint choices are `hf`, `energy` and `forces`. The optional final argument
bounds preliminary DF iterations (default 50). A bound of one exercises the
cold fallback rather than publishing an unconverged reference. Nonfinite
diagnostics of refused preliminary iterations are serialized as JSON null.

## Audit and timing

The output retains density and orbital energies for gauge-independent
comparison, actual Direct SCF and post-SCF Fock counts, physical commutator,
canonical density drift, generalized eigen residual and `C^T S C` error.
Degenerate orbital coefficients are deliberately not compared elementwise.
DF SCF cycle count is reported when `pre_work_counters_complete` is true;
exceptions with unreported work emit null cycles. Both automatic and explicit
preliminary guesses use a single compact SCF attempt. A complete DF physical-Fock
census is not available and is explicitly null. Refused/retried Direct counts are not
reconstructed from the successful attempt's iteration count.

`rhf_seconds` includes all DF preparation/SCF/refusal and Direct solve work.
`endpoint_seconds` includes context initialization, DF work and the complete
cold requested native endpoint through host-returned forces. Input parsing,
post-timing orthogonality audits and JSON serialization are outside that timer.
`native_seconds` measures only the native correlated owner: it includes that
owner's automatic DF guess, but excludes context initialization and an explicit
`df-direct` guess prepared outside the owner. It is not the complete cold endpoint
and must not replace `endpoint_seconds` in acceleration claims.

Run `oracle` for a fresh independent PySCF conventional RHF and `summarize`
for all-repeat-pair gates. Supply `--correlation-oracle` and
`--finite-differences` together to also audit the retained independent
same-Hamiltonian CCSD(T) energy and two-step directional forces:

```bash
python -m benchmarks.df_hf_preconvergence summarize \
  .artifacts/df-hf/direct.json .artifacts/df-hf/seeded.json \
  --oracle .artifacts/df-hf/rhf-oracle.json \
  --correlation-oracle benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json \
  --finite-differences benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json \
  --output .artifacts/df-hf/summary.json
```

Full-component force agreement with the Direct control is not an independent
analytic force oracle. Directional finite differences and physical Lambda/Z
residuals provide the additional independent gates. Native integration and
preliminary-failure tests require `GENERATIVEQC_DF_PRECONVERGENCE_BINARY` and
the explicit `GENERATIVEQC_DF_PRECONVERGENCE_CUDA_TEST=1` opt-in inside Slurm.

`auto-direct` exercises the actual native default owner with fixed preliminary
controls and no exported JK input. `direct` explicitly suppresses automatic
preparation, while `df-direct` preserves the original configurable experiment.
Standalone `auto-direct hf` exercises the same density-preparation policy and
strict exact reference, but does not qualify correlation or force performance.
The `pre_policy_outcome` diagnostic distinguishes compatibility/cache/work/budget
skips from a used or failed preparation. JK auxiliary dimensions in a skipped
record describe the proposed metadata shape, not executed integral work.
`pre_resident_plan_peak_bytes` describes the previewed value-plan workspace,
including a streamed proposal that is rejected; it is not a measured allocation
or proof that the rejected proposal has resident factors.
Additional qualification inputs can be reproduced without an energy/SCF oracle:

```bash
python -m benchmarks.df_hf_preconvergence prepare-case \
  --case propane322 --output .artifacts/df-hf/propane322.input
python -m benchmarks.df_hf_preconvergence prepare-case \
  --case moved --output .artifacts/df-hf/moved.input
python -m benchmarks.df_hf_preconvergence prepare-case \
  --case budget8 --output .artifacts/df-hf/budget8.input
for case in methane34 ethane58 ethane144 butane414; do
  python -m benchmarks.df_hf_preconvergence prepare-case \
    --case "$case" --output ".artifacts/df-hf/$case.input"
done
```

The larger case uses raw orbital/RI basis metadata exported offline from PySCF.
The moved case shifts the first hydrogen by `0.02 bohr` along x; the budget case
changes only the declared numeric budget to 8 GiB, not measured VRAM.
Performance evidence and promotion limits are retained in the
[work/resource decision](../../.agents/notes/implemented/performance/2026-10-09-df-rhf-work-admission.md)
and the historical
[initial default-policy decision](../../.agents/notes/implemented/performance/2026-10-09-df-rhf-preconvergence-default.md).
Do not extrapolate this guarded policy to all bases/elements/backends or to
full-DF-HF gradients. Expand admission only with fresh independent gates and
paired complete endpoint evidence.
