# Structured SCF endpoint qualification

This family retains separate primitive, complete-endpoint and matched-performance
receipts. Its overall status is **PARTIAL**: generated defaults remain unchanged,
`production_library_qualified` is false, and none of #1877/#1878/#1883 is closed.
Passing an admitted endpoint does not qualify unsupported methods/bases, nor does
preparing a cuBLAS binding prove that it executed.

## Follow-up source

The endpoint follow-up is based on master `140160231`; the measured scientific
source is `7d8f4ba22883f9c3accfec70a3e8640510ffbe2c`, with scientific
inventory `9aa64b6d646babde52e5420799f377c01850624e5043f382ef4c2b011d2937f3`.
Receipt-only changes do not change that scientific inventory. This is a follow-up
to merged #2263, not a replacement of its primitive qualification.

- [Scientific receipts](endpoint-science.json) retain every cold/warm/moved
  outcome, independent error gate, failure, source/binary/reference hash and
  accepted-row count. Batch members are not independent process repetitions.
- [Matched performance](endpoint-performance.json) retains three paired process
  costs for each of five phases, SCF iteration/Fock-build observations, actual
  submitted operations, region policy, and negative/unknown profitability.
- [Validation and execution](endpoint-validation.json) records native-suite
  status, primitive replay, memcheck/initcheck, host checks, build provenance and
  transform envelopes. A requested Graph is not an admitted/executed Graph.

The original post-fix endpoint source `607d47b1e2383bdd7618ddc9264f9d7a8205d6b2`
was based on `bd6f78288`, before rebasing onto merged #2263. Its compact
[outcome/decision/validation history](endpoint-history-607d47b1e.json) is historical
evidence, not validation of the new tip. The new source is rebuilt and rerun independently;
no old measurements are relabelled with its SHA.

### Recorded results for `7d8f4ba22`

| Check | Result |
| --- | --- |
| Complete scientific processes | 112: 88 PASS, 16 FAIL, 8 ERROR |
| Scientific rows | 900 retained; 816 accepted, including moved rows from rejected processes |
| Independent references | 72 accepted PySCF/Libxc geometry references |
| Matched performance processes | 127: 120 PASS, 7 ERROR |
| Profitability comparisons | 22: 11 negative, 11 unknown; none admits promotion |
| Primitive arithmetic/replay | 1,680 scenarios, two executions each, PASS |
| Memcheck/initcheck | Eight runs, zero errors |
| Native GPU suites | Seven PASS; KS CUDA final-state suite FAIL |
| Host/type checks | 530 tests and 48 subtests PASS; compiler ty 0.0.82 PASS |

Maximum errors among accepted rows are 7.674e-13 energy, 3.542e-8 density,
7.363e-8 Fock, and 4.125e-8 forces. HF retained library bindings execute in resident
Graphs. The MD-disabled RKS observation executes all four library basis stages
on the 48-public/50-direct domain; its span envelope is 292/4,800. That KS trace
records stream submissions, not Graph admission. These observations establish
executed adoption, not profitability or full-method/default qualification.

JSON publications use losslessly expanding tables to respect the repository
retention budget: combine `defaults` with each `columns`/`rows` pair, then remove
fields identified by zero-based `absent` indices. Scientific phase arrays follow
`phase_order`. Performance identities combine `identity_defaults` and the profile
identity; `model_key` resolves model, reference and artifact facts from the
scientific receipt. Observations borrow that identity with a zero-based
`profile_index`. The final report preserves every paired cost, work count and
policy decision. The historical publication retains outcomes, decisions and
raw receipt/report hashes; its original phases, errors and timings/work
remain in the recorded ignored local directory, not a new archive or Release.

## Admission boundaries

The sweep covers RHF/UHF/RKS/UKS, generated/diagnostic-library offers, one/three
systems, Cartesian/spherical through-d and def2-SVP bases, and changed geometry.
Separate through-f and def2-TZVP fixtures preserve negative complete-force
evidence. Independent references use PySCF 2.14.0 and Libxc 7.0.0; KS forces
reconverge the moving basis and explicit moving Becke grid at steps 1e-3 and 5e-4.
Energy/density/Fock/force gates remain 1e-8/1e-6/1e-6/3e-5. Finite values, shapes,
convergence, CUDA execution and valid physical final states are mandatory.

Large UKS convergence/final-state failures and unsupported KS f-shell forces
remain rejected. The native KS CUDA suite also reports a final-state failure;
there is no qualified matched-master baseline establishing that failure's origin.
These failures are not converted into success, CPU fallback, or profitability.

HF keeps determinant-consistent `P F(P) P` Pulay weights: its signed Gram offer
does not execute and cannot earn profitability. Ordinary semilocal KS may use
MD-J instead of the four Direct basis stages. The separately matched
`--disable-md-j` diagnostic exercises those shared stages without changing the
production value schedule. Provider controls participate in timing identity.
Dense diagnostic stages do not remove the bounded shell-span production route.

## Reproduction

Check out the recorded scientific SHA, install the pinned reference environment,
verify ccache, and configure a CUDA test build with explicit CXX/CUDA ccache
launchers. Build `generativeqc_structured_scf_endpoint_probe` and both public/spd
PBE RKS/UKS stationary companion **manifest** targets; building the main library
alone does not refresh the optional force binaries. See the
[current contract](../../../docs/developer/lowering_providers.md#borrowed-structured-scf-regions)
for target names and ownership.

Generate independent CPU references before starting the GPU allocation:

```bash
python benchmarks/structured_scf_endpoint_qualification.py oracle \
  --case large-spd-spherical --method rks --output "$PWD/.artifacts/oracle/rks"
```

Run clean endpoints and a separate instrumented observation through finite Slurm
allocations on permitted nodes, preserving Slurm's device visibility. For example:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:10:00 bash -lc '
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  for repeat in 0 1 2; do
    for mask in 0 127; do
      python benchmarks/structured_scf_endpoint_qualification.py device \
        --case large-spd-spherical --method rks --batch 3 --mask "$mask" \
        --repeat "$repeat" --graph --disable-md-j \
        --library "$PWD/build/cuda-release-sm120/libgenerativeqc.so.0.1.0" \
        --driver "$PWD/build/cuda-release-sm120/generativeqc_structured_scf_endpoint_probe" \
        --reference "$PWD/.artifacts/oracle/rks" \
        --output "$PWD/.artifacts/endpoint/rks-mask$mask-repeat$repeat"
    done
  done'
```

For execution observation use an independent output, add `--instrumented`, and
set `GENERATIVEQC_DF_TRACE` to its `trace.jsonl`; retain its sibling receipt.
Pass only clean repeat receipts to `structured_scf_endpoint_report.py` and supply
observations with `--traces`. Qualification used reversed mask order on odd
repeats, masks 0/1/2/4/120/127 for large HF, and 0/120/127 for the independent
MD-disabled RKS schedule. Run memcheck/initcheck with `--target-processes all`,
`--error-exitcode 99`, and separate `--log-file` outputs so HF child JSON remains
parseable. Never treat instrumented timings as clean endpoint costs.

## Primitive foundation

This compact [receipt](qualification.json) qualifies primitive arithmetic,
resource facts, masks and replay for scientific source commit
`8f09a10bb8a6228394f6d8cde4168d963fbac646`. It is **not** complete scientific
endpoint or default-route qualification, and closes none of #1877/#1878/#1883.
Subsequent receipt-only changes do not change the qualified scientific source.

The independent diagnostic driver checks generated and cuBLAS routes against
long-double coordinate loops over 1,680 scenarios, twice per scenario. Captured
graphs replay with changed data/occupations/masks at the same device addresses.
Cases cover positive and signed Gram, inverse-root reconstruction, rectangular
basis stages in both shape directions, spin broadcast, sparse shell spans,
hcore publication, poisoned virtual orbitals, invalid eigenvalues, inactive
byte preservation and library scratch alias/capacity rejection. Memcheck and
initcheck each report zero errors. The receipt records the numerical acceptance
threshold; no endpoint timings or speedups are inferred.

Reproduce from the qualified commit on an allowed GPU node using a fresh output:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:10:00 \
  bash tools/qualify_structured_scf_routes.sh \
  "$PWD/.artifacts/structured-reproduction" /group/software/cuda-12.9.1 \
  120 /home/jzzeng/miniconda3/bin/python
```

The script requires verified ccache and clean committed source, records build
commands/dependency files, generated-header/object/binary hashes, compiler/cache
versions and statistics, loaded libraries, source provenance and Slurm device
visibility. `build-only` performs compilation without device execution;
`run-only` checks the recorded source identity before reusing the binary.
Run both `compute-sanitizer --tool memcheck --error-exitcode 99` and
`compute-sanitizer --tool initcheck --error-exitcode 99` against the resulting
driver inside a finite Slurm allocation. Do not alter `CUDA_VISIBLE_DEVICES`.

Raw objects, binary, full dependency/command/device logs and sanitizer receipts
remain in the ignored local directory named by the compact receipt. No archive,
Release, release asset or external backup is published for this qualification.
The [current contract](../../../docs/developer/lowering_providers.md#borrowed-structured-scf-regions)
and [decision note](../../../.agents/notes/implemented/architecture/2026-10-11-borrowed-structured-scf-regions.md)
describe ownership and the remaining default-promotion gates.
