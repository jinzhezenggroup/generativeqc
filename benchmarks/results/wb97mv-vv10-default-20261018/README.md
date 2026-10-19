# Production default: bounded frozen-VV10 inner relaxation

This is the production-default follow-up to the
[original experiment](../wb97mv-vv10-inner-20261018/README.md). Base revision:
`e4aff7810b0462204831eb5168f19b6e217ef520` (master). Only the shared native
proposal-tolerance helper changes scientific execution. The experimental
selector is **not** added to production.

## Change and acceptance boundary

For already-admitted strict frozen-potential proposals, use
`clip(0.01 * maximum_residual, min(1e-12, 0.1 * min(1e-9, density_tolerance)), 1e-5)`
instead of factor `0.001`/cap `1e-7`. The maximum residual is the last complete
target's AO maximum, not the normalized history diagnostic. The lower floor,
initial semilocal stopping rule, budget-selected proposal/confirmation policy,
FP64 arithmetic, all physical acceptance gates, DIIS resets, finite iteration
limits and reserved ordinary full-target tail remain unchanged.

The original matched n4 RTX 5090 water-96 experiment reduced provisional Focks
`31 -> 23`, total SCF/endpoint Focks `36/36 -> 28/28`, keeping five target
Focks/four SCF VV10 pair sweeps. Single cold E+F observations were
`1165.548656 -> 995.149441 s`, or 14.6% lower. The more aggressive factor
`0.1` saved more inner steps but added a pair sweep and was slower. These
measurements used an experimental binary, not the default binary below.

## Actual-default qualification

The native source identity is
`4082c8548753ae72043983fc1d52291f4431f8a4e9fd2a8453cecde5c7bf178f`.
It matches the PR's inventory, independently recomputed from the source
archive and authenticated against the loaded library before either endpoint.
`evidence.json` retains library/raw/density/receipt hashes and all gates.
`probe.py` runs the actual default with all three experiment/seed override
variables absent, a fresh process/Calculator/owner/native density and one
target attempt. The reference is consumed only by the offline reducer.

Slurm job **1123**, n4 RTX 5090, actual default with no policy override:

| Atoms | Provisional / target Focks | SCF / endpoint Focks | SCF pair sweeps | Cold E+F |
| --- | ---: | ---: | ---: | ---: |
| 12 | 25 / 5 | 30 / 31 | 4 | 44.862294 s |
| 96 | 23 / 5 | 28 / 28 | 4 | 995.676682 s |

Seven preliminary native atomic Focks are accounted separately. The 96-atom
provisional stages are `9+4+5+5`, reproducing the conservative experiment's
work reduction. Against the retained master control, maximum energy/force/
density differences are `1.364e-12 Ha`, `8.258e-12 Ha/Bohr`, `8.509e-12`.
The 12-atom independent energy/maximum-force errors are `2.842e-12 Ha` and
`1.195e-11 Ha/Bohr`. All declared gates pass. The 13 device schedule tests in
job 1123 and 221 expanded tests in n1 job **10075** also pass. Local compiled
policy tests report 82 passed/8 GPU skips (the skipped tests run in job 1123);
documentation checks report 17 passed/55 subtests passed.

The fixture remains README water-12/water-96, spherical def2-SVP, Direct RKS,
FP64, grid `48/16/32`, energy/density/screening controls `1e-12/1e-10/1e-12`,
DIIS eight and maximum 100 iterations. It is not DF, def2-TZVPD or OMol25.
Complete endpoint timing includes preparation, SCF, forces and synchronized
host publication. It excludes imports, context setup, Calculator construction,
density export, JSON serialization and owner destruction. Compiler/JIT caches
are reused; no density/orbitals or prior owner are reused.

Acceptance gates are finite physical energy/force/density output, original SCF
stopping controls and consecutive full-target confirmation, no mixed/retry/
fallback work, paired master energy `<=1e-8 Ha`, maximum force
`<=1e-7 Ha/Bohr`, maximum density-element difference `<=1e-7`, and independent
water-12 GPU4PySCF energy/maximum-force errors `<=1e-8 Ha`/`<=1e-7 Ha/Bohr`.
The independent reference is retained, not reexecuted; its published binding
is `benchmarks/results/readme-wb97mv-20261014/summary.json.xz`, SHA256
`bc25a49be54429892af21521d1cf09f0b677d28bfe6d6fff7dc0f8f865b5c519`.
The source/default measurements are new, but comparisons use the original
same-revision control. They are **validation observations, not a new matched
timing campaign**. There is no independent water-96 oracle or general
ground-state/performance qualification.

Validation also includes the compiled CPU tolerance/budget/confirmation
contracts, n4 device schedule/physical-acceptance/warm/exhaustion tests, and
n1 independent fixed-density, finite-difference, nonlocal integration and
mixed-retry tests. The latter run with the actual new native library rather
than relying on an unavailable local library.

## Build and execution provenance

The default library is an incremental Release CUDA 12.9 build on n1, official
`sm_120`, `fast_compile=0`, `portable=0`, using verified sccache 0.16.0 and
retained compiler commands/before-and-after statistics. The original experiment
source is copied before the temporary build-header replacement, restored
byte-for-byte afterward and rebuilt. Original master/experiment libraries and
source snapshots are retained; no source/cache identity is spoofed. A separate
default source archive and library remain bound by their native identity.
The original post-build archive comparison noticed a newly generated Python
bytecode file and stopped the rebuild-after-restore step. That notice is
retained; a separate source-only comparison and successful restore rebuild
are recorded as well. The authenticated default library was already complete;
no scientific endpoint was retried because of this archival check.

n4 execution uses the original fingerprinted private glibc 2.39 loader/runtime
for the n1-built library. This is process-local; system libraries, compiler
subprocess runtimes and Slurm device visibility are not modified. Every real
GPU test/endpoint uses finite Slurm allocation; node3 is not used.

Full records remain under `.artifacts/wb97mv-default-20261018/` in the clean PR
worktree, with execution records on n4 and build/expanded-test records on n1 at
`/data/jzzeng/wb97mv-default-20261018/`. Original control/reference records remain
under the earlier `wb97mv-inner-20261017-mastere4aff7810` campaign. The compact
JSON is not a replacement for those raw histories and density arrays.

## Reproduction

Use this revision's authenticated library/source and the existing basis JSON.
The native library and portable compiler cache must be available; a baseline
binary or the experiment-only selector is not a substitute for this default.
On n4, within the retained runtime environment, choose a fresh output path:

```bash
srun --partition=main --nodelist=node4 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:50:00 \
  bash -c 'source /data/jzzeng/wb97mv-inner-20261017-mastere4aff7810/env.sh; \
    root=/data/jzzeng/wb97mv-default-20261018; \
    export GENERATIVEQC_LIBRARY="$root/library/libgenerativeqc.so"; \
    export PYTHONPATH="$root/source:$root/source/python"; \
    export LD_LIBRARY_PATH="$root/library:$LD_LIBRARY_PATH"; \
    unset GENERATIVEQC_EXPERIMENT_VV10_INNER_POLICY; \
    unset GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED GENERATIVEQC_VV10_NATIVE_SEED; \
    cd "$root/source"; \
    "$PYTHON" "$root/probe.py" --atoms 96 --source "$root/source" \
      --basis /data/jzzeng/wb97mv-inner-20261017-mastere4aff7810/def2-svp.json \
      --output "$root/recheck-job$SLURM_JOB_ID/result.json"'
```

Reduce only the original qualification population with the retained raw roots:

```bash
python benchmarks/results/wb97mv-vv10-default-20261018/analyze.py \
  --root .artifacts/wb97mv-default-20261018 \
  --historical /home/jzzeng/codes/qc-external-basis/.artifacts/wb97mv-inner-20261017-mastere4aff7810 \
  --job-id 1123 --output .artifacts/wb97mv-default-20261018/reduced.json
```

This offline reduction does not start a GPU job. It refuses missing endpoints,
nonzero scheduler return, failed gates or changed source/library/consumer bytes.
The retained result-directory scripts are measurement snapshots, excluded from
the repository's ordinary formatting/lint inventory along with other results.
