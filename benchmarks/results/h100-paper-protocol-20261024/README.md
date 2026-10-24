# H100 paper protocol (issue #2725)

This family owns the offline protocol, not device execution. `manifest.json`
uses `generativeqc.h100-paper-protocol.v2`. Consumers identify an input by
`cells[].id`, an arm by `cells[].arms[]`, and a process by `(cell, arm, repeat)`
with repeat 0, 1, 2. `schedule` is the normative rotated/interleaved order.
`input_sha256` is SHA-256 of UTF-8 sorted-key compact JSON (`allow_nan=False`)
of `input`; it is not a working-copy file hash. `sources` and `references`
bind immutable Git OIDs, paths, decoded selectors and raw blob hashes.

Each future attempt is an append-only receipt with schema
`generativeqc.h100-paper-attempt.v2`; `protocol.py` supplies validation and
`attempt_template`. A retry uses a new attempt ID and retains every failure.
No receipt is a launch request. Empty attempts mean no observation in this
family; they do not establish global NOT_RUN. Unobserved quantities are null
with an explicit UNAVAILABLE status, never zero. The evidence-map family
consumes this interface and must not redesign the source/input/gate identities.

The core has three CBS systems (C2/QM7 1, C4/48, C7/5604), each with Standard
and Fast, and eight HF/DFT input cells (water48/96 × PBE0 Direct/DF,
WB97M-V Direct, r2SCAN-3c DF), each with Standard, Fast and GPU4PySCF.
Three independent processes per arm give 90 processes: 18 CBS cold calls
and 72 HF/DFT processes each with one cold and three warm calls (288 calls).
Within-process warm calls are not three independent process repetitions.

The prepared v1 snapshot remains immutable at `df5c44a81c5d73210a8d2f7f09ba4930d7f67210`:
69,646 Git-LF bytes, SHA-256 `dd96c17e702101d21352deb4f867ab0fbb7bce66700222d49fc67bbdef23185e`.
V2 adds the root-authorized CBS schedule-port cohort and its build/preflight
bindings; `cells`, `sources`, `references`, `schedule` and execution caps remain
identical to prepared v1. Existing consumers keep their prepared bindings;
adopting v2 is an explicit addendum, not an overwrite of historical contracts.

## Reproduce offline

From the repository root, using Python with the pinned Git objects available:

```text
python benchmarks/results/h100-paper-protocol-20261024/protocol.py verify
python benchmarks/results/h100-paper-protocol-20261024/test_protocol.py
python benchmarks/results/h100-paper-protocol-20261024/protocol.py attempts --input <attempts.json> --require-complete
python benchmarks/results/h100-paper-protocol-20261024/protocol.py cost --input <pilot.json>
```

`build` emits the frozen manifest to stdout. `verify` independently rebuilds
the manifest from original Git LF blobs, authenticates source patches in
temporary directories, verifies reference bindings and exact selected geometry,
and reports device science separately. It needs no compiler, CUDA or PySCF.
Missing Git objects are errors; the reader never fetches or executes an oracle.
All admission uses explicit exceptions and remains active under optimized Python.
Historical assertion-based verifiers must run without optimization.

`python benchmarks/results/h100-paper-protocol-20261024/replay_history.py`
recomputes the selected 24 historical HF/DFT populations (96 samples) using
the existing host qualifier, then invokes the delivered #2721 verifier at its
pinned revision. This command uses NumPy and the repository's host Python
dependencies, but performs zero new scientific calls. It also prints the
source-authentication and reference-availability inventory for reproducibility.

The attempt reader validates the receipt contract. Its complete state is
`COMPLETE_RECEIPT_CONTRACT`, **not scientific acceptance**. Hash-bound independent
raw replay must establish every declared gate result before a paper table can
use an attempt. `timing_statistics` recomputes cold median/range from three
processes and retains three separate warm arrays; the warm aggregate uses the
three process medians. Failed attempts remain in the input ledger and are not
turned into timing samples. There are no RTX 5090 repeat statistics inferred
from its single historical CBS observation.

## Source comparability

| Family | Measured scientific source | Reconstruction | H100 requirement |
| --- | --- | --- | --- |
| HF/DFT | `cbd4d8074de35105717c8aac7cd1f86babbe90ec` | Base `f5ca5398f25ed7b31285367b5b8fd829d3b024b5` + original provenance patch, decoded SHA `3ebd93ce481b47058e20990f99f11a1c9d349d6780cc5a63286583bbd3316041`; three changed LF files equal measured commit | Rebuild sm90; old DSO `d3f883dc…` is provenance only |
| CBS | Base `976f756b391bc65edb79f76df1a8c3fe807518a7` + qualified patch | Decoded SHA `5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`; all six retained candidate file hashes match | Rebuild sm90; old DSO `65a401ed…` is provenance only |
| Later version | Current master `72e0a7e5…` contains later WB changes | Separately versioned paired cohort, only after root review | Cannot replace frozen HF/DFT and retain hardware-only label |
| Optional Lambda | Separate source family | Authenticate only when its trigger is met | Never inherit the CBS core identity automatically |

The frozen CBS resolver at `976f756b` rejects `auto/sm90` when there is no
tuned/compatible profile. Its existing explicit `portable_cuda` path resolves
the generic fallback. Root's 2026-10-24 PR #2742 handoff authorizes only the
task-owned build configuration `portable_cuda/sm90` for the same frozen
scientific implementation. This is cohort
`cbs-frozen-science-portable-cuda-sm90.v1`: **target, profile and generated
schedule/artifacts differ** from historical sm120 tuned execution. It is not
an identical-tuned-schedule or strict hardware-only comparison. No source,
parser, default or runtime edit is authorized by this decision. If the legal
portable path fails or needs runtime/scientific changes, stop the affected
cohort and report to root/production owner for a separately approved paired
version; other source families can continue independently.

`execution_contracts[family][arm]` identifies each execution cohort separately
from the scientific source. Each v2 attempt binds its canonical SHA-256 plus
the canonical `profile_decision` SHA-256. A successful receipt additionally
contains `runtime.build_preflight` and its canonical SHA-256. This preflight
binds the source-object SHA-256, target, environment and new library SHA-256,
actual build-config and generated-artifact inventory SHA-256, and profile
resolution receipt SHA-256. CBS requires observed `native_profile: portable_cuda`,
`profile_match: portable`, `tuned: false` and the retained `auto_sm90_outcome:
ValueError`. Preserve the original failure log and explicit portable resolution
as raw evidence. HF/DFT profile resolution remains UNAVAILABLE until its own
preflight; GPU4PySCF's native profile is `not applicable`. Neither inherits CBS
qualification. These fields admit a receipt contract, not authenticate raw
observations or authorize launch. Independent replay must verify the referenced
configuration, generated inventory, resolution log and DSO bytes.

Historical GPU4PySCF reports Python 3.13.9 / NumPy 2.4.6; the prepared qz layer
uses Python 3.11.16 / NumPy 2.2.6. Record versions per engine, including the
actual contraction dependencies. Host, package and schedule differences remain
confounders. CPU configure/build permission is distinct from SDK acceptance,
device/scientific PASS and root's GPU scheduling authority.

`7522994e…` is an evidence archive, not the measured implementation. The
HF/DFT patch/commit comparison permits only the additional Agent Note delta.
Its historical full 5,852-file source-manifest is UNAVAILABLE here; the retained
claim and manifest hash do not reconstruct that missing original. This does
not prevent authentication of the accessible measured Git tree and patch.

Frozen source drivers are immutable. The old CBS CLI requires Slurm; qz must
not fabricate `SLURM_JOB_ID`. A reviewed task-owned transport adapter may call
the original scientific functions while recording its own `driver_sha256` and
the original `source_driver_sha256`. This does not authorize runtime/science
changes. A compatibility change to production creates a new paired cohort and
must be reported to the root coordinator before proceeding.

## References and gates

HF/DFT reuses the original 40 independent oracle receipts, selected by exact
method and Bohr geometry, plus the original auxiliary tables. Spherical
def2-SVP is used except r2SCAN-3c's def2-mTZVPP; charge/spin are zero.
Every cold/warm observable must pass E ≤ 1e-5 Eh/atom, Fmax ≤ 1e-4 and
Frms ≤ 3e-5 Eh/Bohr. Independent denser-grid and CPU/GPU checks use one
quarter of these margins. Preserve full moving-grid response and independent
D4+gCP corrections. DF approximation is included against the direct target;
different DF metric policies or initial guesses are not equal-work claims.

CBS uses the pinned QM7 dataset, all-electron spherical cc-pVDZ/cc-pVTZ,
shared JKFIT for RI-JK and correlation, full RI-JK response, HF exponent 1.63,
MP2 exponent 3 and DZ CCSD(T)-MP2 correction. The 34,359,738,368-byte budget
is a numeric capacity contract, not physical VRAM or a measured memory peak.
Fast-vs-Standard composite gates are 1e-5 Eh / 1e-4 / 3e-5 Eh/Bohr;
constituents divide these allowances by the expanded coefficient L1 norm.
Independent constituent/composite energy gates are 5e-8 Eh. Two-step
directional forces use 1e-4/3e-5 Bohr and 5e-6 Eh/Bohr; translation uses 1e-6.
C4/C7 directions use seed 709 and the merged #2721 raw receipts; central
ENERGY reuses #2685. C2 uses its actual archived vector, not the C4/C7 seed.
C2's 24 original displaced energies remain UNAVAILABLE (#2705); never infer
them from a stored projection. These gates do not establish full-vector
alkane accuracy or physical CBS-limit accuracy. #2679 is already aggregate-closed;
#2721's external merge without formal APPROVED remains an explicit exception.

Before timing, #2727 replays the four archived water-algebra groups
(exact/correlation-only vs full RI-JK × original/displaced geometry). Each has
two Standard/2048 MiB, two Fast/2048 MiB and one Fast/256 MiB endpoints:
20 total. Keep the original Standard fallback gates 1e-8 Eh and
Fmax/Frms 3e-6, and Fast composite/constituent gates above. These synthetic
STO-3G/def2-SVP pairs are algebraic tests. The separate physical H2 fixture
retains 1e-9 Eh / 2e-6 Eh/Bohr. #2719 only delivered conventional/exact CPU
H2/water; DF/RIJK/CUDA remain UNAVAILABLE. H2 has no nonzero triples.

## Finite execution and first wave

The root coordinator owns resource authorization and launches. This family
does neither. Before any launch recheck actual owner/job state, source and
environment identity, resource quota/priority, input/gate readiness and cost.
Use one allocated H100, preserve its visibility, never parallelize competing
timing processes on it, and do not run new CPU references unnecessarily.

1. #2726 authenticates source/environment and its separately authorized tiny
   smoke; #2727 completes supported correctness/fallback fixtures. The tiny
   smoke is a separate non-timing qualification slot, excluded from the 90 core
   processes/306 calls. Original HF/DFT CLI rejects `--warm-repeats 0` and has
   a Slurm guard: it is not a runnable three-call qz smoke command. Use a
   root-approved task-owned adapter for the original functions with independently
   retained adapter/source-driver hashes and no scientific implementation edits.
2. Pilot C2 (#2728) and water48 PBE0 Direct (#2731) one cold call per arm,
   gated before expanding. Pilot measurements are diagnostics, not replacements
   for independent formal processes. HF/DFT formal pilots must include the
   complete cold+three-warm lifecycle before estimating that process's cost.
3. Pilot each remaining cell/arm stepwise; recompute the cost worksheet, then
   obtain root authorization for that bounded expansion. Advance C4 (#2729),
   water48 WB (#2732)/r2SCAN-3c (#2733), then the respective water96 cases;
   C7 (#2730) follows successful smaller CBS and its own pilot.
4. Follow each cell's rotated/interleaved schedule, using three independent
   processes per arm. A job holds at most three processes serially, process
   cap 3,600 s plus 30 s termination grace, job cap 14,400 s. No automatic retry;
   at most two explicitly authorized attempts per slot. A source mismatch,
   nonfinite value, failed science, OOM, timeout or budget refusal stops that
   affected cohort and retains its raw failure. Oversized pilot projections
   require root review rather than silently raising caps.

The worksheet input maps each measured cell ID to its measured arms, each containing
`process_wall_seconds`, `setup_seconds`, `price_per_gpu_hour`, `status: PASS`
and `receipt_sha256`. Partial pilots produce costs only for observed arms;
unmeasured arms remain UNAVAILABLE and the full-matrix total is null. A worksheet
never requires launching the whole pilot matrix first. There is no populated
H100 cost or speed estimate.
The proposed cap is ceil(2 × observed process wall + setup + 30), limited by
the hard process cap. Cost includes the pilot and up to six capped formal
attempts per arm, with termination grace and setup charged once per bounded job.
It computes the admitted serial batch size and number of jobs under the job cap.
Add actual CPU/build/retention/queue
cost separately; a job also must fit setup, serial process caps and grace.
Actual priority, price and time limits are re-read at submission.

Conditional-only tasks: #2734 VV10 reuse needs the corresponding core WB
observation and authentic same-binary ablation inputs; the nine old missing
originals (#2706) block only their historical replay. #2735 moved geometry
requires completed stationary correctness and an explicit lifecycle question.
#2736 Lambda requires core profiling that identifies its bottleneck plus its
separate source contract. #2737 C7 memory sweep requires successful main
32 GiB C7 and evidence of capacity sensitivity. #2738 empty/primed caches
requires a measured cold compile question and separate cache cohorts. Each
needs root scheduling approval; none silently expands the first wave.

Record actual precision/fallback/refinement, SCF/Fock/action counts (missing =
UNAVAILABLE), thread pools, host/affinity/cgroup/RAM, GPU/driver/power/clocks,
Python/CUDA/package/native/AOT/cache identities, constructor-to-host-E+F and
process wall separately. Compiler startup, prebuilt AOT and oracle work have
different timing boundaries. Diagnostics remain separate from clean timings.
Cross-machine ratios are system observations until host effects are controlled.
No positive speedup is required. #2739 consumes the frozen identities and
independent raw replay; it cannot promote this offline protocol into H100 PASS.
