# Decision: prepare reproducible joint CUDA schedule trials, not guessed defaults

Status: implemented; source-trial tooling only, GPU qualification outstanding
Date: 2026-10-06

## Problem and scope

A CUDA parameter change is not necessarily a launch-only edit. Stationary
geometry thread count, pair rows and scratch limits are shared between Python
resource planning and the generated native admission constants. Grid tile size
also affects ordinary force resource selection, actual execution and active-AO
maps. Raising a constant at only one boundary can make the candidate invalid,
select an unrelated fallback, or lose the identity of the code being measured.

The user's requested audit covers Direct force CTA, grid tiles/budgets,
stationary Becke resources, XC matrix tiles, shared-memory admission and generic
launch widths. Direct static-128 promotion already has PR #2026. PR #2029 repairs
launch-resource assessment, including the static/dynamic shared-memory distinction.
This tool covers the remaining concrete schedule experiments without duplicating
those PRs or treating historical dense-grid timings as current sparse-grid gains.

## Implementation

`tools/cuda_schedule_trials.py` accepts a strictly validated JSON trial and
creates a detached worktree from a resolved commit. It does not alter the
invoking checkout, run GPU commands, create commits, publish evidence, or choose a
winner. The incumbent trial produces no source changes. Only nominated schedule
expressions are changed; scientific formulas, cutoffs, source coefficients and
warp ownership constants are not blanket-rewritten.

Supported source axes are geometry and cooperative thread widths 32/64/128,
Becke pair rows 2/4, geometry scratch 8/16 MiB, ordinary force tiles
256/512/1024 with explicit host/device budgets, XC matrix tiles 8/16/32, and the
physical-PBE point launch width. Other point consumers retain their current
width. The finite candidate listing includes separate stages and an explicit
joint Becke-128/rows-2/scratch-16/grid-1024 candidate; it is not a performance ranking.

Geometry edits target the compiler-owned constants only after checking their
imports/use in the native-layout emitter and the corresponding native resource
symbols. Fresh source generation therefore sees the same configuration as the
resource planner. Grid edits bind the requested tile in both the ordinary
workload and the actual force kwargs; composite policy is untouched. AST byte
spans preserve non-ASCII text, comments and CRLF. Unexpected source shape,
unknown options, ambiguous anchors, dirty tracked checkouts and existing outputs
fail rather than partially rewriting the invoking checkout.

The detached worktree retains a complete Git patch and a manifest with the base
commit, all read source hashes, changed-file before/after hashes and patch hash.
Build, numerical and complete-endpoint validation remain explicitly `not-run`.
Raw worktrees cannot be written into reviewed benchmark result destinations.

## Usage

From the repository root:

```bash
mkdir -p .artifacts/cuda-trials
python -m tools.cuda_schedule_trials --list-candidates > .artifacts/cuda-trials/candidates.json
python - <<'PY'
import json
from pathlib import Path
root = Path('.artifacts/cuda-trials')
trials = json.loads((root / 'candidates.json').read_text())
(root / 'joint.json').write_text(json.dumps(trials['joint-becke128-rows2-scratch16-grid1024']))
PY
python -m tools.cuda_schedule_trials --trial .artifacts/cuda-trials/joint.json --output /tmp/generativeqc-cuda-joint
```

Build the **detached** source with the repository's normal CMake configuration,
verified `ccache --version`, explicit CXX/CUDA cache launchers and before/after
cache statistics. Preserve `CCACHE_BASEDIR` and exact final source/library hashes.
Run CUDA qualification and benchmarks only in a finite Slurm GPU allocation,
without changing assigned device visibility. This preparation command does not
schedule or perform that work.

## Required device acceptance before any production-default PR

Retain the same mathematical grid, basis, method, precision and numerical gates;
record actual AO-map selection/work and all other selected routes, not merely the
requested trial. Larger tiles can change AO-map unions or select a dense fallback;
byte-budget changes are separate resource regimes, not free speedups. Reconcile
those changes rather than silently calling the trials identical-work experiments.

Record final-kernel registers, spills, static/dynamic/driver-reserved shared
storage, thread/grid dimensions and actual occupancy counters when available.
Do not infer measured occupancy from the 606-lane arithmetic or assume that
128 threads imply 50% occupancy. Static shared arrays cannot use opt-in capacity
just because a tuning ceiling is raised; this tool deliberately refuses such an
axis. A native dynamic-memory consumer must obtain and record its granted limit.

Require independent energy/full-force checks, tails, both spins, changed geometry
and sanitizer coverage, followed by clean 48/96-atom cold, at least five warm,
moved and at least five moved-warm calls. Use genuinely interleaved comparisons
for promotion, retain all negative samples and actual SCF/Fock histories, and
keep intrusive profiling separate. Default-sensitive golden tests and capacity
contracts must be reconciled explicitly for any eventual production change;
this tool never weakens them or rewrites their hashes.

## Tests and limitations

Executed locally: 64 Python tests passed, including real local Git worktree
creation, unchanged caller checkout, patch/manifest hashes, atomic preflight
rejection, Unicode/CRLF, compiler/native coupling, scoped point/matrix changes,
ordinary policy/execution parity and unchanged composite behavior.
One additional test checks all source anchors against a complete repository
checkout and is left for repository CI; this environment could retrieve sources
through GitHub but could not clone the repository into the test container.
No NVCC, GPU or ccache is available locally. No CUDA compilation, numerical
qualification, speedup or promotion is claimed by this source-tooling PR.

Agent: ChatGPT
Model: GPT-6 Astra Pro
