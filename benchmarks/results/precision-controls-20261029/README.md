# Native precision-budget controls qualification

PR #2869 moves the incumbent Fast budget-Fock control selection to the shared
native precision policy. Model qualification remains with the selected Direct
or fitted owner. Native thresholds preserve explicit caller ceilings; only a
missing additive protocol permits compatibility. This is a scoped #2227 R4/R6
slice, excludes R3 and does not complete the parent.

Candidate production source is `15d8894c90a5f00c7ab03ef3034c4bed796dd43f`,
compared with #2859 production `7adf42ceaab0fb32bb69232676fbfbba326d441e`.
Both CPU and CUDA matrices pass 1,792 complete calls, with identical scientific
work. Maximum median candidate/baseline ratios are 1.0051806418477298 (CPU)
and 1.0114521078447871 (CUDA), below the unchanged per-case 1.05 gate.
CPU energies/forces are exact; CUDA maximum differences are
1.4210854715202004e-13 Eh and 1.325606291402437e-13 Eh/bohr.
Calculator construction is inside scalar and complete fleet timing, since this
is where control selection executes. Source-isolated workers retain cold, warm,
moved and cleanup behavior. These matched-input regressions do not establish
a fastest production configuration or new scientific admission.

Slurm 14846 completed with exit 0 on node1/main/RTX 5090, in 20m20s of its
finite one-hour allocation. It preserved assigned device visibility `0`.
47 independent-reference CUDA tests passed with candidate compatibility poisoned.
Release CPU validation passed 129 CTests and 355 Python tests plus 25 subtests.
Installed public-header SDK tests ran with `-O3 -DNDEBUG` and no Python or compiler
on PATH; CUDA SDK execution was inside the same Slurm allocation.

The complete untimed profile has identical 94,042 kernels, 149 graph calls,
5,344 explicit device allocations and 118,503,293-byte dynamic peak. Final
explicit dynamic live bytes are zero; static storage is 1,823,689 bytes.
H2D, D2H and D2D counts/bytes are identical. `profile-summary.json` records the
scope and SQLite hashes; opaque provider/driver storage is outside this peak.

`identities.json` binds sources, binaries, exact raw samples, validation receipts
and unsuccessful host-test assertions. Original logs, XML and large profiler
databases remain in ignored local artifacts. Accepted CPU/CUDA samples and
readable qualification sources remain here. `qualification/run-gpu.sh` preserves
the executed environment and inputs; submit it with:

```bash
srun --partition=main --gres=gpu:5090:1 --nodelist=node1 --nodes=1 --ntasks=1 \
  --time=01:00:00 bash qualification/run-gpu.sh
```

Paths in that script identify the frozen experiment installations and must be
reconstructed for another machine. `qualification/qualify.py --help` lists the
source and installed-library arguments for CPU/CUDA replay; CPU used one thread
and affinity CPU 0, CUDA used the recorded Slurm affinity. The CPU grid/basis is
(12,6,12)/STO-3G and CUDA is (24,12,24)/def2-SVP spherical. The independent tests
use their own qualified reference settings, preserved at the measured source.
Existing precision-admission raw recovery uses the exact reviewed metadata from
the sibling #2864 stack, preserving the unchanged 80 MiB evidence budget.
