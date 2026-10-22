# Independent full-DF CBS energy qualification

Refs #2679. **INCOMPLETE: C1, C3 and C4 pass; C5–C7 references remain in progress.**
This is an energy-only slice. C4/C7 two-step directional-force and translation
checks, and optional C3 full-vector forces, remain outside this delivery. The
parent issue must remain open.

## Protocol and independence

The protocol is all-electron, spherical cc-pVDZ/cc-pVTZ DF-MP2, exponential HF
extrapolation with exponent 1.63, MP2 exponent 3, and the DZ DF-CCSD(T)−DF-MP2
correction. RI-JK RHF and correlation share cc-pVDZ-JKFIT/cc-pVTZ-JKFIT, matching
the frozen native full-DF reference/response model. No force or response is
recomputed by this energy-only oracle.

The original [QM7 dataset](https://www.quantum-machine.org/datasets/) is
17,884,280 bytes, SHA-256
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
Indices 0, 8, 48, 183, 293 and 5604 retain the exact original float32 Bohr values.
The complete dataset was independently downloaded and checked locally and
remotely; all six extracted geometries equal the native protocol records.
`inputs.json` contains no native energy, density, orbital or amplitude seed.
The measured runtime JSON used CRLF: its exact bytes and original SHA-256
`9934f166348cf8425e8ba13ce827dfd720aaeff9779fd99eae449271fbe7dc97`
are preserved in `inputs-measured.json.xz`. The review/replay `inputs.json` uses
LF; replay proves that both decode to identical data. The source patch binds LF
code and this canonical input, while each raw receipt identifies its actual input
bytes. The running remote input is immutable.

`oracle.py` uses PySCF 2.14.0, a fresh default MINAO initial guess, RHF energy/
gradient tolerances 1e-13/1e-10 and 200 cycles, then DF-CCSD energy/amplitude
tolerances 1e-12/1e-10 and 150 cycles. All electrons are correlated (`frozen=0`).
Each basis's exact PySCF records are hashed before computation and compared to
the native basis provenance; these raw records accompany the reference output.
PySCF is used only as an independent oracle and does not enter a production path.

## Results and retained records

The constituent and composite energy gate is **5e-8 Eh**, without relaxation for
Fast. Each policy has five total-energy constituent comparisons and one composite
comparison. `comparison.json` retains every value and absolute error; missing
references remain `INCOMPLETE`, never inferred from Standard.

| Molecule | QM7 index | Reference seconds | Maximum error, either policy (Eh) | Status |
| --- | ---: | ---: | ---: | --- |
| C1H4 | 0 | 5.368 | 5.200e-11 | PASS |
| C3H8 | 8 | 102.720 | 6.499e-10 | PASS |
| C4H10 | 48 | 313.568 | 7.981e-10 | PASS |
| C5H12 | 183 | unavailable | unavailable | INCOMPLETE |
| C6H14 | 293 | unavailable | unavailable | INCOMPLETE |
| C7H16 | 5604 | unavailable | unavailable | INCOMPLETE |

Reference runtimes are reported separately and are not speedup denominators.
There are no new native timing or performance-ranking measurements.

Each `receipts/C*.json.xz` contains the original reference JSON text, verbatim
SCF/CC logs, outcome and stdout, preserving their raw hashes. Checkpoints remain
in the persistent remote task directory. The first C1 attempt failed in a
diagnostic call after SCF because `get_grad` requires explicit orbital arguments;
its source, receipt, log and exit 1 remain under `failures/` and
`oracle-attempt1.py.xz`. The corrected attempt restarted from a fresh MINAO guess.

## Frozen native identity

Comparisons read full raw Standard/Fast receipts from
[`cbs-fast-memory-20261021/summary.json.xz`](../cbs-fast-memory-20261021/summary.json.xz),
pinned at repository commit `4a783a27e104e9392a7227b7efca79abcfba60ae`.
The original publication manifest and complete campaign verifier pass on
committed LF bytes. Windows work-copy CRLF bytes and default GBK decoding caused
initial replay failures; neither was accepted as a scientific failure or repaired
by editing historical evidence.

The measured native source is base `976f756b391bc65edb79f76df1a8c3fe807518a7`
plus patch SHA-256
`5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`;
library SHA-256
`65a401ed3db8fb303a5146f134cdb227af78928affe27e1f7571c09f270112de`.
`replay.py` reconstructs and checks all files in that recorded source manifest,
validates every native constituent's model and coefficient, and checks oracle
energy reconstruction. This qualifies those historical records only. It supplies
no new current-master installed-build, CUDA-device, force or performance evidence.

## CPU resources and reproduction

Actual compute uses own qz CPU Notebook `i2679-cbs-prep-1023`, created
2026-10-23 01:10:25 Asia/Shanghai, selected container suffix `0020ef93afaa`, node
`cpu-nat-576`, project 原子级化学反应基座模型2.0 / CPU资源空间 / CPU资源-2.
The allocation is 4 CPU / 16 GiB, cgroup `cpu.max=400000 100000` and
`memory.max=17179869184`; 128 visible affinity CPUs do not mean 128 allocated CPUs.
Numeric pools use four threads and PySCF's memory limit is 8000 MB.
The requested priority 10 was capped to actual HIGH/6.

Image: `generativeqc-science-py311-cuda129:20261015`. The task's fresh
`env/bin/python` explicitly reads the immutable image venv's site-packages after
`source /opt/generativeqc-reuse/activate.sh`; no system package was modified.
`environment.json` records actual package versions, cgroup, Python executable and
28 numerical shared-library/executable hashes. No compiler/JIT build occurred.

The persistent task root is
`/inspire/ssd/project/chemicalreaction/czxs25220150/i2679-cbs-energy-20261023-a1`.
`run-oracle.sh` uses per-molecule leases and temporary directories, a six-hour
timeout, retained outcomes, and independent output names. No fake Slurm identity
is set. The separate 55 CPU / 300 GiB Notebook `i2679-cbs-energy-1023` remains
queued. A single prep creation POST timed out and the platform exposed two
same-name prep objects; the other object received no calculation. No instance was
stopped/deleted and all remain subject to platform lifecycle/reclaim rules.

For a new independent run, stage the pinned full dataset, `inputs.json`,
`oracle.py`, `environment.py` and shell drivers under a fresh absolute task root,
create its isolated environment, verify the image's PySCF version, and run:

```bash
bash run-oracle.sh "$ROOT" 1 4 8000 "$ROOT/attempt1"
# Verify C1 before expanding the cohort.
bash run-suite.sh "$ROOT" 4 8000 "$ROOT/attempt1"
```

Replay the retained evidence from the repository without PySCF or CUDA:

```bash
python benchmarks/results/cbs-independent-energy-20261023/replay.py \
  --receipts benchmarks/results/cbs-independent-energy-20261023/receipts \
  --output comparison-replay.json
```

Use `--require-pass` only to require all six original energy gates to pass.
The publication decision remains inconclusive while references are incomplete.
