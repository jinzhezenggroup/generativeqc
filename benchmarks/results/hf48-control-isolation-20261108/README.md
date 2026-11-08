# HF48 one-public-control diagnosis (not performance/default qualification)

The unchanged independent E`1e-8` / maximum-force`1e-7` diagnostic fails for
untouched-default HF48 on both A100 and RTX5090. Changing **only screening**
from`1e-12` to`1e-14` does not fix it. Changing only the public energy or
density request does fix it at the original screening. This isolates a
convergence-family boundary for this geometry, not an integral-screening floor.

| Card | Public request | Target iterations | Maximum force error / Eh Bohr^-1 |
|---|---|---:|---:|
| A100 | default |10|1.818375e-7|
| A100 | energy`1e-12` only |14|1.651491e-9|
| A100 | density`1e-10` only |21|4.234084e-11|
| A100 | screening`1e-14` only |10|1.818431e-7|
| RTX5090 | default |10|1.818376e-7|
| RTX5090 | energy`1e-12` only |14|1.651497e-9|
| RTX5090 | density`1e-10` only |21|4.236466e-11|
| RTX5090 | screening`1e-14` only |10|1.818431e-7|

Each maximum covers one complete cold endpoint and three own-density frozen
warm calls. Each public condition has a fresh process/owner; native imports
of PySCF/GPU4PySCF are blocked. Full force vectors, controls, target and
initial-guess work, scheduler/device receipts, logs and exact driver text are
retained losslessly in`observations.json.xz`.

**Internal coupling:** either tighter public energy or density request also
switches the qualified physical-stationarity ceiling from`1e-5` to
`min(1e-8,density_tolerance)`. These runs do **not** isolate public energy versus
that internal ceiling, or prove that either control alone is the sole cause.
No reference is computed in a native production process. Independent CPU
full-Fock reference16815 is reused unchanged; its actual orbital-gradient norm
is`4.44794e-11`, screening`1e-15`, and energy tolerance`1e-11`.

## Identity and interpretation

Measurements reuse the authenticated historical image at commit`62d9f2da7`
(master`74d06855c`), source`33db42ef97a4ce677b614a0900e18ba395c5837b7021c47df7f1e4ca1664d817`,
ELF`60f40b1565c6147fec5cce0bcc77f16338a41eadcc436d8c8f675d08d1871c17`.
They are not a newly built qualification of later master integration or PR head.
A100 job229107 and Slurm16820 each finish successfully; CPU229149 only
transports the existing A100 archive. A failed direct-SSH retrieval and a
receiver-readiness timeout do not cause a replacement GPU job.

Endpoint timings are retained for diagnosis, not ranked performance evidence:
this is one fresh process per distinct control, not a balanced same-control
cohort. It cannot establish a default speedup or the per-paired-warm3% guard.
The old default failures remain failures. README's looser observable contract
is separate; neither contract changes. Do not silently make all defaults
tighter, relabel strict requests as default passes, or blame screening.

Replay and host regression checks:

```bash
python benchmarks/results/hf48-control-isolation-20261108/replay.py
python -O benchmarks/results/hf48-control-isolation-20261108/replay.py
PYTHONPATH=python:. pytest -q tests/python/test_hf48_control_isolation_evidence.py
```
