# Packed Fock queue synchronization

The retained `diagnosis.json.xz` contains the complete initial failure logs,
commands through the recorded Slurm/source context, core/build identity,
public water3 receipt and the passing ordinary/memcheck controls.

On source `121b14ab59aa5ed94c2311de634e11e745e8923a`, complete HF synccheck
reports 32 errors in the generated ssss persistent Fock kernel. The public
water3 Direct-HF diagnostic reports 416 errors in that same kernel. Turning off
validation GEMM does not avoid the failure. These instrumented observations are
excluded from benchmark timings.

The candidate uses warp barriers for exactly 32 queue participants and retains
CTA barriers for wider queues. Eighteen focused host tests pass. Source
`c62e9d7dce87477738311f145f1838d4ad99d538` passes the original complete-HF
synccheck, including its GEMM-disabled control. The public water3 check now
reaches a wider `ddpp` queue and fails with 7,584 synchronization errors. Those
logs are retained under `candidate_device_validation`. This is a partial repair;
this publication does not assert GPU acceptance or a performance gain.

Read the lossless evidence without GPU work:

```python
import json
import lzma
from pathlib import Path

path = Path("benchmarks/results/packed-fock-warp-sync-20261024/diagnosis.json.xz")
record = json.loads(lzma.decompress(path.read_bytes()))
print(record["initial_logs"]["qualification-native/synccheck-incremental-hf.log"])
```
