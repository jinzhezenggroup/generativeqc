# Fock provider and device-graph synchronization diagnosis

The final change corrects the generated-exchange provider assertion: automatic
MD-J may change the overall schedule label while generated K remains selected.
The test checks the value-only dispatch contract and retains the independent
ERI/J/K, spin, representation and fallback comparisons. No production Fock
barrier change or performance gain is asserted.

The corrected full provider test passes ordinarily and under synccheck (zero
errors); the public water3 Direct-HF endpoint also completes.
`provider-validation.json.xz` binds those outputs to the source, binary, commands
and Slurm allocations. This is integrated-source qualification, not a separate
standalone-master GPU build. Applicable pre-commit checks pass.

## Standalone synchronization controls

`graph_synccheck_probe.cu` links only CUDA. A producer stores 7; a consumer
unconditionally synchronizes all threads and writes 2,816 checked outputs.
Run every GPU command through a finite Slurm allocation and preserve its device
visibility. Compile with the verified compiler cache, for example:

```bash
sccache --version
sccache nvcc -std=c++20 -O3 -lineinfo -arch=sm_120 graph_synccheck_probe.cu -o probe
compute-sanitizer --tool synccheck --error-exitcode 99 ./probe 0
```

| Mode | Execution | Plain output verification | Synccheck 2025.2.1.0 | Synccheck 2025.4.1.0 |
| --- | --- | --- | --- | --- |
| 0 | Ordinary launches | pass | pass | pass |
| 1 | Ordinary host Graph | pass | pass | pass |
| 2 | Same Graph with device-launch flag | pass | exit 99 | exit 99 |
| 3 | Intentionally divergent barrier, ordinary Graph | diagnostic only | exit 99 | exit 99 |

The device-launch-enabled Graph also passes memcheck and racecheck. Hardware is
RTX 5090, driver 580.95.05, compiled with CUDA 12.9.1 for sm120. The mode-2
Graph does not perform a device-side launch. Full endpoint synccheck remains
instrumented-unqualified on this stack; the standalone failure is evidence of
a tool/Graph interaction, not a passing production synchronization result.

`graph-diagnosis.json.xz` retains the controls, source digest, commands,
allocations, tool versions, failed candidate cohorts and isolated actual-library
ddpp replay. Its minimal v2 negative control keeps all lanes alive after the
conditional barrier; the earlier control permitted nonparticipants to exit and
therefore did not demonstrate invalid synchronization.

## Retained unsuccessful candidates

`diagnosis.json.xz` preserves the initial source `121b14ab59aa5ed94c2311de634e11e745e8923a`
and the partial warp candidate. `iterations.json.xz` preserves the unselected
Rys-emitter attempt. The generic candidate `f8edc52a6d7fe331be55eb7b35e106f9ea72f5fb`
still failed complete water3 synccheck; adding warp joins did not repair it.
All these production edits were withdrawn. The restored integrated source is
`92674610bac506d76b762bc30594e72aae0a1baa`.

Read lossless evidence without GPU work:

```python
import json
import lzma
from pathlib import Path

record = json.loads(lzma.decompress(Path("graph-diagnosis.json.xz").read_bytes()))
print(record["logs"]["graph-synccheck-minimal-v2/old-2.log"])
```

The [decision note](../../../.agents/notes/rejected/2026-10-24-packed-fock-warp-synchronization.md)
records why a moved sanitizer failure was not accepted as a production fix.
