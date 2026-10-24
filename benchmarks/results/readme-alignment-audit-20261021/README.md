# Default-alignment audit: completed large-case checks

The compressed bundle retains original WB97M-V Direct96 Standard/Fast/reference
receipts, independent XC5/6 and VV10 4/5 references, the matching CPU spot, and
receipt-bound qualification. Slurm 11405 completed the previously missing
96-atom reference. Every retained sample passes the unchanged common energy/force
gates; reference stability and CPU/GPU agreement pass quarter gates.

At baseline source `02337e0cf`, Standard warm median is 45.438 s versus default
GPU4PySCF 120.961 s (2.662x). Native still uses the historical VV10 density
cutoff 1e-8 and the reference uses 1e-10. This is qualified baseline evidence,
not a replacement measurement of the later density-domain or Fast fixes.

The original r2SCAN-3c DF96 receipts, Slurm logs and exit codes are retained
verbatim. Both jobs reached four hours: Standard completed cold (6546.999 s)
and two warm calls (3686.229/3686.284 s); Fast completed no call. Their original
`running` receipt status is preserved together with the terminating scheduler
evidence. Neither is a completed cold-plus-three-warm population.

The full frozen campaign accounts for 192 populations: 188 complete, two
WB97M-V DF96 resource failures and these two timeouts. Updated qualification
is 170 passed, 18 failed (HF DF and M06-2X water-3), four incomplete/unverified.
These counts describe the baseline campaign, not all-method superiority.
See the [settings audit](../../../.agents/notes/implemented/performance/2026-10-20-readme-default-alignment-and-ranking-reversal.md).

Recheck the retained WB numerical population with:

```bash
PYTHONPATH=python python - <<'PYTHON'
import json, lzma
from pathlib import Path
from benchmarks.readme_defaults_qualification import qualify
p = Path("benchmarks/results/readme-alignment-audit-20261021/receipts.json.xz")
b = json.loads(lzma.decompress(p.read_bytes()))
r = b["numerical_receipts"]
measurements = [v for k, v in r.items() if "-direct-" in k]
references = [v for k, v in r.items() if "-direct-" not in k]
result = qualify(measurements, references)
assert result == b["qualification"]
assert all(row["status"] == "qualified" for row in result)
print("All retained WB97M-V Direct96 populations qualified")
PYTHON
```
