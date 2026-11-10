# Original KS scientific gates (2026-11-10)

All original RTX 5090 numerical tasks pass; no grid, scientific tolerance, iteration cap or physical Hamiltonian is substituted. Diagnostic fast-compile runs do not qualify production, performance, actual Graph launches or unmeasured GPUs.

Same-device parent/default water96 reproduces 45/288 force failures at 4.794283938736044e-7. Parent/tight-only falls to 1.4027001782324078e-11 on the identical original grid. Integrated repair default/tight pass at 3.398525905140559e-10 / 1.4007461857090675e-11 against 1e-7. This demonstrates stopping sensitivity in the identified current base, not historical #2910 causality.

All 8 original UKS controls pass (40 states / 20 analytic-force states), canonical max 9.267731582163208e-9 < 1e-8. All 4 original range-only PBE DF-primary/Exact Direct-LR RKS/UKS Cartesian/spherical controls pass (16 E/F/physical states), force max 6.686293893228869e-12 < 3e-5.

Source d3f8a62df6aa371ad75449406053d7ab3333c97a; library 41efe7a154cfd8e428edbc9ff4c25c9a99f42c380839866d893dbc9002c8c34e. Exact native input, Slurm UUIDs, controls, quantitative gates and provider boundaries are in evidence.json.xz; paired diagnostics are in samples.json.xz.

## Full lossless receipt recovery

Complete vectors and diagnostics remain in Git commit `de57dc9c3118d47bc3eb2692a7b51320fad311db`. Recover with:

```bash
git show de57dc9c3118d47bc3eb2692a7b51320fad311db:benchmarks/results/ks-original-complete-20261110/samples.json.xz > /tmp/ks-complete-samples.json.xz
python -c "import json,lzma; print(json.load(lzma.open('/tmp/ks-complete-samples.json.xz', 'rt')).keys())"
```

The selected record authenticates these exact bytes. No lossy rounding, status rewriting, external archive, release upload or evidence-budget increase is used. Raw histories remain ignored and authenticated by receipt/history hashes. The previous inconclusive publication and exploratory missing-SR/LR-sum failure remain recoverable with their own identities. See the implemented Agent Notes for the controlled diagnosis and rejected alternatives.
