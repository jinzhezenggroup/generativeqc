# Ordered Cooperative Becke Normalization

This publication retains complete PBE0/def2-SVP RKS E+F pairs on n2's RTX PRO
6000 Blackwell, under finite `main/gpu:pro6000:1` Slurm allocations and ccache.

| atoms | replay | serial median (s) | cooperative median (s) | improvement |
| ---: | --- | ---: | ---: | ---: |
| 96 | warm | 23.572726 | 22.878456 | 2.945% |
| 96 | moved-warm | 23.582680 | 22.907833 | 2.862% |
| 48 | warm | 8.663366 | 8.518470 | 1.673% |
| 48 | moved-warm | 8.663816 | 8.530103 | 1.543% |

Five interleaved A/B pairs support each row. Both 96-atom rows pass the shared
greater-than-2%/relative-MAD gate. The 48-atom point estimates are below that
benefit threshold; they establish no material regression, not a significant win.
Every timed call performs one actual SCF iteration/Fock build. Native density
updates are publicly frozen; whole gradient owners are replaced/primed outside
timing. Raw files retain every vector, solver history, prime, setup and counter.
No opaque density-byte hash or bypassed SCF is claimed.

Independent GPU4PySCF 1.8.1 references use exactly the same explicit grid and
basis. Maximum errors across setup, primes and timed calls are `1.033e-10`
hartree and `3.125e-11` hartree/bohr, within unchanged `1e-8`/`1e-7` gates.
The 96-atom force still visits 10,758,389,760 pairs in each primal/reverse phase
and 226,492,416 normalization atom entries. This is scheduling, not work removal.

## Provenance And Scope

- Matched timing: job 2500, base `83109befe5b97a5542b30e169da74c81d04719e6`
  plus `source.patch`, source identity `b0e32abf342aa53877f2a350801a1dfdc1bc74d341df0cebe82a738d74429d2f`.
- Integration qualification: job 2505, base `80296618e2003427b220331e1c0d4dad6752391e`
  plus `integration-source.patch`; actual hashes and receipts are in
  `qualification.json`. The normalization compiler/helper/native files are
  byte-identical across cohorts. No performance claim on the integration base.
- Final gates: 49 normalization/interface/fallback tests, four clean sanitizers
  on 39 CUDA cases each, 144 synthetic native-owner cases and five CPU lifetime
  cases. Small three-atom RKS/UKS/finite-difference tests exercise the retained
  fallback; physical admitted UKS/96 remains unmeasured.
- Cold/moved setup calls are not matched A/B evidence. No 5090 measurement,
  intrusive phase-profile result, whole-core compile cost or concurrent endpoint
  device-peak measurement is claimed. Numerical acceptance does not fabricate
  the stricter generic memory/compilation performance-promotion envelope.

## Offline Verification And Reproduction

```bash
PYTHONPATH=python:. python -m pytest -q tests/python/test_becke_normalize_evidence.py
```

The test uses the shared publication reader, authenticates selected files and
recomputes independent whole-vector gates, medians, semantic work and actual
solver iteration counts. Changed vectors/counters cannot reuse a passing summary.

For exact timing-source restoration, retain this directory outside the source
checkout, check out the measured base, apply `source.patch`, and verify
`source-files.sha256`. Set `PYTHON`, `CUDA_ROOT` and optionally `HOST_CXX`, then
run `bash <retained-directory>/reproduce.sh` on n2. It uses ccache and finite
Slurm allocation without changing scheduler visibility. `PROFILE=1` optionally
adds a separate intrusive campaign; it is not part of the retained timing proof.
Full build/test/profiler logs remain in ignored artifacts. No Release or external
archive is needed for reproduction.
