# Guarded DF force response defaults

Full evidence resides in this PR's Git history, identified by `snapshot.json`.
Master had 3060 bytes free under the unchanged 64 MiB cap; existing evidence
is untouched.

Measured source: frozen `2a6de26eea1f83e9fd1e8df01c189fea44058b6b` plus the
retained-shell/default overlay, not later rebased master. FP64 ethane230;
single-pass shell derivatives and both Lambda matrix audits stay matched.

- Job 1492, n5 RTX 5090, five samples per side: default fusion/scratch changes
  complete E+F 227.335725 -> 219.841134 s (-3.2967%); peak VRAM -824 MiB.
  Lambda time is unchanged; scratch sharing is a memory optimization.
- Separate job 7829, n1 RTX 5090, Q32/Q48 ABBA: Lambda 79.083091 -> 76.944110 s,
  complete E+F 220.128467 -> 217.917966 s (-1.0042%), peak VRAM +3906 MiB.
  Q32 remains default. Do not pool these GPU cohorts or add their reductions.
- Job 7826: 779 tests pass, including host planning and real-device scientific
  cases; 17 memcheck cases report zero errors/leaks. Large independent FD covers
  two components/two steps; all water coordinates have independent FD gates.

## Recover and verify

Fetch this PR's full head history if using a shallow clone, then restore offline:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/df-response-defaults-20261012/snapshot.json \
  --output .artifacts/df-response-defaults-restored
```

Recovery verifies all original SHA256s/sizes: raw numerical outputs, VRAM
samples, independent inputs, Slurm scripts, source patch, validation envelopes,
and excluded/failed attempts.
