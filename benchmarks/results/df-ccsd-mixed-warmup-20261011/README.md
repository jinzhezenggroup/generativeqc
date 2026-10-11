# DF-CCSD mixed warmup and exact Lambda scratch reuse

## Scope and source identity

Five alternating AB/BA pairs measure complete, cold energy **and force** calls
for ethane / aug-cc-pVTZ / aug-cc-pVTZ-RI (230 AO, 488 RI, 9 occupied and 221
virtual orbitals). RHF and orbital/nuclear response use the exact conventional
Hamiltonian; only correlation is density-fitted. Every call starts in a fresh
process without supplied electronic state. Compilation is excluded, and the
compiled executable is reused. The (T) mixed-precision option is disabled.

The comparator is frozen master `98263e06ca13aa88cee504f1f105f5cf3ef6d92a`,
not whatever master contains when this evidence is read. The measured candidate
is that commit plus `measured-source.patch.gz`; it was dirty at measurement.
The publishing commit also adds evidence, updates its note link and registers
the opt-in policy; none changes measured executable sources. `provenance.json` binds the exact
patch, input, binaries, controls, hardware and retained records by SHA-256.
No later publishing commit or rebuilt binary is claimed to have been measured.

## Results and limitations

| Median | Frozen master | Candidate |
| --- | ---: | ---: |
| Complete native E+F, seconds | 305.987797367 | 293.743218727 |
| Full process wall, seconds | 306.40 | 294.14 |
| CCSD including strict replay, seconds | 63.829336933 | 51.812468808 |
| Lambda, seconds | 78.945867989 | 79.036452954 |
| Sampled device occupancy peak, MiB | 31,529 | 27,547 |
| Lambda owned device bytes | 15,542,559,936 | 10,324,999,872 |

Complete native E+F time falls 4.0017%, CCSD time falls 18.8266%, and sampled
device occupancy falls 12.6296%. Occupancy uses 200-ms NVML samples of total
device usage: **it is not an exact allocator peak**. Exact sequential Lambda
scratch reuse avoids a second 5,217,560,008-byte reservation; Lambda arithmetic
does not speed up. These scoped results do not promote the opt-in precision
policy to a default or establish performance for other sizes/devices.

Maximum paired energy error is `6.963318810448982e-13` hartree and maximum
force-component error is `1.759259404821023e-11` hartree/bohr. Original independent
FP64 CCSD replay residuals remain at most `1e-10`; Lambda, Z and stationarity
gates remain below `1e-8`. `independent-accuracy.json` retains H2/water/LiH
same-Hamiltonian oracle energies and two-step directional finite differences.

CCSD evaluations change 38 → 36 (20 FP32 warmup, 16 strict FP64); strict Q4
occupied-pair admissions all succeed. FP64 GEMMs change 54,862 → 48,520, with
23,180 FP32 GEMMs and 183,089,660 converted elements. Contraction terms increase
26,799,066,988,432 → 34,721,071,141,244; logical packing traffic increases
3,270,860,758,368 → 3,453,086,946,496 bytes. This is a measured precision/schedule
tradeoff, **not** reduced total work or traffic. Lambda retains Q32, 21 iterations,
22 actions and its independent matrix audit. No discarded JK attempt occurs.

The first compact variant is rejected: complete E+F regresses
303.359 → 353.916 seconds even though occupancy falls. Its earlier switch,
strict Q2 and unavailable pair fold increase CCSD evaluations to 48. The
[decision note](../../../.agents/notes/implemented/performance/2026-10-11-df-ccsd-mixed-warmup-shared-lambda-scratch.md)
explains the corrections; `rejected-v1.json.gz` retains the negative samples.

## Retained records and reproduction

- `summary.json`: final medians and numerical gates, without duplicated samples.
- `samples.json.gz`: original complete analysis, including all ten unrounded
  endpoint records, full process wall measurements and occupancy peaks.
- `occupancy-samples.json.gz`: all original timestamps and memory-used values
  for every call. Timestamps use a first timestamp and adjacent integer
  millisecond deltas, verified by exact round-trip reconstruction. Unused
  clock/power telemetry is omitted; original CSV hashes remain in provenance.
  No occupancy sample is removed or rounded.
- `measured-source.patch.gz`: dirty-source reconstruction against frozen master.
- `measure.sh` and `analyze.py.gz`: exact historical measurement and gate scripts.
  The runner contains machine-local dependency paths, not a portable installer.
- `independent-accuracy.json` and `provenance.json`: independent numerical
  records, qualification summaries and authentic source/binary/storage identities.

The shared [ethane input](../df-lambda-gemm-20261004/ethane230.input) has SHA-256
`9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
Reconstruct the candidate by applying the decompressed patch to the frozen base.
Build both libraries and benchmark executables with the same verified ccache,
CUDA 12.9.86, GCC 11.4.0, Release and sm120 settings. Arrange the historical
runner's `master/` and `candidate/build-cuda/` paths and the shared input before
running five pairs, keeping Slurm's assigned device visibility:

```bash
srun --partition=main --nodelist=n1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=01:00:00 bash measure.sh /path/to/run-root 5
gzip -dc analyze.py.gz > /path/to/run-root/analyze.py
python /path/to/run-root/analyze.py /path/to/run-root --minimum-pairs 5
```

The qualification suite reports 51 passed, nine expected CPU-parameter skips
and three unrelated (T)-mixed tests deselected, not 51 distinct GPU tests.
Separate water complete-E+F memcheck reports zero errors and zero leaked bytes.
CPU qualification reports 62 passed / 76 expected opt-in skips; additional
precision/lifetime contracts report 55 passed. Full routine logs, build trees,
libraries, scheduler debris and rejected source archives remain local in the
sibling `evidence-ccsdt-mixed-20261011/` audit directory; they are not Git payload.
