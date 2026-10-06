# Residual Direct-force resources: #1892 P0-B

This publication records diagnostic and experimental evidence against measured
base `de18539de8a17fbae61b590f7f28e739c9a5be19`, **not a production promotion**.
Only the CPU resource collector and its tests enter the production checkout.
`cta-experiment.patch` reconstructs the private, default-off experiment; it is
not applied to the shipping source. #1892 and P0-B remain open.

## Review without a GPU

From the repository root:

```bash
PYTHONPATH=python:. python benchmarks/results/direct-force-resources-1892-20261005/verify.py
```

The verifier authenticates every publication member, restores exact original
JSON bytes in temporary storage, recomputes all 96 native energy/force gates
against the independent cold/moved oracle, and regenerates every complete
endpoint median and actual iteration/build comparison. Gzip changes storage,
not numerical precision, ordering, original reference hashes or tolerances.
Missing HF Fock-build telemetry remains null, not inferred as one.

`summary.json` contains job 6112's clean paired HF/PBE0 48/96-atom matrix, with
cold, five warm, moved and five moved-warm calls per arm. `samples/` retains
all original native and reference values, protocols, toolchain/device probes,
native KS histories where available and exact loaded library identities.
`profile-summary.json` separately records intrusive single-warm Nsight Systems
job 6111 and actual launch geometry. Raw profiler databases, logs and scheduler
receipts remain in ignored `.artifacts/p0b/` storage; their hashes are retained.

## Results and decision

| Clean endpoint | Control warm (s) | 128-thread warm (s) | Change | Control moved-warm (s) | 128-thread moved-warm (s) | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF 48 | 0.751831379 | 0.751084708 | -0.10% | 0.750311397 | 0.751064248 | +0.10% |
| HF 96 | 2.976780333 | 2.969434790 | -0.25% | 2.976284504 | 2.977499500 | +0.04% |
| PBE0 48 | 11.569291767 | 11.040451683 | -4.57% | 11.567456890 | 11.038006328 | -4.58% |
| PBE0 96 | 39.966333989 | 38.445335492 | -3.81% | 39.916961476 | 38.396261070 | -3.81% |

All warm/moved-warm rows have one actual SCF iteration; PBE0 also exports one
Fock build. Maximum energy/force errors over all 96 native endpoints are
`1.0504663805477321e-10` Hartree and `2.95813651352006e-11` Hartree/Bohr
against `1e-8`/`1e-7` gates. HF is compatibility evidence, not evidence that
this generic-force consumer became faster. Keep every cold/moved branch:
HF-96 cold executes 25/26 iterations; PBE0-48 moved executes 15/12 builds;
PBE0-96 cold/moved execute 28/25 and 17/12 builds. Do not normalize these
timings or attribute solver-trajectory differences to CTA scheduling.

The earlier four-arm 48-atom triage, job 6095, retains the disabled same-binary
control and the 64-thread alternative in `triage-summary.json`. The disabled
arm matches the sealed control closely; 64 threads has no material warm win.
Job 6091 passes through-f for `0`, `64`, `128`, `invalid` and all six actual
memcheck/initcheck/synccheck runs. `qualification-summary.json` binds the logs;
`ownership.json` records the separate 6150-case host admission/drain model.
These are measured qualification receipts, not synthetic GPU pass records.

The independent job 6111 trace confirms one generic restricted full-range force
launch with grid X 1360: sealed control and disabled candidate use block X 256,
while the enabled candidate uses 128. GPU intervals are respectively
`2.252254201`, `2.252667086` and `1.700066311` seconds. Both real NVTX ranges
are retained. This attribution supports the launch mechanism, **not** a clean
timing replacement, an achieved-occupancy claim or a spilling/local-traffic
measurement. No NCU hardware counters were collected.

**Decision: inconclusive for production promotion.** Retain 128 threads as a
promising isolated candidate. NCU counters, qualified P0-A composition,
complete promotion provenance/compilation/allocator gates and an interleaved
promotion assay remain outstanding. Neither this PR nor permission to proceed
authorizes sudo, a driver profiling-policy change or closing #1892.

## Source identity and reconstruction

- Control library SHA-256: `daf5d5b0f3bdca15bdf0e56243f0163455767333c2fa8ff6b9285f4a561ce2af`.
- CTA candidate library SHA-256: `cca2d24221b6c040a32907b70cb77590526a6dd891da1ba14352e9b8b4472098`.
- `candidate-source-identity.json` binds the measured base, whole tracked-source
  inventory hash and sole changed file, `src/scf/cuda/direct_bounded_fallback.cu`.
- Apply `cta-experiment.patch` only to a separate checkout of that exact base.
  Use release sm_120 CMake builds with verified ccache CXX/CUDA launchers and
  fast compile disabled. Set `CCACHE_BASEDIR` to the checkout root; do not clear
  the cache, use sloppiness or alter source identities to manufacture hits.
- The default-off switch is `GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS`;
  only `64` and `128` change the existing full-range force boundary. Invalid,
  zero and unset settings preserve the control launch. The fixed queue and
  indexed pages remain 256 entries and 16 x 64 candidates. Admission packets
  and warp drain must follow the actual CTA size or work is skipped. This is
  not an isolated geometry-only experiment and does not change launch bounds,
  scientific ERI/WeightedIntegralIR ownership, generated dispatch or SR/LR/value
  boundaries.

Historical drivers retain their actual node1 paths and prerequisites rather
than pretending to be portable installers. `run-matched-matrix.sh` and
`run-profile48.sh` show the scientific commands, reversed size-dependent arm
order, independent references and finite Slurm steps. `run-endpoint-arm.py`
substitutes only the archived HF harness's two Git provenance queries, not a
scientific owner. `verify-matched-matrix.py` is the unchanged measured verifier.
`verify-profile48.py` rechecks raw trace hashes/ranges/launches when the ignored
original job directory is available.

Any new real-GPU reproduction, sanitizer or profile must retain Slurm's device
visibility and use a finite step such as:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:55:00 bash run-matched-matrix.sh endpoints
```

The archived runners additionally require a finite parent allocation and the
original sealed build/qualification assets; adjust installation paths, not the
scientific protocol. n1 was chosen from the allowed hosts; there is no claim of
multi-host qualification. No new device experiment is run during PR packaging.

## Static-resource reporting boundaries

`control-resources.json` and `cta-resources.json` retain final-linked CUOBJDump
reservations. `shared-control.json` and `shared-warp-candidate.json` bind job
6083's attribute-only driver probe. For this sm_120 driver the 1024-byte reserved
block allocation accounts for control linked SHARED 4108 versus application
shared 3084 (and rejected warp candidate 4192 versus 3168). The exact extracted
ELF symbols expose `.nv.reservedSmem.cap = 0x400`. Do not universally subtract
1024 across architectures/tools. Linked STACK 90680 and LOCAL 0 remain distinct;
zero LOCAL/CUPTI local columns do not prove absence of executed local traffic.

`rejected-warp-summary.json` and `rejected-warp-experiment.patch` preserve the
earlier coupled warp-page/operator-specialization negative result, with
`rejected-warp-profile-summary.json` providing separate intrusive attribution.
48/96-atom PBE0 warm endpoints regress by 10.31%/2.36%; the 96-atom generic
force interval regresses by 14.48%. Do not transplant that rejected code into
production or mistake that workload for a post-qualified-P0-A residual.
