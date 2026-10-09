# Explicit implicit-Hcore / class-major-force PBE0 endpoints

These observations qualify the **composed explicit selection**, not Hcore
alone, cold starts, a global default change or an isolated kernel speedup.
The measured tree is `88cfa701753e91aa3abbe17877fb239ee3fd9ab4` plus
`measured-source.patch.gz`. Slurm 6921 runs on n1's RTX 5090, assigned device
visibility `0`, with a finite 35-minute limit. The common native library is
`1e2059954b4b003e717a0e30c68c55891ab0d7b26f0b7dd543539801cb4cc7a5`.

## Complete endpoint timing

PBE0/RKS, spherical def2-SVP water clusters, the pinned rational-Legendre /
Legendre-trapezoid 48 x 16 x 32 grid and full-density reference Fock rebuilds
are unchanged. Timings enclose a synchronized prepared E+F call including
returned host forces. Both arms share one native binary/CUDA context but have
separately prepared force owners: force selection is captured at construction.
Each owner's native density/coordinate bytes are frozen and verified before
and after both geometries. **Cross-owner seeds differ**; this is not an
identical-byte same-owner comparison.

Baseline selects `GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_MAPPING=shell_warp` and
`GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES=0`; candidate selects
`nucleus_cooperative` and `1`. Materialized derivatives remain enabled in both.
The angular qualification route and intrusive Becke profiling are disabled.
One clean population retains five ABBA-interleaved samples per side/phase,
every prime, every setup and both owner constructions. There are 46 complete
calls per size: 20 samples, 20 primes, four setups and two constructions.

| Atoms / phase | Baseline median s | Candidate median s | Gain | Robust floor |
| --- | ---: | ---: | ---: | ---: |
| 48 / warm | 5.582238134 | 5.333044238 | 4.46405% | 3.94885% |
| 48 / moved-warm | 5.524045762 | 5.321095161 | 3.67395% | 2% |
| 96 / warm | 17.062031366 | 16.045000125 | 5.96079% | 2% |
| 96 / moved-warm | 17.046683248 | 16.009522241 | 6.08424% | 2% |

All four gains strictly exceed
`max(2%, 2 * (relativeMAD_baseline + relativeMAD_candidate))`.
No size, sample or failed cohort is discarded and no timing retry is used.
Cold/moved setups are retained for science/work validation, not advertised
as matched-work timing gains. The existing Hcore-only five-repeat matched-owner
population failed at 48 warm (1.98585%, below 2%); the earlier three-full-domain
force split inflated producer work 1.5x and was rejected. Their rationale and
negative results remain in the linked Agent Notes and ignored raw artifacts.

## Scientific, work and resource checks

Every complete call passes the independent retained GPU4PySCF 1.8.1 / PySCF
2.14.0 reference: `1e-8 Eh` energy and `1e-7 Eh/Bohr` force gates. Maximum
errors are `6.13909e-12 Eh / 2.46954e-11 Eh/Bohr` at 48 atoms and
`1.04592e-11 Eh / 2.95027e-11 Eh/Bohr` at 96. Every warm prime/sample
uses one Fock build/iteration, physical residual at most `1e-10`, and no warm
fallback. Recorded grid/AO semantic work, resource bounds and traffic agree
between arms. Raw records retain quantities rather than only pass flags.

The independent intrusive diagnostic is **not timing-promotion evidence**.
It retains 14 complete calls per size, per-class admission equality, launch
resources and the actual rectangular producer census. At 48 atoms the new
producer visits 136 dd x 1024 dp = 139,264 candidates in 4,352 pages; at 96,
528 dd x 4096 dp = 2,162,688 candidates in 67,584 pages. Total logical pages
rise only 1,033,088 to 1,037,440 and 4,476,288 to 4,543,872. Generic/dddd
keep their original geometry-live full domains. The class CTA uses all 256
coefficient lanes, 214 registers, 416 static and 42,984 dynamic shared bytes,
and 144 local bytes. No resident pair list or production topology readback
is added. Force-owner density bounds, exact screening/masks and fallback
admission remain unchanged.

Six independent libcint Cartesian/spherical RKS/UKS/disabled-K force cases
execute without skips on the frozen mathematical source. The actual current
native build has byte-identical generated pair-gradient/source-contraction
headers and executes the #2153 K/convergence integration: 14 original
energy-change passes plus the two precisely rechecked snapshot cases. Those
two originally failed for absent packaged PBE0 artifacts; the original failed
JUnit and the successful package/recheck are both retained. Targeted memcheck
and racecheck each execute one independent Separate/Combined/replay case;
there are zero errors/warnings. A qualification-only argument-preserving
observer proves three actual class launches per tool, and submits no CUDA
work itself. It is absent from clean timing.

`explicit-default.patch.gz` is a subsequent **untimed** bridge admission fix:
an unset mapping retains the old prepared-Hcore schedule instead of silently
promoting the unqualified Hcore-only path. The explicit measured selections,
all CUDA kernels and algebra are unchanged. Four compiled host admission
tests and four device RKS/UKS snapshots cover default and explicit selection.
The separately built followup library identity is in `provenance.json`; it
is not relabelled as the timed binary. Cooperative force remains opt-in.

Actual isolated compile-cost and peak-memory budget measurements were not
taken. Compiler-cache receipts, resource bounds and construction free-device
snapshots are retained, but are not substituted for those measurements.
`qualification.json` therefore marks the shared formal default-promotion
envelope **not run**. These data support only the declared composed endpoint.

`unsigned-portability.patch.gz` is a second **untimed** followup. It uses
unsigned comparisons for physical pair orientation instead of provider-dependent
signed-only `max/min` overloads; casting their narrowed results is not safe
above INT32_MAX. Actual-source compiled probes include the UINT32_MAX boundary.
The focused CI-repair set executes 512 host cases. Slurm 6930 separately builds
the repaired tree through the existing verified ccache and executes one
independent libcint Separate/Combined/replay case without skips; the retained
qualification-only observer proves three class-kernel launches. Its library is
`3a606d0ec5cd1267935327672c45a5bf7a15f1d10a8626e221502f529623c9d8`.
The first reconfiguration attempt (Slurm 6929) fails before compilation because
the copied CMake configuration retains its old checkout path; that failure is
retained, not relabelled as a numerical failure or successful first build.
All 31 original archives remain byte-identical. No timing population or sanitizer
matrix is repeated for this expression-only portability repair.

## Reproduction and retention

- `paired-{48,96}.json.gz`: all construction/setup/prime/sample outputs and work.
- `reference-{48,96}.json.gz`: unchanged independent references and full protocol.
- `summary.json`, `provenance.json`, `qualification.json`: exact medians and scope.
- `measured-source.patch.gz`, `explicit-default.patch.gz`: original source deltas.
- `unsigned-portability.patch.gz`: subsequent, separately qualified portability fix.
- `paired-force7.py.gz`, `paired-ao-endpoint.py.gz`: exact driver and owner adapter.
- `verify-{formal,seeds,diagnostic}.py.gz`: independent acceptance checks.
- `diagnostic-*.json.gz`: independent mechanism checks/class census and TSV data.
- `qualification-receipts-*.json.gz`: relevant compiler commands/cache receipts,
  every named test outcome and original JUnit hashes; not full logs/traces.
- `checksums.json`: compressed and uncompressed content identities.

Decompress the scripts, relocate their machine-local `ROOT`/toolchain paths
and recreate the source/driver layout named by `env-next.sh` and `formal.sh`.
Reconstruct the pinned measured parent and apply only `measured-source.patch`
for the measured tree. The basis remains at
`benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`. Configure the
retained Release/sm120 flags with a verified sccache/ccache launcher. Build
`generativeqc` and the generic/spd PBE0 RKS/UKS manifest targets before snapshot
qualification. Place the exact owner adapter in `evidence/bulk-point-v7/`,
the driver in `evidence/force7-class-master-v1/`, and references at their
`reference-{48,96}-retained-perf5.json` names. Apply the default-preservation
patch only in a separate followup tree; do not overwrite the measured binary.
For the repaired native source, apply the unsigned-portability patch after the
default-preservation patch. This three-delta reconstruction is checked against
the current CUDA/CMake source bytes; the timed source and library stay pinned
to their original identities.

All real GPU work must use a finite compatible allocation and preserve its
device visibility, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --nodelist=node1 --time=00:35:00 bash formal.sh
```

Full native binaries, intrusive diagnostic records, NPZ seeds, logs and JUnit remain ignored at
`.artifacts/force7-class-stage/` and
`n1:/data/jzzeng/qc-next-hotspot-20261009-125a4e33f/`. Recomputing the full
seed audit requires those retained NPZ files alongside the raw records; output
numerics and timing can also be audited directly from the compressed JSONs.
No Release, tag or external backup is created. Master #2167/#2173 only change
post-HF DF admission/CC replay, so no duplicate PBE0 matrix or timing campaign
is launched merely for those commits; measured source stays explicitly pinned.

To keep the existing 64 MiB aggregate limit, nine prior PBE0 numerical JSON
records are losslessly gzipped (449,503 bytes saved), with publication storage
paths/checksums and the two historical summary links updated. Decompression
is byte-identical to the parent Git objects; scientific values, acceptance
decisions, sample order and source identities do not change. Full new logs and
intrusive diagnostic streams remain ignored rather than hidden in archives.

See the [class-major decision](../../../.agents/notes/implemented/performance/2026-10-10-order-seven-class-major-force.md)
and [implicit-Hcore decision](../../../.agents/notes/implemented/performance/2026-10-10-ks-implicit-cooperative-pairs.md).
