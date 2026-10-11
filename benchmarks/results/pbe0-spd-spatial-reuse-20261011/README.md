# PBE0 primitive-local spatial Coulomb reuse

This source-matched stage qualifies a bounded automatic force route, not
universal KS performance or a cross-code wall-time comparison. The compiler
reuses repeated spatial Coulomb states with the existing scalar mathematics:
order-six primitive recurrence evaluations fall from 256 to 22–54, enclosed by
at most 64 doubles. Screening and gradient accumulation order remain unchanged.
No retained quartet inventory, extra global traversal or dense allocation is
introduced. Generic low, order-seven/eight and mixed-f fallback consumers remain.

Unset/empty `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS` now automatically requests
reuse for method-authenticated exact-direct, full-range PBE0/RKS, explicit FP64
and at least 384 public AOs. `0` is an opt-out; `1`/`spatial-reuse` is a guarded
explicit request. Preference is captured at preparation. Resident token,
primitive-pair, topology, prefix and derivative-capability admission remain
mandatory. Generic/HF/RSH consumers do not infer method identity from exchange
coefficients. See the
[implemented decision](../../../.agents/notes/implemented/performance/2026-10-11-pbe0-spatial-force-reuse.md)
and [current options](../../../docs/user/ks_options.md).

## Complete scoped endpoints

RTX 5090/sm_120, CUDA 12.9, release/full-precision native image, five interleaved
pairs per geometry. Each size uses one image/process/context with two independent
resident owners. Every timed endpoint includes synchronized energy **and analytic
forces returned to the host**. Seed density/coordinates are immutable within each
owner and geometry; seeds are not assumed bit-identical between owners.

| Atoms / AOs | Geometry | Baseline median (s) | Candidate median (s) | Reduction | Gate |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 / 384 | warm | 4.863446675 | 4.569519117 | 6.0436% | 2% |
| 48 / 384 | moved-warm | 4.806275673 | 4.598930910 | 4.3140% | 2% |
| 96 / 768 | warm | 14.104704417 | 13.352007795 | 5.3365% | 2% |
| 96 / 768 | moved-warm | 14.248137016 | 13.543867871 | 4.9429% | 3.1329% |

All four pass the unchanged descriptive gate
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`. Jobs 7339/7342
each complete 30 numerical calls and 20 timed samples, exit zero. Baseline is
explicit `0`; candidate selection is **genuinely absent**, not explicit `1`.
Both arms set `GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=1`, so #2220's gain
cannot be attributed to this optimization. All raw timings, including slow
observations, are retained; no pilot/failed/other-source cohort is pooled.

All 60 complete calls independently pass 1e-8 Eh / 1e-7 Eh/Bohr against the
retained GPU4PySCF/PySCF full-density energy and analytic-gradient reference
with CUDA-libxc 7.0.0, using
the same geometry, spherical def2-SVP H/O basis and Becke grid. Maximum E/F
errors are 6.366463e-12 / 2.474398e-11 at 48 atoms and
1.136868e-11 / 3.752010e-11 at 96 atoms. Cross-arm density drift is
6.978307e-13 / 1.840028e-12, below 2e-11. Timed calls retain one iteration,
one Fock build and available semantic-work parity. Capacity counters are not
executed screened J/K quartet counts. Construction/setup/priming calls are
numerically checked but are not substituted for the warm timing cohort.

These are warm and changed-geometry complete endpoints. No cold-SCF,
other-method, smaller-owner, larger-system or other-GPU timing is newly claimed.
The publication envelope accepts the **numerical** scope. Four passing scoped
timing assessments justify the bounded default; formal broad performance and
production envelopes stay `not-run`. Equation/IR/schedule identities or runtime
memory peaks are not fabricated to turn those envelopes green.

## Source and build provenance

Measured dirty source is master
`637466c624c4c4c4dd4eb7c511a2c88675744325` plus `production.patch.xz`:

| Artifact | SHA256 of decoded/original bytes |
| --- | --- |
| Base Git archive | `af2baa707ed021c7287e6942173e25e3141b5c7154e0fdcfdd115001ed7d6afd` |
| Production patch | `7cfa7717e6682b5ccc95fa8d494d4a0e052b4258f1b33ba37341477618765f44` |
| Original qualification patch | `d8a05faeb7a258cc9ad3a33478c7d8be14dcba3c458bc800bfefd5c697752cf4` |
| Native library | `47209e21c0fb92038475c0f7705f3540d0ea06bc77c0f3afab4201bfd3b26a49` |
| Stationary PBE0 AOT library | `969d92da3008b710c287c673d778194485443ba2517f5b0b334d99bdf89192b5` |

The measured product is not attributed to a future publication commit. The
original qualification patch includes the production patch and its original
test/doc overlay: apply it **instead of**, not on top of, the production patch.
`qualification-final.patch.xz` reconstructs the publication's final qualification
files from the same master. Its later combined-layout adapter/CMake/doc/inventory changes
do not change the frozen production files or rebuild the native library.
The exact combined adapter used in CPU build 7332 is retained in raw receipts.

Build 7321 verifies ccache 4.5.1, actual CXX/CUDA launcher invocations and
before/after statistics. No cache is cleared or disabled. Large build logs,
full source manifest, all compiler commands and linked resource listings remain
ignored with hashes/sizes in the raw envelope; selected actual launcher commands
and static resource summaries are retained. They do not stand in for device
execution. The generic native kernel section is byte-identical to master:
28,528 bytes, SHA256
`1983a5611f97e81bd970adc8859b4164d0ba846b8f5b34b6f3fd9e4102e84339`.

## Qualification and retained failures

- 561 focused host tests pass; compiler/shared-SCF checkers report zero errors.
  Earlier host fixture failures remain historical receipts, not successful runs.
  A further 139 focused default-inventory/policy/publication tests pass on the
  final non-product overlay; no successful GPU cohort is rerun for these edits.
  Five artifact-specific offline regressions additionally authenticate this
  bundle's source, all numerical/timing records and the retained safety failures.
- GPU 7323 passes 64 independent source-major derivative calls and five FCI
  complete-force cases. GPU 7334 additionally passes 64 combined-output calls
  directly contracted against CPU ERIs. Spin, opposing-density, coefficient-mask,
  both geometries, Cartesian/spherical and exact-zero gates remain unchanged.
- Admission 7324 observes 20 complete unset-selector calls, with real cache/generic
  low/seven/eight routing and no wrong-width or unknown/obsolete consumer.
- Full memcheck passes both layouts with zero errors. **Unfiltered racecheck
  7325 times out after 15 minutes**, driver exit 143; it is not qualified.
- Filtered racecheck 7330 uses toolkit-supported
  `--kernel-name kns=direct_force_spatial_spd_low_kernel`, with zero hazards.
  Its parent exits 1 because a four-layout coverage assertion exposes that the
  original oracle exercises separate output only. This failure is retained.
- Additional combined probe 7334 passes full memcheck and filtered racecheck.
  Qualification-only observers record separate `[0,24,0,24]` and combined
  `[24,0,24,0]` cached launches, with zero wrong widths/unknown variants. Together
  they non-vacuously cover all four variants; **no observer is loaded in timing**.
- Dependency jobs 7327/7328 and preflight 7331 exit 1 before any timed sample.
  Their failures are retained; no samples from them are discarded or pooled.
- Static final-linked variants have REG255/SHARED4173/LOCAL0. STACK is
  RKS combined16840/separate25512 and UKS combined16824/separate25176 bytes.
  These fields do **not** prove runtime peak memory or spill-free execution.

Earlier two-pair spatial pilots and rejected scheduling/source histories remain
separate in `prototype-receipts.json.xz` and the linked Agent Notes. Their initial
96-atom density failure is not explained or erased by the successful shipping
cohort. Slurm accounting is unavailable here; actual srun exits, job-local traps
and terminal queue checks authenticate completion, not invented accounting data.

## Storage-only compaction

This bundle fits the unchanged 64-MiB aggregate and 2-MiB change-review budgets
by whitespace-only re-encoding seven older JSON members, saving 668,780 bytes.
`storage-compaction.json` records their original Git revision, original/stored
sizes and SHA256s, and identical canonical parsed-data identities. Every parsed
measurement, array order, tolerance, failed status and historical decision is
unchanged. Paths remain unchanged for existing readers; original exact bytes
are recoverable from the recorded Git revision. Large-member review rows are
updated to current storage identities. This is not deletion of slow/failed
samples, a new scientific qualification, an increased budget or external backup.

## Decoding, verification and replay

`raw-receipts.json.xz` contains byte-preserving text members under `members`, with
individual SHA256/size records, plus `omitted` identities for retained local large
logs, manifests and actual NPZ files. `prototype-receipts.json.xz` contains selected
earlier frozen evidence, explicitly excluded from promotion. Every tracked file
is below the hard 1 MiB limit. `publication.json` authenticates all selected bytes;
`validation.json.gz` contains 120 quantitative E/F block checks for all 60 calls.
Intermediate `source-freeze`, safety and continuation-state receipts retain their
then-pending decisions. The final `promotion-decision.json` and two completed
independent analyses supersede those decisions, without rewriting historical bytes.

Decode text members without changing their original bytes:

```python
import hashlib, json, lzma
from pathlib import Path

payload = json.loads(lzma.decompress(Path("raw-receipts.json.xz").read_bytes()))
destination = Path("decoded")
for name, member in payload["members"].items():
    data = member["text"].encode("utf-8")
    assert len(data) == member["bytes"]
    assert hashlib.sha256(data).hexdigest() == member["sha256"]
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
```

Source reconstruction is `git archive` of the exact base, followed by the decoded
production or original qualification patch. Configure a Release CUDA build with
`GENERATIVEQC_ENABLE_CUDA=ON`, `GENERATIVEQC_CUDA_ARCHITECTURES=120`,
`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, the stationary AOT providers, and verified
compiler launchers. The preserved CMake configuration and actual commands in raw
receipts specify the remaining build/toolchain settings. Build without bypassing
the shared compiler cache; compare source manifests and artifact identities.

The recorded site environment is retained as `retained-env-next.sh` and
`retained-env-base.sh`; their absolute paths require the recorded site layout or
explicit relocation. Place `retained-paired-ao-endpoint.py` at
`$ROOT/evidence/bulk-point-v7/paired-ao-endpoint.py`, and retained reference JSON at
the corresponding `reference-{48,96}-retained-perf5.json` paths. The basis input
already lives at `benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`.
For a clean rerun, use fresh evidence directories and the recorded environment/
selectors from `run-performance-v3.sh`; its one-shot admission/source preconditions
must be satisfied rather than removed silently. The complete command is:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:20:00 \
  bash run-performance-v3.sh 48
```

Run 96 separately with the same finite command and argument `96`. Preserve the
Slurm-assigned `CUDA_VISIBLE_DEVICES`; do not run on maintained-excluded node n3.
The runner produces new NPZ seeds and uses `analyze-spatial-endpoint.py` to check
their exact bytes and blobs, own-before/after equality, cross-arm drift, work,
numerical gates and interleaved timings. Actual original NPZ bytes remain ignored
locally and were independently verified before publication; their digests **do
not reconstruct those bytes**. Offline E/F/raw-timing/publication checks need no
GPU, but cannot replace NPZ validation if the local blobs are unavailable.

No Release, tag, release asset, external archive or publishing workflow is created.
