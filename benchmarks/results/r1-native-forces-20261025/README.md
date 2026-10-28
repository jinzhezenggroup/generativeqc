# R1 native force qualification

> **Retention note (2026-10-28):** The current checkout retains the compact numerical/
> resource qualification receipts only. The original full-precision paired
> samples, per-case comparison table, diagnostic trace summaries, source
> inventories, ECP follow-up patch and executable reproduction drivers are
> recoverable without loss from immutable ancestor commit
> [`796f66f4fd935ca88abd885321738ea48e9f8db2`](https://github.com/njzjz-bot/vibeqc/tree/796f66f4fd935ca88abd885321738ea48e9f8db2/benchmarks/results/r1-native-forces-20261025).
> The original per-file SHA-256 values remain in `validation.json` and in
> the archived `publication.json`. Recover and verify the exact source with:
>
> ```bash
> mkdir -p /tmp/r1-native-forces-evidence
> git archive 796f66f4fd935ca88abd885321738ea48e9f8db2 benchmarks/results/r1-native-forces-20261025 \
>   | tar -x -C /tmp/r1-native-forces-evidence
> cd /tmp/r1-native-forces-evidence/benchmarks/results/r1-native-forces-20261025
> ```
>
> This retention-only change does not requalify measurements or modify any
> scientific limits. Restoring a complete campaign to a separate checkout is
> required to replay the original paired-sample audit.

The frozen implementation `685478ca1111407573a5cbfa813d63ae0a5713ce`
passes 192 paired public E/F comparisons against #2782 at
`df2ded411074d3b04b5c236b83d760ca1aafc295`. Both ABBA and BAAB orders
pass the 1.05 median-time ratio gate, with equal SCF iterations, Fock builds
and warm-start flags. Maximum median ratio: **1.0063663195**. Maximum energy
and force differences: **9.10e-13 Eh** and **8.41e-12 Eh/bohr**.

Each order has two blocks per version, three warmups and eleven measurements
per block: 22 measurements per version/order/phase. Cold, warm and genuinely
moved geometries are separate endpoints. Tracing is disabled during timing.
The cases cover eight named CUDA methods, both spins and Direct/DF, plus
36- and 48-atom PBE/DF/def2-SVP systems. MN15 DF has no admitted parent
endpoint; both new spin cases pass but are excluded from timing ratios.

| Endpoint | Cold ratio ABBA / BAAB | Warm ratio ABBA / BAAB | Moved ratio ABBA / BAAB |
| --- | --- | --- | --- |
| WB97M-V DF RKS | 0.2325 / 0.2321 | 0.7339 / 0.7308 | 0.2123 / 0.2123 |
| WB97M-V DF UKS | 0.2358 / 0.2369 | 0.7880 / 0.7916 | 0.1793 / 0.1802 |
| PBE DF, 36 atoms | 0.4725 / 0.4737 | 0.4886 / 0.4856 | 0.4315 / 0.4218 |
| PBE DF, 48 atoms | 0.5940 / 0.5959 | 0.4166 / 0.4164 | 0.5470 / 0.5573 |

The archived `samples.json.gz` retains measured E/F arrays, times and work
counters; `comparison.json` records every comparison and gate; and
`provenance.json` stores authenticated source censuses, binary identities and
the hardware/toolchain. These original files are not duplicated in this checkout.
Timing ran in Slurm job 14328; diagnostic replay ran afterwards in 14329,
on node1/main/RTX 5090 with Slurm-assigned visibility. Compiler caches were
reused. The later parent commit `5f715501` changes CI time budgets and their
tests only; it is integrated without relabeling these measured revisions.

Restore the historical directory as shown above, then reproduce from that
restored directory with clean, built checkouts at the two frozen revisions. Both need
the native CUDA library and stationary AOT compatibility packages under
`build/native-ks-catalog`. Put the matching Python/compiler dependencies and
CUDA tools on PATH, then run from this directory:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --time=01:30:00 \
  bash reproduction/reproduce.sh BASELINE_CHECKOUT CANDIDATE_CHECKOUT OUTPUT_DIRECTORY
```

The wrapper verifies source bytes against Git before each block and preserves
Slurm device visibility. The auditor treats missing blocks as incomplete.

Independent scientific checks are separate from cross-version parity:
CPU CTest 120/120; CPU named force matrix 40 passed, 12 backend skips;
final CUDA numerical matrix 70 passed, 4 backend skips; range selection
6 passed; molecular WB97M-V Direct/DF/fitted-LR memcheck 6 passed with zero
sanitizer errors. Native CUDA CLI checks cover 32 E/F endpoints with Python
and compilers absent from PATH. The committed tests include independent
PySCF E/F/D/W, finite differences, spin/angular domains, larger molecules,
corrections and transaction/lifetime failures. Retained log hashes and
selected outcomes are in `qualification.json`.

## Resource-planning follow-up

After the frozen campaign, explicit-budget ECP testing found two defects:
per-atom ECP core counts were passed to a scalar shape field, and native ECP
derivative scratch was charged with the energy-setup lifetime. The follow-up
reduces the core counts and charges ECP/one-electron force temporaries while
SCF and stationary arenas are live. It does not modify force equations or
the measured all-electron endpoints. the archived `ecp-resource-followup.patch` preserves
the exact measured changes from the frozen implementation for the relevant
sources and tests; `ecp-resource-followup.json` records source/binary hashes.

Slurm job 14348 passes all six ECP budget/replay/failure cases. Both Cartesian
and spherical RKS/UKS fleets use native forces, with zero rejected allocations.
Measured explicit-device plus provider-reservation peaks are 216,199,192 bytes
(RKS) and 216,060,080 bytes (UKS), below the respective planned limits
216,652,669 and 216,513,616 bytes. CPU CTest remains 120/120; resource/CI-budget
tests pass 130 cases, and preliminary-guess tests pass 62 cases.

## Limits and unsuccessful runs

The CUDA API AUTO-precision provenance assertion fails on both the unchanged
parent and candidate (`method=6 bits=0 strict=0 iterations=0`); it is preserved.
Earlier source-sync, extracted-harness, stale refusal expectations, ECP budget
and WB97M-V schedule failures are not relabeled as passing runs. The final
LR schedule shares the existing specialized omega=0.3 launcher.

The archived `trace-summary.json` retains provider counters and original
diagnostic stream hashes; full profiler streams remain in ignored local artifacts. These counters
are not whole-process RSS or simultaneous device peaks. The follow-up ECP
ledger covers explicit owned buffers and provider reservations, excluding
driver/pool and opaque library internals. No cold compilation cost or
independently recomputed per-sample residual trace was measured. Accordingly,
the publication accepts numerical parity and records the separate endpoint
non-regression result; it does not claim universal performance promotion,
complete R2 resource accounting, or closure of the entire #2227 tracker.
