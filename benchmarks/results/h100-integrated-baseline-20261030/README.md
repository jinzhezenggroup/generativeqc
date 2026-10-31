# H100 integrated baseline: complete 99-process E+F matrix

Storage update: restore the historical payloads with [the offline recovery commands](../h100-history-retention-20261031/README.md) before running the replay examples below. Original scientific results and gates are unchanged.

The frozen `000517f82dc62130459a6fd92a27518a158c49cc` baseline returns
396 qualified complete energy/analytic-force observations from 99 processes.
At the selected whole-matrix `tile1024` policy, 19/30 physical primary inputs
have a single-process GPU4PySCF/native warm ratio above 1.05; the complete-set
geometric mean is 1.183780. Direct contributes 8/16 and DF 11/14 such ratios.
The alternative `tile256` policy has 13/30 and geometric mean 0.938940.
Both complete policies, including every unfavorable observation, are retained.

These are exploratory one-process comparisons. The original final protocol
still requires at least 20/30 physical inputs, three fresh independent process
pairs per input, median paired ratio >1.05, every pair >1, all scientific gates,
and complete-set geometric mean >1. Warm repeats and policy arms add no votes.
This record does not accept that target or qualify the later Direct candidate.

## Conditions and scientific checks

- Native source: `000517f82dc62130459a6fd92a27518a158c49cc`; actual core
  SHA-256 `87f4d770773d3597da62a9cfa41fde28f882f5244ba692fa6f74962873835dfd`.
  The build receipt binds all 33 core/AOT artifacts and sixteen SM90 AOT domains.
- Actual H100 80GB HBM3, SM9.0, MIG disabled, driver 595.58.03, 700 W;
  UUID `GPU-61443ba8-021b-abb8-7c7b-f7a60647a191` for all 99 processes.
  Allocation: 20 CPU, 200 GiB memory, 16 GiB shared memory. BLAS/OpenMP
  consumers use one thread each. Execution is a finite qz Job.
- 66 native Standard processes: the same 33 cases at each of two global
  tile policies. They share 33 fresh stock GPU4PySCF 1.9.0 reference processes
  on that same physical allocation. The runtime audit checks files from its
  selected wheels and records base dependencies' versions and import paths.
  It does not byte-authenticate the imported base PySCF tree or the complete
  dependency closure. No Fast arm is in this record.
- Each process requests one cold and three warm complete E+F calls, using
  its own frozen cold density for warm seeds. Host timing includes synchronized
  CUDA execution and complete forces returned to host.
- All 396 E+F observations pass the original independent oracle gates:
  E/atom <=1e-5 Eh, Fmax <=1e-4 Eh/Bohr, FRMS <=3e-5 Eh/Bohr. Replay also
  crosscompares every native sample with every fresh reference sample for
  its case: all 1,056 full-vector comparisons pass those same gates.
- HF DF uses the explicitly matched cc-pVTZ-JKFIT accuracy candidate.
  Both WB97M-V DF coverage cases remain preexcluded because their approximation
  contracts differ. The tiny HF control remains outside the 30 primary inputs.
- The historical oracle-reuse audit explicitly records fourteen unavailable
  denser-grid producer byte streams. Existing oracle vectors, scientific inputs
  and unfavorable history remain retained. This packet does not reconstruct
  those missing producers or replace the separate six-case reference replication.

## Offline replay

From the repository root:

```bash
uv run --no-env-file --no-project --python 3.12 --with numpy==2.2.6 \
  python benchmarks/results/h100-integrated-baseline-20261030/replay.py
```

The reader reconstructs every force list before checking its normalized JSON
hash; checks all 509 original campaign members, the supplementary admissions,
case order, physical denominator, source/core/AOT/input/device/allocation and
producer bindings; directly checks the published default and explicit HF input
exports; recomputes every scientific qualification; and recalculates
both whole-policy summaries. Explicit exceptions keep checks active under
`python -O`. It does not execute archived campaign producers or launch compute.

`forces.npy.xz` contains all 83,052 coordinates as little-endian FP64, without
rounding or downcast. `documents.json.xz` retains scientific observations,
plans, qualifications, campaign/runtime receipts, AOT manifests and supplementary
protocol/input/build/selection admissions. `bindings.json.xz` retains original
private byte identities, normalized JSON identities, all force slots and text
boundaries. `text-streams.bin.xz` retains every original campaign producer,
stdout/stderr stream and telemetry member in the normalized representation.

Only the private shared-storage root is replaced with `<H100_PORTABILITY_ROOT>`
in JSON strings/keys and text streams. Normalized public bytes are not identical
to private originals. Original hashes remain provenance metadata; recovered
public hashes are checked separately. Replay updates only the qualifier's
canonical receipt identity for this root replacement and requires all other
qualification fields to match exactly. Scientific values and order are unchanged.

This public replay checks build artifact bindings and retained SM90/FP64 AOT
metadata; it does not replay the large build binaries or complete compiler logs.
Actual kernel attribution, native/memcheck, moved-geometry/lifetime, larger
ragged batches, full request budgets, Fast and final three-pair performance
acceptance remain separate gates.

## Selected-policy physical primary comparisons

Ratios below are fresh-reference/native medians of each process's three warm
complete E+F calls. Each arm has one process; none is a final majority vote.

<!-- The complete table is generated from the independently replayed packet. -->

| Method | Atoms | Route | GPU4/native |
| --- | ---: | --- | ---: |
| hf | 48 | direct | 4.127202 |
| hf | 48 | df | 1.377540 |
| hf | 96 | direct | 1.144788 |
| hf | 96 | df | 0.666212 |
| pbe | 48 | direct | 1.519970 |
| pbe | 48 | df | 1.655767 |
| pbe | 96 | direct | 0.922362 |
| pbe | 96 | df | 0.989606 |
| pbe0 | 48 | direct | 1.640800 |
| pbe0 | 48 | df | 1.714674 |
| pbe0 | 96 | direct | 0.828173 |
| pbe0 | 96 | df | 1.078457 |
| b3lyp | 48 | direct | 1.669744 |
| b3lyp | 48 | df | 1.812656 |
| b3lyp | 96 | direct | 0.863723 |
| b3lyp | 96 | df | 1.081062 |
| r2scan | 48 | direct | 1.668238 |
| r2scan | 48 | df | 1.738515 |
| r2scan | 96 | direct | 1.069537 |
| r2scan | 96 | df | 1.128651 |
| r2scan-3c | 48 | direct | 0.769285 |
| r2scan-3c | 48 | df | 1.104530 |
| r2scan-3c | 96 | direct | 0.539937 |
| r2scan-3c | 96 | df | 0.834331 |
| m06-2x | 48 | direct | 1.711790 |
| m06-2x | 48 | df | 1.888057 |
| m06-2x | 96 | direct | 0.955851 |
| m06-2x | 96 | df | 1.232240 |
| wb97m-v | 48 | direct | 0.711178 |
| wb97m-v | 96 | direct | 0.614919 |

## Historical payload recovery

The large original payloads for this historical source are stored byte-for-byte in existing Git history. Numerical summaries, unfavorable outcomes, original hashes and test fixtures remain in this checkout. Follow [the offline recovery commands](../h100-history-retention-20261031/README.md) before replay. Recovery does not rerun chemistry or promote this source to current production acceptance.
