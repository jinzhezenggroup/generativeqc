# PBE0 H100 Direct/DF complete endpoint matrix

Storage update: restore the historical payloads with [the offline recovery commands](../h100-history-retention-20261031/README.md) before running the replay examples below. Original scientific results and gates are unchanged.

Refs #2731 and #2739. After the historical missing-companion pilot, a separate registered
single-H100 Job returned all 144 complete E+F endpoints: water48/96, Direct/DF,
Standard/Fast/GPU4PySCF, three fresh processes per group. Offline replay validates
all 620 retained raw members, complete force binary64 vectors, independent
level-5 energy/force gates, level-4/5 stability, actual work and process statistics.
This is a focused evidence candidate; parent closure awaits independent review.
The historical spent pilot and its failures remain separate and unchanged.

Run `python -I -S -B replay.py` using Python 3.10+; no network, solver, native
library, GPU or oracle execution is used. `evidence.tar.xz` is a standard TAR/XZ
containing every raw member, full supplements and an index of public SHA256 values
and separate original-byte SHA witnesses. Path aliases redact private canonical
roots only. For changed members, original bytes are unavailable in this public
capsule; a recorded original hash is not a hash of the public transformed bytes.
Archive modes are public packaging modes, not unrecorded remote modes.

`summary.json` gives all 12 groups: cold median/range of three independent
processes; warm median/range of three process medians, each from its own immutable
cold density. Direct and DF are distinct scientific models. All eight native
groups were slower than their same-model GPU4PySCF groups by both endpoint medians;
DF Fast did not improve these medians. The fixed #2731 reporting contract requires no positive speedup; the broader
H100 optimization/superiority goal remains unmet by this report. No
cross-model or pure-GPU/device ratio is inferred. Historical RTX 5090 values are
single-population context; independent repeats are not fabricated.
Cold denotes an application-object cold endpoint, not a clean compiler-cache or
process-startup measurement. Prebuilt core/AOT products and persistent caches
remain in use; the actual cache launcher is retained, while cache hit/miss
telemetry is UNAVAILABLE. No default promotion is requested or established.

The scientific source is frozen cbd4d807, with exact driver/core/companion,
geometry, basis/input archive and independent oracle identities retained.
The 12,130-file source is a direct canonical Git archive of the measured commit;
the qualified patch is provenance and is not applied twice. The compact source
binding retains the Git tree/archive/patch identities and published proof pointer.
Authenticated reconstruction and 12,136-member postflight integrity are separate
from the UNAVAILABLE original complete measured-source manifest. Runtime package
and Python queries from released matrix workers remain UNAVAILABLE. The separate
POSTFLIGHT_FROZEN_ENV_METADATA appendix does not fill that gap. CUDA runtime/driver
API values are retained; power, clocks, temperature, utilization and peak device
memory are unavailable. This campaign does not qualify the separate SDK foundation.

## Minimal publication scope

All 620 public raw members, full forces/binary64, endpoint timings, work, gates and original identities remain retained. The bounded producer recipes in restore.json reconstruct 360 redundant endpoint/controller/worker members exactly before all 620 public hashes are checked. The optional full postflight inventory, transport envelope and pretty report/replay sidecars are explicitly listed with original SHA256/length in supplements/scope.json. The compact postflight receipt retains full receipt SHA/count/aggregate bytes/status and selected driver/core/companion identities; its full 12,136-row inventory is not reconstructible here. Full originals remain local/in the user project and in the complete candidate option. No quantum values, gates or semantic work are omitted.

RMS reductions use `math.fsum` so Python 3.10 and newer interpreters reproduce
the same strict derived summary. All original binary64 values, raw member
hashes and scientific gates stay unchanged; no approximate summary comparison
is used. The earlier naive-sum Linux replay refusal remains in the PR evidence.

## Storage

The existing 80 MiB aggregate and 2 MiB change-review limits remain unchanged.
Room for this live numerical report comes from separately verified Git recovery
of four superseded HOST producer/failure capsules in
[`h100-direct-wb97mv-20261024`](../h100-direct-wb97mv-20261024/README.md#lossless-storage-and-availability).
All archived bytes were stock-restored from a protected published ancestor before
retirement. Its 98 actual E/F returns, 72 formal returns, 10 NOT_RUN, independent
oracles and full runtime/scientific/source originals remain in-tree and still
replay offline. Optional HOST history is audited explicitly. This storage change
adds no numerical qualification or current-default performance claim.

## Historical payload recovery

The large original payloads for this historical source are stored byte-for-byte in existing Git history. Numerical summaries, unfavorable outcomes, original hashes and test fixtures remain in this checkout. Follow [the offline recovery commands](../h100-history-retention-20261031/README.md) before replay. Recovery does not rerun chemistry or promote this source to current production acceptance.
