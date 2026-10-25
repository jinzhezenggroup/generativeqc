# GFN2 mixer endpoint qualification

The existing native CUDA bootstrap has an explicit opt-in journal mode. It calls
`execute_restricted_gfn2_cuda` with fresh restricted SCC, FP64, history eight,
damping 0.4, 300 K, energy/charge tolerances `1e-12/1e-10`, maximum 300 iterations,
energy and host forces. It enables the existing mixer diagnostic specialization
only for the ON cache. Default bootstrap tests and production admission stay
unchanged; no extra CMake target or production timer is required.

Prepare independent references on a CPU resource with `tblite==0.6.0` and NumPy:

```sh
python benchmarks/gfn2_mixer_endpoint.py prepare /absolute/experiment/oracle
```

This uses the seven pinned small fixtures and the existing six-bohr water-cluster
recipe at 48/192 intrinsic AOs. Each original geometry has a nonrigid moved partner
and a fresh tblite reference at accuracy 0.01 and 300 K. Small original references
also pass the pinned fixture's unchanged `5e-7` energy/force gates. Shell and
orbital dimensions come from tblite, independently of the CUDA history generator.
Optional `--waters 8 32 128` includes the separate 768-AO domain; its historical
independent `5e-7` E/F failure remains unresolved. Retain every new failure or
timeout with its source identity rather than widening acceptance or substituting
peer agreement.

Use a source-matched native CUDA build and the existing bootstrap executable. If
compiling the test source directly, compile it through the verified compiler-cache
launcher and link the exact qualified native library and CUDA runtime as in
`tests/python/test_gfn2_cuda_bootstrap.py`. Record compiler/cache versions and
actual launcher commands; reuse the shared cache. Do not fabricate a Slurm job ID
to invoke that pytest wrapper outside Slurm. The standalone executable can run in
an explicitly authorized NVIDIA Job with its actual resource identity recorded.

```sh
/absolute/bootstrap /absolute/experiment/oracle/oracle.txt \
  --mixer-endpoint-probe /absolute/experiment/mixer.jsonl 5
python benchmarks/gfn2_mixer_endpoint.py report \
  /absolute/experiment/mixer.jsonl /absolute/experiment/oracle/oracle.json \
  /absolute/experiment/report.json --repetitions 5 \
  --failures /absolute/experiment/mixer.jsonl.failures.jsonl
```

The protocol retains one cold call, five cache-warm fresh-SCC calls, one moved
call, and five moved-replay calls for each OFF/ON cache. Arms alternate actual
execution order for every paired sample. Cold construction is reported separately
and charged in paired totals; later fixtures share process/library caches, so
these are not all fresh-process cold calls. ON totals include synchronous receipt
readback; file formatting/writing is outside the measured endpoint. The native
sampler measures endpoint and readback intervals in both arms, including the OFF
readback branch overhead. Both intervals must be finite and positive. Constructor
timing must be positive for cold calls and zero for retained-cache phases; absent
or zero executed intervals cannot qualify complete-endpoint timing. The native
journal flushes each settled result; the main and lifecycle journals together
are bounded to 64 MiB. Native receipt
storage is separately bounded to 64 MiB by the existing runtime; for this single
system and maximum 300 iterations its configured reservation is
`8 + 300 * sizeof(Gfn2SccMixerDeviceReceipt)` bytes. This is a configuration-derived
reservation, not a device peak-memory measurement. The snapshot API does not
expose allocator peaks or actual graph submission count; report those as
`UNAVAILABLE`, retaining the actual submission boolean and graph-family field.

Raw records retain actual atomic numbers/coordinates and full energy/forces,
convergence/status/iterations, call/plan
identity, graph family/submission/completion, attempted versus retained receipts,
ring/live-history state, actual coefficient/Gram/combination visits, completion
bits and stage `clock64` intervals. The report reconstructs chronological slot
occupancy and component visits without calling the production generator.
Slot cardinalities multiply the vector dimension to count coefficient/combination
and ordered-pair Gram visits without dimension-sized host enumeration. Empty,
truncated, partial or restarted trajectories remain `INCOMPLETE`; malformed or
identity-mismatched evidence is rejected. OFF/ON matching requires equal SCC work
and statuses plus E/F agreement within `5e-14`, independently of the unchanged
tblite scientific gate. Existing bootstrap failure/replacement/recovery tests and
host transaction tests must also pass on the frozen source.

The lifecycle sibling retains the existing water scenarios: first NaN-position
failure, healthy recovery, failed replacement with iteration limit 301, reuse,
pre-reset rejection with limit zero, and healthy reuse after rejection. These
test-only failure settings are recorded separately from the fixed positive
qualification settings. These controlled NaN-position transaction failures require
API status 6, incomplete endpoint and unchanged result sentinels. The zero-limit
phase requires synchronous API status 1, incomplete endpoint, unchanged sentinels,
no snapshot or receipts, and zero call/plan identity. Unknown failure domains are
not qualified by this protocol. Mixer completion is recorded before device
restoration, so final API status alone cannot infer transaction completion. The
validator checks call/plan rollback and snapshot invalidation, and independently
checks recovered E/F and history work.
Every available snapshot, including a failed call, must retain a nonzero candidate
plan token and device 0 for this qualification. Unavailable pre-reset rejection
keeps its zero call/plan tokens, device -1 and default graph metadata. Bounded
fallback family 0 cannot claim graph submission; families 1/2 require submission
before their available snapshot can qualify. A failed transaction may still have
submitted a graph, so submission must not be inferred from final API status.
`mixer.jsonl.device.json` records the CUDA API's actual device 0, UUID, visible
device count, runtime/driver version and compute capability. A node-level
`nvidia-smi` inventory alone may include GPUs outside the process's visibility.

Cycle shares use only intervals from **one CTA invocation**. The denominator is
the sum of its measured residual/history/solve/combination stages; it excludes
normalization, commit, launch overhead and the rest of SCC.
Every executed measured stage must have a positive interval before publishing
its share. The first invocation executes only the measured residual stage, with
history/solve/combination intervals remaining zero. A single missing executed
interval makes the receipt incomplete even when the sum of other intervals is
positive. Do not aggregate
cycles across SMs or convert them to milliseconds using nominal clock rate. A
complete SCC wall-time denominator is not exposed by the current receipt, so mixer/SCC
fraction is `UNAVAILABLE`. Report complete endpoint timing alongside local stage
shares without calling either a whole-graph mixer fraction.

The report's `PASS` means the supplied journal passed diagnostic/endpoint checks.
Real-device acceptance additionally requires an accompanying source/build/resource
manifest: frozen Git head/tree and clean status or reproducible snapshot, actual
executable/library SHA-256, CMake flags and compiler/cache identity, Toolkit,
driver/device, command/exit/timeout, input/oracle/raw hashes, resource/job/PID and
artifacts. A synthetic journal, a compiled probe, a running job, or an unbound
library is insufficient. Keep credential-bearing proxy/WSS/access URLs and
transport stdout out of this manifest and public evidence.

Restricted diagnostic qualification leaves provider selection,
unrestricted/ragged acceptance and the parent product/scientific scope open.
