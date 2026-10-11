# Selective canonical ERI residency: scoped RHF E+F experiment

The measured source is frozen master `637466c624c4c4c4dd4eb7c511a2c88675744325`
plus `measured-endpoint-source.patch.xz`. It uses an experimental 14-bucket mask
`66058240`, retaining 2,146,754,256 FP64 value bytes under a 2-GiB additional
ceiling and a **4-GiB complete numeric budget on both sides**. Master's full
4,328,718,848-byte value inventory cannot fit this complete budget even before
mandatory metadata, solver and scratch reservation. Its existing exact fallback
is therefore a real baseline, not an artificially disabled cache.

`comparison.json` recomputes the local paired gates; `samples.json` retains all
24 complete phases including full forces and exact iteration/Fock/response work.
`oracle.json` is independent tightly converged CPU PySCF/libcint at the base and
0.05-Bohr moved geometries. `source-observers.json` retains completed native
source inventories, selected bytes, uncached recurrence counts and reference
numeric peaks, together with actual interleaved process order from trace clocks.

The endpoint is bounded **physical-reference RHF export plus native RHF frame
response and nuclear assembly**, not ordinary public HF/PBE0 SCF. Source values
are regenerated for their own geometry; scalar final audits independently
recompute the source. No chemistry, screening or FP64 precision policy changes.
The default complete-cache/automatic policy is unchanged. These are three local
fresh-process pairs on one Slurm-assigned RTX 5090, not shared formal/global
performance or default promotion. Uncached full compilation cost and actual
context/stack/allocator-rounded VRAM peak were not instrumented; complete numeric
admission, finite source audits and fallbacks are separately checked.

## Reproduction

The compact patch preserves every compiled native change, the complete endpoint
and oracle clients, and the real-GPU response tests. Optional profiler/selector
and host-test edits remain recoverable from the original full patch at Git
revision `0d27821557ba77bfd6e1a65316ed88bc20061fbd`, path
`benchmarks/results/selective-eri-rhf-partial-20261011/measured-source.patch.gz`.
Its exact checksum and byte size are retained in `publication.json`; it is not
required for endpoint reproduction. All numerical records remain lossless gzip
JSON, readable through the shared publication readers. The reproduction script
restores the exact input bytes from `basis.input.gz` before launching Slurm.

Create two checkouts of the pinned master; keep the baseline immutable and apply
`xz -dc measured-endpoint-source.patch.xz | git apply -` to the candidate. Build both
with identical Release, CUDA 12.9.1/SM120, `portable_cuda` AOT configurations,
reusing supported CXX/CUDA `ccache` launchers. Use the same CPU-linalg fallback
configuration when reconstructing this run; CUDA physical work remains on GPU.
Set the roots and dependency/toolkit/Python paths described in `reproduce.sh`.
Run it from a controller exposing `main --gres=gpu:5090:1`, such as n1, never n3.
All real-GPU work goes through its finite `srun` allocation; do not override
`CUDA_VISIBLE_DEVICES`. Identity hashing, input deserialization, process startup
and CPU oracle work are explicitly outside the production phase timings.

## Retained boundaries and negative evidence

- n1 job 7436: 58 independent resident-provider tests pass; job 7451: all 88
  native frame-response GPU tests pass.
- Job 7439: actual moved partial provider memcheck reports zero errors/leaks.
- Job 7447: zero-mask, zero-ceiling and one-byte ceilings complete 12 E+F phases
  with no cached values and independent force error below 1.246e-9 Eh/Bohr.
- Job 7445: the initial 2-GiB complete budget cannot admit the selected 1-GiB
  subset after mandatory state; both sides correctly retain exact Direct.
- Job 7454: one full 1-GiB-subset/3-GiB-budget pair passes numerical/work gates
  but has no Cold improvement (357.955 s versus 356.296 s). It is a screening
  result, not a repeated statistical conclusion.
- Job 7457: the larger subset's Cold-only preflight is not same-device paired
  qualification. Only complete job 7464 supplies the three-pair comparison.

Raw stdout, scheduler states, full profiler streams, interrupted screens,
compiler-cache logs and native binaries remain in ignored local/remote artifacts;
this compact bundle retains the values required to recompute the scoped claim.
