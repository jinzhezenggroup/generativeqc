# Decision: compare matching CPU and CUDA spin identities

Status: implemented
Date: 2026-10-25

PR #2276 compared a CUDA descriptor's aggregate two-spin-layout hash with the
CPU polarized point hash. After explicit restricted lowering landed upstream,
these intentionally differed. All 47 automatic CUDA KS lifecycle/policy tests
then failed during preparation with "CPU/CUDA scientific identity mismatch".

Keep the aggregate identity, which binds both actual CUDA layouts, and retain
their individual mathematical identities in the private library-owned AOT
descriptor. Compare the polarized CPU identity and, for explicit total-density
LDA, its restricted identity separately. Require matching ingredient masks and
domain versions. Non-LDA equal-spin lowering retains the canonical polarized
contract; it does not impersonate a separately registered total-density CPU
descriptor. A missing or stale identity remains a hard admission failure at both
method preparation and detached final-state validation.

This extends a private, rebuilt descriptor, not the public C ABI or launcher
signatures. It changes no generated mathematical expressions, work floors,
precision gates, or optional mechanism qualification. Removing identity checks
or replacing the aggregate with only its polarized component would lose
restricted-science authentication and was rejected.

The first incremental rebuild exposed a second mismatch: the CPU generator's
import depfile omitted compiler/data bytes read only for scientific identity.
CUDA had a complete inventory dependency; CPU retained stale hashes after a
CUDA compiler module changed. Both now track the common/integral/XC Python and
JSON inventory and pinned Libxc inputs. The executable CMake regression changes
unimported inputs and adds a new module, checking regeneration and no-op builds
for CPU and enabled/disabled CUDA packages. Cache clearing is unnecessary.

Compiled host tests execute actual emitted descriptors against independently
generated CPU identities for LDA/GGA/meta-GGA and reject stale, missing, swapped
aggregate, mask and domain inputs. The opt-in real-GPU lifecycle suite remains
the independent PySCF energy/density/Fock gate; its command and results are kept
with the repair evidence. Empty-beta physical birth remains separately blocked
as documented in `../../rejected/2026-10-25-tpss-empty-spin-raw-work-clearance.md`.

## Validation

The full real-GPU suite passed all 47 cases on node1/main, `gpu:5090:1`, CUDA
12.9.1, sm120, in 265.49 seconds. Eleven representatives cover LDA/GGA/meta-GGA,
both spins, selector and MethodIR inputs, and cold/warm/moved endpoints. Three
negative cases reject unqualified execution policies. The 44 lifecycle receipts
retain all 132 phase rows with PySCF 2.14.0/Libxc 7.0.0 comparison errors,
endpoint times and semantic work. Physical arrays are compared during the run.
This is numerical regression evidence,
not a paired performance qualification. All original error gates remain intact.

Final focused host checks pass 164 tests (build invalidation, emitted identity,
final-model/resource admission and precision census). Changed-file pre-commit,
ownership, compiler structure and retention checks pass. The LDA test now
expects its actual total-density domain, and receipts distinguish aggregate,
polarized and restricted identities. After the final null-aggregate guard was
hardened, the exact final source passed 15 selected GPU cases covering the three
families, both spins and both inputs, plus the native final-state suite,
DF-range-exchange CPU-oracle control and precision census. The guard is also
covered by negative compiled host tests; no numerical expression changed.

Reproduce with the eleven-function AOT selection listed in
`tests/python/test_libxc_automatic_point_native.py`, then:

```sh
export PYTHONPATH=python:.
export GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so"
export GENERATIVEQC_AUTOMATIC_CUDA_KS_TEST=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python -m pytest -q tests/python/test_libxc_automatic_cuda_ks.py
```

Slurm visibility is preserved. Raw logs and receipts, including failed
intermediate attempts, are retained in ignored `.artifacts/repair-20261025/`
and `/home/jzzeng/tmp/qc-pr2276-repair-20261025/evidence/` on n1.
