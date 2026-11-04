# Actual-CUDA occupied-XC finite-input counterexample

On November 4, 2026, a bounded actual-device probe reproduces #2751 against
scientific source `9cfb77727661cc64e47938d9d0ba65a157427439`. This is **not a
repair**, a public molecular SCF reproduction, or an endpoint speedup.

## Method and observation

`emit_probe.py` extracts the production emitted density-product and dense feature
consumers, occupied projection and feature consumers, and native structural
witness. It does not reimplement their equations. The stored density fixture is
formed in host FP64 using explicit FMA before transfer; **density formation is
not itself a tested GPU kernel**. All inputs and published outputs are finite.

The original fixture has 16 AOs, one doubly occupied orbital, an orthonormal
frame with first columns `(1e-200, 1, 0, ...)` and `(1, -1e-200, 0, ...)`, and
one grid point with AO0 `1e200` and all other values/derivative jets zero.
The ordinary control uses an identity frame and AO0 `0.25`.

| Fixture | Structural witness | Stored D00 | Dense rho | Occupied rho | Reported error |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary control | accepts | 2 | 0.125 | 0.125 | 0 |
| Stored-density underflow | accepts | 0 | 0 | 2 | 0 |

The other four feature components are zero in both routes. Thus the problem is
observable on an actual RTX 5090, not just in the earlier CPU barrier simulator.
It still does **not** establish that an ordinary molecular SCF reaches this
fixture, or explain the independent moved/fresh PBE issue #2915.

## Source and execution identity

`observation.json` retains the device/runtime versions, exact vectors and these
identities. Generation imports the clean archived master compiler; the extracted
native witness is byte-identical between that archive and the local checkout.

- Slurm job **15784**, `node1` (`n1`), partition `main`, one `gpu:5090:1`,
  finite `00:02:00`; assigned `CUDA_VISIBLE_DEVICES=0` preserved unchanged.
- NVIDIA GeForce RTX 5090; driver API version 13000, runtime version 12090.
- CUDA 12.9, nvcc V12.9.86; C++17, `-O2 --fmad=false -arch=sm_120`.
- Verified `sccache 0.16.0` wraps compilation; existing compiler cache retained.
- Emitted CUDA SHA-256:
  `c9ddc45b091de3e48a52ac7140f90a6d77a2bacbc3d3c1059a07a2ff3bab1268`.
- Executable SHA-256:
  `74a6c841cd7c151b96ae45f2b301855eedec1506ad7ef992a310e5f39f772d6c`.

Raw source, build, scheduler and device logs remain under
`/tmp/qc-2751-*-20261104.*` locally and the emitted source/executable on `n1`.
Compiler-cache before/after statistics are retained locally; shared-server
counter changes are not an isolated cache-hit measurement for this probe.

## Reproduction

Generate without a runtime or GPU, using the scientific source named above:

```bash
PYTHONPATH=.:python:tests/python python \
  benchmarks/results/occupied-xc-finite-input-20261104/emit_probe.py \
  --output /tmp/occupied-finite.cu
sccache --version
sccache /group/software/cuda-12.9.1/bin/nvcc -std=c++17 -O2 \
  --fmad=false -arch=sm_120 /tmp/occupied-finite.cu -o /tmp/occupied-finite
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 /tmp/occupied-finite
```

Compile/run on an approved node, never node3. A successful probe process means
the observation was collected: **it does not mean scientific equivalence**.

## Required repair remains open

Structural identity/generation/shape admission cannot certify stored-density
rounding. A fix must account for its absolute residual, amplified through the
actually consumed AO jets and applicable XC energy/force contract, or enforce
a proven fail-closed domain restriction. Preserve original-D fallback,
cancellation regressions, ordinary admitted cases and admission cost in complete
endpoint qualification. Unmerged #2597 is a candidate to re-review, not shipped
repair evidence. No tolerance, default, scientific capability or issue state is
changed by this probe.
