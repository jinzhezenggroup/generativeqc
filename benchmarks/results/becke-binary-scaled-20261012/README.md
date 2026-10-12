# Bounded binary-scaled Becke products — October 12, 2026

This is source-matched complete E/F evidence on master `6f3840ad7`, CUDA 12.9.1,
SM120 Release, RTX 5090, Slurm node1. It is not a kernel-only or cold-speedup claim.
The user waives a 2% minimum gain, but the robust noise, independent scientific,
native input/export/ownership, and no-extra-SCF/Fock gates remain intact.

| Atoms | Endpoint | Baseline median s | Candidate median s | Gain | Robust floor |
|---:|---|---:|---:|---:|---:|
| 96 | warm | 12.965453807 | 12.540530946 | 3.27735% | 0.35562% |
| 96 | moved-warm | 12.963502206 | 12.486285910 | 3.68123% | 0.71481% |
| 48 | warm | 4.517080698 | 4.443798706 | 1.62233% | 1.29186% |
| 48 | moved-warm | 4.497400131 | 4.445111461 | 1.16264% | 1.57429% |

Job 7645 retains five interleaved pairs per geometry, 30 complete E/F calls per
size, all preparation calls and all 40 timed calls. Every timed call is 1 SCF /
1 Fock. Cold is 17/18 versus 17/17 at 96 and 19/20 versus 19/20 at 48; moved
preparation is 12/12 for both sides. The 48 moved portfolio fails the noise gate
and is retained as failed; it does not qualify automatic promotion. Automatic
admission is narrowed to 96–128 atoms with existing phase storage, center geometry
and no primitive-adjoint alias. Only 96 has a claimed endpoint gain here; 128
has numerical/routing coverage, not an endpoint speedup claim.

## Retained Capsule

- `raw-receipts.json.xz`: both complete endpoint records, source/binary hashes,
  independent E/F errors, frozen native input checks, work counts and failures.
- `validation.json.xz`: verified cache/build/source provenance, 2165 host tests,
  32 final policy contracts and 18 final native CUDA tests.
- `measured-source.patch.xz`: reconstruct the measured broader-admission image
  from the pinned master; no larger-XC or other pending optimization is included.
- `final-admission.patch.xz`: the subsequent host-only narrowing of automatic
  selection, with its executable tests. The selected 96-atom device equations,
  storage and launches are unchanged. Final routing is verified on a rebuilt
  source-matched artifact; full timing samples are not rerun until lucky.
- Reproduction drivers and independent inputs remain recoverable from existing
  Git object `43ad1b336`, at this directory's `drivers-and-inputs.tar.xz` path.
  The current tree does not duplicate that historical payload or relax the
  aggregate evidence budget.

The device numerical contracts cover explicit/default selection, small-owner
fallback, serial/cooperative normalization, implicit/external inputs, 128 atoms,
primitive scratch alias refusal and invalid publication. Scaled products have
independent Decimal product/derivative, finite-difference and invariance tests.
No approximate reciprocal, cutoff or tolerance relaxation is used.

## Reproduction

Apply the measured patch to the pinned master to reproduce its exact scientific
source. Build Release/SM120 with a verified sccache or ccache CXX/CUDA launcher,
retaining before/after statistics and never disabling the shared cache. Required
targets are `generativeqc` and `generativeqc_stationary_pbe0_rks_spd_manifest`.
For the final narrowed policy also decompress and apply `final-admission.patch.xz`; test it with
`tests/python/test_becke_scaled_native.py` and, inside finite Slurm,
`tests/python/test_becke_scaled_native_cuda.py`.

Recover the driver/input archive into an ignored experiment root with
`git show 43ad1b336:benchmarks/results/becke-binary-scaled-20261012/drivers-and-inputs.tar.xz > .artifacts/becke-drivers.tar.xz`.
Extract it there. The recorded
recipe names the original source/environment root and must be adapted to the
reconstructed checkout and installed Python/CUDA dependencies. Retain the same
scientific inputs, five-pair order, force policies and gates; do not import or
refine the saved final density to manufacture matched inputs. All real-GPU
execution requires finite `srun --partition=main --gres=gpu:5090:1`; preserve
Slurm's `CUDA_VISIBLE_DEVICES` and do not use node n3.

Full logs, native binaries, source capsules and native checkpoint arrays remain
in ignored local/remote artifacts. The retained values suffice to recompute the
reported medians, robust floors, no-extra-work and scientific acceptance. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-12-becke-binary-scaled-products.md)
and issue #2288 for earlier failed directions and the changed acceptance policy.
