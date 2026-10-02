# Decision: PBE0 complete-endpoint baseline without changing production policy

Status: implemented
Date: 2026-10-01

## Problem

The old README DFT matrix timed SCF energy only. Those PBE0 observations cannot
be compared with HF's energy-plus-host-force endpoint or used to characterize
analytic-force scaling. The user prioritizes a PBE0/def2-SVP baseline while
requiring preservation of all ongoing OMol25 performance work.

## Decision

Share the existing OMol25/HF-style independent-process endpoint runner and
validator through an immutable method/basis/schema description. PBE0 uses its
registered `pbe0-rks` native identifier and the common moving-grid GPU4PySCF
adapter, but never attaches WB97M-V's VV10 density-domain context or nonlocal
grid records. Keep both molecular geometries and all five frozen engine-local
warm replays. Observe already-produced native force-work metadata through a
benchmark-local wrapper; do not enable resource/profile/performance switches
or rerun a scientific force solely to obtain counters.

Validate every call against the independent original/moved oracle before
plotting any native median. Retain exact raw journals as deterministic gzip,
with decompressed hashes bound to compact JSON. A missing/nonconverged reference
causes an explicit skipped-native outcome, not a native timing or an invented
reference. Keep the current production force guards unchanged.

## Rejected alternatives

- Reusing old energy-only DFT times would silently omit analytic forces.
- Copying the full WB97M-V runner would duplicate the endpoint contract and
  risk applying VV10 policy to PBE0.
- Removing the global-hybrid 128-AO admission guard to populate 24–96 atoms
  would assert unsupported capacity without a qualified resource inventory.
- Loosening GPU4PySCF convergence only at failing sizes would no longer be the
  stated HF-matched matrix. Failure remains a scientific/status result.

## Evidence and consequences

Slurm 11967 supplies independent GPU4PySCF reference endpoints; 11969 qualifies
all 36 native calls at 3/6/12 atoms (24/48/96 spherical AOs). Native original
warm medians are 1.281241/3.612441/15.322144 seconds versus reference
1.088028/1.842444/1.859755 seconds. Native is slower at every qualified size,
despite taking one warm SCF step; counts and full-endpoint timings are not
iteration-normalized. Cold includes runtime compilation/cache setup, which
is particularly large for the first native size.

The 24/48/96-atom reference cold SCF fails its unchanged 100-step protocol.
Their native timings are not measured. The native force domain independently
remains bounded by 128 AOs/32 atoms and one million grid points. Neither this
baseline nor the preserved OMol25 3-atom improvement establishes through-100
atom parity. All previous production changes, notes and historical results
remain, with a pre-task patch/untracked-file/library checkpoint.

## Revisit when

Optimize the measured complete endpoint and extend authoritative resource
qualification before adding larger native curve points. Investigate reference
convergence with explicit diagnostic evidence; any changed protocol needs a
separately identified matrix rather than overwriting this baseline.

## References

- `benchmarks/readme_pbe0.py`
- `benchmarks/readme_omol25.py`
- `benchmarks/results/pbe0-def2-svp-20261001/README.md`
- `benchmarks/results/omol25-wb97mv-20261001/default-hf-cartesian/`
- `python/generativeqc/_stationary_cuda.py`
