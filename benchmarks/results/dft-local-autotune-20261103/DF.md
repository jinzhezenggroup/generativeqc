# Bounded native density-fitting policy qualification

`df-receipts.json.xz` retains raw reports, independent fitted-HF references,
all accepted/rejected candidate pairs, fresh holdouts, native allocation/work
receipts and actual recipe replays. It is compressed JSON, not a native binary
or a globally activated profile. Earlier campaigns retain their original source
identities; integration checks do not relabel old measurements.

## Admitted scope

FP64 RHF, at most 16 orbital and 128 auxiliary AOs, full-rank auxiliary metric,
fixed relative cutoff `1e-10`. UHF is refused because the native endpoint does
not expose a complete observed Fock census. Composed KS/DF resource ownership
is not qualified. Neither limitation is bypassed by disabling ResourceBudget.
The independent CPU reference supports RHF/UHF, but that does not qualify a
native UHF tuning consumer.

Orbital and auxiliary decimal basis tables are fixed before search. PySCF
2.14.0/libcint independently constructs integrals and fitted-HF analytic forces
including auxiliary response from those same input tables. Named-basis aliases
with different rounded constants are not substituted. Absolute oracle gates
are `1e-9` Hartree, `1e-8` maximum force and `3e-9` force RMS. Precision, basis,
metric threshold, screening and rank are not performance knobs.

## Initial source-matched campaign

Source `ef8336881`, native input identity
`3469a459afca5f7a59845dad262fcbbccc99cc6b44f036ab0807577184e69157`.
Slurm jobs 1448/1449 on n4 RTX 5090/SM120, CUDA 12.9.86, GCC 12, normal Release
with fast compilation disabled and verified ccache launchers.

| Workload | Fixed DF workspace hint | Result |
| --- | --- | --- |
| Water, STO-3G / auxiliary def2-SVP | 64 MiB | Packed values win; independent replay succeeds |
| Water dimer, same bases | 1 MiB | Baseline correctly refuses insufficient metric/tile workspace |
| Water dimer, same bases | 8 MiB | All six candidates fail the performance gate; retain incumbent |

Water selects `GENERATIVEQC_DF_VALUE_STORAGE=packed`. Its fresh five-pair
holdout is **13.386 -> 12.665 ms, 1.057x**, bootstrap lower 95% **1.055x**.
This is the complete frozen-density warm energy plus analytic-force endpoint,
excluding preparation, cold convergence and warmup, not a kernel-only result.
Each sample retains three observed Fock builds and metric rank 24. The complete
admission-ledger peak rises from 746123 to 795963 bytes; this timing win is not
a claim of reduced total memory. The 64 MiB hint is a DF workspace input, not
the complete composed-memory ceiling.

The dimer retains metric rank 48 and three observed Fock builds at 8 MiB. All
six candidates pass independent numerical, work and native-resource gates,
but none passes the required 1.02x significant endpoint improvement. These
negative results, including the very slow serial derivative mapping, remain
in the archive. No dimer winner or streamed-path speedup is claimed.

The real GPU suite checks every proposed DF policy, including full-search
policies, against independent E/F, observed full-rank/Fock work and native
resource admission. Initial focused CPU validation passes 183 tests with 11
optional cases skipped; pre-commit passes.

## Updated-parent integration

Source `e9c8b3a00`, native input identity
`413127726534f605780873d80f46ed23e717969d07423c3e067a6293cbd32496`,
is rebuilt normally after integrating the updated XC/J/K parent and its master
changes. Slurm job 1450 independently passes all **10** opt-in XC/J/K/DF GPU
tests. The broader focused CPU suites pass **187**, with 11 optional cases
skipped, including the compiled library/source ABI check.

Water again selects packed values: fresh holdout **13.378 -> 12.660 ms,
1.057x**, lower 95% **1.052x**. The maximum independent energy/force errors
are `7.11e-14` Hartree / `1.10e-12` Hartree/Bohr. The 8 MiB dimer again passes
all six policies' numerical/work/resource gates but produces no winner.
These reports are retained separately from jobs 1448/1449; timing samples
from different source identities are never pooled.

The accepted DF recipe actually replays under CUDA with three observed Fock
builds. The same integration job also repeats and replays all three J/K
recipes, still selecting Rys Coulomb rather than proving a K-specific win:

| Integration J/K workload | Fresh holdout baseline -> candidate, ms | Speedup / lower 95% |
| --- | --- | --- |
| Water PBE-RKS/def2-SVP | 82.359 -> 71.021 | 1.160 / 1.154 |
| Water PBE0-RKS/def2-SVP | 113.287 -> 101.911 | 1.112 / 1.102 |
| Water PBE0-UKS/def2-SVP, cation doublet | 218.781 -> 196.430 | 1.114 / 1.112 |

H2 PBE-RKS/sto-3g XC search installs a tile-512 profile in the isolated
qualification cache. Actual Calculator replay selects 512, preserves explicit
tile 64 and retains native tile 256 under a 1 GiB ResourceBudget, with matching
energies and forces. The archive retains these separately labelled integration
reports and the runtime transcript; the user's default profile cache is not
changed. Local pre-commit and the offline Libxc catalog check pass.

## Reproduction and consumption

Decode the archive with Python `lzma.open`; each campaign run's `report.workload`
contains Bohr geometry, charge/spin, batch, immutable basis names and fixed
workspace hint. Build its exact source revision with SM120 Release, fast
compilation disabled and verified compiler launchers. Install the autotune
reference extras. Run under a finite Slurm allocation on an available GPU,
preserving Slurm's assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 bash -lc 'python -m generativeqc.tune_cli water.xyz \
    --method rhf --component df --basis sto-3g --auxiliary-basis def2-svp \
    --df-memory-mib 64 --units bohr --representation spherical \
    --repeats 5 --max-candidates 6 --budget-seconds 600'
```

Use an isolated `GENERATIVEQC_PROFILE_CACHE`. An accepted recipe is replayed
with `python -m generativeqc.execution_tune recipe.json replay.json`, also under
finite `srun`, before replacing its referenced native library. Replay rechecks
the exact source/device/library/workload and raw holdout gates without a new
production CPU solve. Ordinary Calculator instances and the user's active
profile index are unchanged. New machine/noise conditions can select another
winner or none. No speedup is generalized to larger systems or KS/DF.
