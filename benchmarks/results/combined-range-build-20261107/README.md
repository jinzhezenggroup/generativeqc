# Direct composite exchange construction: complete cold qualification

WB97M-V/full Direct, 96 atoms, spherical def2-SVP, unchanged standard FP64,
grids, screening, convergence and analytic forces. Cold is
constructor+prepare+SCF+analytic-force+host-export. Warm is the first same-owner
call from its own frozen cold density, not a pooled warm median.

Final native source, after integrating master `dafa29f83` (#3016):
`ea83781aafef1d8e4f8159033ce12f2aed096cb7aaeb07de397d71da32ee9a13`.
SM80/SM120 Release ELF SHA256:
`e27f500d7bba457002cec893fb7762ee6a411a84846843525702082f4c97428f`.
The archive retains the build driver, verified sccache0.16.0 version, 99.68% cache
hit receipt, exact executable reconstruction receipt and original-runtime restoration.

## Same-image complete endpoints

The rollback sets only `GENERATIVEQC_KS_COMBINED_RANGE_BUILD=0`; the candidate
leaves both that control and `GENERATIVEQC_KS_COMBINED_RANGE_SOURCE` genuinely
unset. Fresh processes use the same source/image and scientific controls.

| GPU / finite allocation | Incumbent cold | Default cold | Incumbent warm | Default warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226634 | 88.980278 s | 83.334962 s | 13.913736 s | 13.918371 s |
| RTX5090 / Slurm16640 | 178.994173 s | 168.198248 s | 13.645767 s | 12.760090 s |

A100 saves **5.645316 s / 6.34% complete cold**. Its warm is effectively unchanged
(+0.03%, with identical work). 5090 improves **6.03% cold / 6.49% first warm**.
No fresh gpu4pyscf latency comparison is claimed. The retained prior default is
also checked as a historical guard, not relabeled as a new paired observation.

### Separately retained pre-master qualification

`pre-master-evidence.json.xz` preserves the original source
`56034cd22620e5ded0fd231556609642ee7595cebfe34f98961ff9b3822bc478`, ELF
`1c193f6ae50516613de16c2da1c5662d771b4edfd644e1256309f4e6cb1d37ff`.
These measurements are not observations of the final master-synced image:

| GPU / finite allocation | Incumbent cold | Default cold | Incumbent warm | Default warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226550 | 89.088205 s | 83.401132 s | 13.842545 s | 13.903494 s |
| RTX5090 / Slurm16639 | 178.695741 s | 167.787624 s | 13.652372 s | 12.697615 s |

## Actual work and scientific gates

- Both arms retain 16 cold iterations/Focks (11 provisional + 5 target), four
  cold VV10 sweeps, one warm Fock and zero new warm sweeps.
- A100 resident source outputs decrease 3,746,443,432 -> 1,873,221,716; K actions
  remain 16/1. 5090 source outputs stay 1,873,221,716, while K actions decrease
  32 -> 16 cold and 2 -> 1 warm. No warm call constructs new sources.
- The producer constructs `LR + (c_full/c_LR)Full` in one allocation. Order-3--9
  whole-shell consumers share Hermite geometry and combine radial roots before
  the common recurrence/contraction. Component fallback uses the existing paired
  evaluator: not every angular class is claimed to use one radial recurrence.
- Retained source bytes remain 14,985,773,728, below the unchanged 16-GiB ceiling.
  The second complete temporary source is eliminated. Inverse metadata remains
  bounded by the admitted allowance; no GPU-name rule or CPU solver is introduced.
- Paired energy discrepancy <=3.79e-14 Ha/atom and force discrepancy
  <=1.48e-12 Ha/Bohr. Every complete endpoint also passes the retained independent
  energy/force oracle; production imports of PySCF/gpu4pyscf are blocked.
- Six Slurm independent raw/composite, signed, displaced and f-shell fixtures
  pass with zero skips; memcheck reports zero errors. A100 independently passes
  its 96-AO composite, immutable identity and usable raw-fallback gates.
- Host quadrature covers all simplex cells through ffff, signed cancellation,
  omega zero/finite/large and incumbent derivative/reachability behavior.

## Replay and retained failures

Run `python benchmarks/results/combined-range-build-20261107/replay.py`.
The reader checks both source/image cohorts separately, including the final
cohort's immutable reference to the original evidence. `summary.json` is derived,
not the evidence of record. `evidence.json.xz` retains
complete exported observables, actual scheduler/image/source receipts, a native
and compiler reconstruction patch, build-cache receipts, independent oracle
semantic values and gate results. Acceptance remains active under Python `-O`;
corruption tests cover force, work, identity, selector, VV10, source, oracle and time.

The superseded 3f90e9d6419f build was interrupted before GPU qualification after a
host-only finite lookup and formatting correction; its original borrowed runtime
was restored/hash-verified. Slurm16638 failed an added fallback fixture that
accidentally requested derivatives: no endpoint ran. Correcting only the probe
request preserved the final native identity and ELF. All development failures and
skips remain distinct from passed GPU gates. The A100 download was hash-verified
before retiring its owned SCP EOF wait, and both finite transfer services are
stopped. The final cohort additionally retains successful Slurm/LSF completion
receipts and its own authenticated download EOF retirement. An initial local host
selection included seven CPU diagnostic tests requiring a native library absent
from this worktree; that failure log remains distinct from the passing host-only
population. The real-GPU complete endpoints independently gate actual J/K work.
Host-only GPU-fixture skips are not counted as real-GPU qualification.

Default promotion is resource/capability based. Partial/host/disk/mixed/DF,
incremental, graph-chunk and application-ledger owners retain their providers.
`BUILD=0` selects the bounded incumbent construction; `SOURCE=0` selects raw LR.
Neither rollback disables a profitable LR cache to manufacture a streaming win.

See the [decision](../../../.agents/notes/implemented/performance/2026-11-07-direct-composite-source-build.md)
and tracking issue #2976.
