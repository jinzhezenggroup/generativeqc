# Forward-only distinct W seeds for complete CUDA DF forces

Force extension of #2232, independent of #2298/#2299/#2300. Measured frozen
parent `34dd51ed3ebc5b03b17f095de2cd7b39903923c8` plus the retained owner,
private-diagnostic and benchmark overlay. Generated mathematics, six independent
cotangents/pullbacks, precision, trial-state DIIS, replay and Q8 are unchanged.
Both arms use matching headers/binaries; the candidate library, complete endpoint
and force/primitive probes are rebuilt for the private diagnostic layout change.

Four fresh cold ABBA force processes, ethane230/o9/v221/Q488, FP64, DIIS8,
64 GiB, CC Q8/Lambda Q32, finite Slurm 2905/node2 RTX PRO 6000:

| Median seconds | Original seeds | Forward copies |
| --- | ---: | ---: |
| Complete wall, startup through teardown | 275.614076 | 274.137995 |
| Joint triples response | 39.289207 | 37.652091 |

Observed complete saving **0.5356%**; both candidates beat both controls.
Descriptive scoped evidence, not statistical/global performance or latest-master
whole-library timing. Numerical scope accepted; broad production remains not-run.

405 vector cubes, 1,953 distinct seeds; candidate **3,906 W GEMMs**, **477 D2D
copies**, **41,189,373,576 logical copy bytes**. Original six-seed inventory implies
4,860 W GEMMs; baseline did not expose a per-W counter. Both native Fock work
ledgers reconcile exactly from this inventory and the unchanged 357 panel GEMMs:
9,924,048,505,176 -> 8,739,854,014,866 summands, saving 1,184,194,490,310.
Reverse triples work stays 9,224,865,365,040. CC work and Lambda/Z actions stay
unchanged. Complete logical numeric capacity stays 19,509,839,897 bytes; no new
numeric scratch/provider. Copy payload is not measured DRAM traffic or peak VRAM.

**38** independent force/public cases, **17** joint derivative/precision/paging/
budget/unsafe-arithmetic cases and all four large energy/residual/Lambda/Z/
force-direction gates pass. Maximum force difference 1.40e-14 Eh/bohr; independent
direction error 7.11e-9. Water memcheck: zero errors. **133** inventory tests pass.

Raw receipts preserve exact lengths/SHA-256, runner, finite Slurm recipe, binary
and device identities, all JSON/resources, test XML, original baseline sources,
compiled consumer sources and both failed/successful build logs. Reconstruct
the measured source with `production-source.patch.gz`. Use the retained qualifier
and correctly rebuilt matching private consumers:
`srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1
--cpus-per-task=2 --time=00:30:00 bash qualify-feature.sh force-distinct
candidate-force-distinct 8`. Preserve assigned device visibility. No Release,
release asset/tag, publishing workflow or external archive is created.
