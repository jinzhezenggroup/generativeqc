# Decision: enable bounded Ragged XC in ordinary native KS

Status: implemented
Date: 2026-10-12

## Decision

Make ordinary native KS request Ragged XC by default. Explicit
`GENERATIVEQC_CUDA_XC_RAGGED_BATCH=0` retains the former 32–128-AO admission;
`GENERATIVEQC_CUDA_XC_COMPACT_BATCH=0` disables compact contractions entirely.
The lower-level caller-owned plan remains explicit. This supersedes the
default-off decision in `2026-10-12-experimental-ragged-xc.md`, without rewriting
its historical source-matched evidence.

The user explicitly requests promotion based on the positive cold and moved
complete-endpoint populations. Warm differences did not pass the descriptive
2% noise floor. This is a bounded scheduling default, not an all-workload or
all-device significant-speedup claim, and it does not reinterpret the original
performance assessment as passing.

## Invariants

- Retain every selected AO map, density off-diagonal block and potential pair.
- Keep per-entry tile/addition order and unique output ownership; no FP atomics.
- Preserve strict FP64, exact J/K, original grid, screening, SCF and force gates.
- Charge all optional panels, features, local potentials and descriptors to the
  existing byte cap; preserve shrinking and one-tile/provider fallbacks.
- A public ledger with no optional arena reservation retains its incumbent path.
- Do not include the rejected ACE implementation in this PR.

## Evidence

The historical master experiment measures five interleaved complete E+F
trajectories per arm on 96 atoms / 768 AOs / RTX PRO 6000 Blackwell. Cold and
moved median time reductions are 5.124% and 4.168%; warm and moved-warm changes
are 1.682% and 1.392%, below the noise gate. All 48 independent scientific
records pass the unchanged energy/force gates. Native oracle, boundary,
capture/replay, memcheck and racecheck evidence is retained in the preceding
note. Serial scatter was measured slower and rejected.

The promotion branch starts at current master
`a50eb54f362a1b27edcb0f06353e7a04aa9979e2`. Its qualification uses an unset
Ragged control in the candidate and explicit zero in the incumbent, with clean
timing trajectories separated from diagnostics. Current-master device
qualification remains pending while the PR is a draft; retain its final results
under `benchmarks/results/ragged-xc-default-20261012/` before marking it ready.

## Revisit when

Broaden endpoint/device evidence or refine continuous resource/work admission,
not molecule/device whitelists. Keep opt-outs and report warm neutrality or
executed-iteration variations rather than excluding inconvenient samples.
