# Decision: Share modeled CPU KS force admission between CLI planning and execution

Status: implemented
Date: 2026-11-23

## Decision

`resources --forces` composes the same native CPU KS fleet inventory that
`run --forces --host-bytes` checks before preparation. The latter reuses the
already expanded basis. Retain either the dense or packed ERI value owner
for each item, charge maximum serialized force scratch, and add the native
named correction capacities when present. The packed-ERI limit still caps
only the packed values; it is not interchangeable with the host limit.

An inventory is not a prepared property advertisement. The native prepared
owner still admits or rejects the actual force context, and an unbounded
run does not compute a host inventory. CUDA, DF, unqualified graphs and
nonordinary guesses do not acquire force-budget support here. The reported
host peak is not a process-RSS or recurrence-temporary bound. No force
equations, provider selection, or SCF stopping rules change.

## Evidence

On base `cae2959e1`, the CLI runtime test checks that the exact effective host
limit succeeds and one byte less fails before an SCF trace, including with
reserve applied. The ordinary and budgeted results match. It exercises
dense PBE0 and corrected PBE/PBE0, and corrected packed B3LYP UKS. The packed
fleet test also verifies exact aggregate value bytes and a one-byte packed
shortage. Four installed SDK/CLI/Python force cases matched the independent
PySCF oracle, with the CLI running at the exact modeled peak.

Seven interleaved complete-process timings per variant used the same CPU-only
native library, with Python and the compiler absent from the CLI's search
path. Energies, forces, iterations, grid points, terminal Fock builds and
packed ERI work receipts match across variants. Times are medians in seconds;
the host peak is the modeled capacity, not a measured allocation peak.

| Case | Before | Ordinary | Exact cap | Host peak (B) | Grid points | Iterations / Fock builds | Packed bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H₂O / PBE0 RKS | 0.05667 | 0.05643 | 0.05785 | 67,271,656 | 2,326 | 10 / 12 | — |
| HeH / B3LYP-D3(BJ) UKS | 0.03131 | 0.03112 | 0.03240 | 67,197,230 | 1,264 | 6 / 8 | 48 |
| CsH / B3LYP-D3(BJ) RKS, scalar ECP | 3.23802 | 3.24042 | 3.24216 | 121,456,470 | 33,516 | 14 / 16 | 758,640 |

Ordinary timings changed by −0.43%, −0.60%, and +0.07%, respectively. Running
at the exact cap added 2.52%, 4.13%, and 0.05% relative to ordinary execution.
Raw commands, traces, timings and binary hashes are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-force-host-bench`. This is scoped CPU
evidence; no real-device qualification is implied.

## References

- [Packed CPU ERI force plans](2026-11-23-python-packed-eri-force-plans.md)
- [Separate CPU Hessian host admission](2026-11-22-cli-cpu-hessian-host-admission.md)
