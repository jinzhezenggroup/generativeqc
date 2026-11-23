# Decision: Compose explicit CLI cold starts into host admission

Status: implemented
Date: 2026-11-23

## Decision

Use the existing native cold-start parser, target gate, local numeric cap,
and source inventories for `resources` and `run --host-bytes`. The CLI
admits explicit Direct FP64 CPU RHF/RKS contexts and leaves the SDK's
prepared solver in charge of the actual outcome. RKS computes stationary
forces after its energy solve, so its force request does not imply the
RHF solver's force-mode restriction on HF and coarse-LDA initial guesses.

The inventory reserves all possible seed matrices and one serialized
source workspace on top of the conservative target envelope. HF/LDA use
the larger of their native local numeric bound and the existing source
layout, rather than assuming a configurable numeric cap covers every
source allocation. An explicitly short local cap or the known native LDA
source-domain decline needs no seed or source reservation. A short global
host budget is an admission failure, not authorization to change the caller's
guess. The dry run normalizes metadata only; a budgeted `run` borrows the
same system handle it will prepare, rather than normalizing a second copy.

This first cutover does not change Python's legacy resource composition,
which is more conservative in some cold-start lifetimes and can still
reserve a source even when a local cap would skip it. It does not broaden
CUDA/DF, unrestricted, automatic or second-order CLI planning.

The later [Python cold-start cutover](2026-11-23-python-preliminary-host-admission.md)
uses this layout for explicit Direct FP64 CPU RHF/RKS plans.

## Evidence

On base `cae2959e1`, 152 CPU CTests, 193 selected installed Python tests,
and strict Sphinx passed. Native CLI tests verify an admitted fleet charges
three seeds but one source workspace, and one byte below the composed host
budget fails before preparation. At the local cap and one byte below, all
four H2/PBE policies match the installed SDK and Python in numerical cap,
outcome, preliminary Fock work, energy and force. The SDK and CLI ran with
Python and a toolchain off the runtime search path. The RHF cases also check
the independent STO-3G H2 energy in the existing CLI contract.

Eight interleaved measurements per variant after warmup on one EPYC 7K62
core include process startup, preparation, SCF, force, and publication.
PBE0/def2-SVP water medians were 0.284–0.333 s across all four cold starts.
Ordinary unbudgeted execution changed by −0.72% to −0.19% relative to the
prior install; adding exact-cap admission to the new install cost 0.38–0.58%
on water. Tiny RHF/H2 had a 1.15 ms (13.3%) opt-in overhead. Matched energy,
forces, target iterations, preliminary Fock builds and traced work counts
were unchanged: HF/LDA/MINAO/Hückel used respectively 11/17/0/7 preliminary
Focks and 12/12/13/13 target RKS physical Fock evaluations. Raw runs,
receipts and compiler-cache statistics are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-preliminary-host`.

## References

- [CLI host force admission](2026-11-23-cli-cpu-ks-force-host-admission.md)
- [Python dense force budget](2026-11-23-python-dense-eri-force-plans.md)
