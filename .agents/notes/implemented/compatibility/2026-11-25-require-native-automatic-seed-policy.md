# Decision: Require native automatic seed policy

Status: implemented
Date: 2026-11-25

Python, the CLI and SDK already use the native combined query to choose the
default initial-guess provider. Retire Python's fallback that recombined older
provider bits and duplicated element, ECP and method selection. A missing
policy symbol is an unsupported ABI, not an Hcore or MINAO decision. An
explicit Hcore choice and explicit preliminary providers bypass automatic
selection as before.

Keep the native per-item CPU transport, the native CUDA eligibility check for
each member of the shared batch descriptor, and Hcore fallback on an actual
negative native eligibility result. Allocation and preparation fallback
remain separate from ABI selection. The tests exercise missing symbols,
invalid decisions, explicit choices and mixed native execution.

Refs #2227.
