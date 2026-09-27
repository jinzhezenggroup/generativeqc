# Decision: generate Direct source contraction at build time

Status: implemented
Date: 2026-09-27

## Problem
The qualified Direct-HF source-contraction helper remained a 292-line handwritten
scientific CUDA owner shared by Fock, force fallback and Schwarz consumers even
though its surrounding production ownership had moved toward compiler-generated
integral code.

## Decision
Emit the existing source-contraction implementation from the scientific compiler
at build time and delete the source-tree CUDA owner. Consumers include the
generated header directly. The arithmetic, reduction order, shell dispatch,
order-two compact specialization and native queue/screening policy are unchanged.

## Rejected alternatives
Rewriting the contraction around a new Direct-only IR was rejected because the
existing IntegralIR/AOT stack already owns the long-term mathematical direction.
Deleting the fallback outright was also rejected because Fock, force and Schwarz
still rely on its qualified unsupported-domain coverage.

## Invariants
- No selector, screening threshold, queue topology or schedule changes.
- Generated execution preserves the previous arithmetic and reduction order.
- Remaining native recurrence dependencies stay explicit until separately retired.
- CUDA compile and endpoint qualification remain merge gates.

## Evidence
Structural regression tests require all three production consumers to include the
generated header and require the former native owner to be absent.

## Revisit when
Retire the generated fallback itself when the generated shell-class registry covers
all supported production domains with matched numerical/resource/endpoint evidence.

## References
- #356
- #1464

Agent: ChatGPT
Model: GPT-5.6 Sol
