# Decision: reject unqualified unrestricted orbital-g consumers

Status: implemented
Date: 2026-11-10

The shared generic full-range DF source supports orbital g for the qualified
restricted full-DF closure. Source capability alone does not qualify an
unrestricted Fock/SCF consumer. Direct C API Fock-plan creation could bypass the
Python method preflight, and resident UHF also has single and bucket entry paths.

A shared host spin-domain check now rejects unrestricted orbital l > 3 before
CUDA one-electron or DF preparation in those consumers. The bucket checks every
member before preparation or cache mutation. Restricted g and unrestricted f
admission are preserved, with source value-math/range restrictions independent.

Do not close this gap by lowering the shared source bound, which would remove
qualified restricted support, or by widening the provider capability declaration,
which would falsely claim unrestricted qualification. Revisit only after an
independent complete unrestricted endpoint qualification establishes that scope.

The executable native C API regression fails with the original constructor and
passes after the gate. Host-only validation does not establish numerical
corruption in the old route or qualify new GPU/performance behavior. The existing
full-DF qualification evidence remains historical and unchanged.
