# Port the measured one-electron angular/module specialization

Status: implemented; current-head numerical/CI results are tracked in the PR
Date: 2026-11-17

The RTX5090 cohort in `paper-hfdft-cold3-20261115` measured experimental
`4b8b103a57fbd4fd794e17dab405b6940b3adcea`, based on `17f9317a`. The annotated
`paper-hfdft-rtx5090-4b8b103a` tag preserves that original commit and ancestry.
This change ports its program delta to current master; it does not reacquire or
relabel the archived observations as current-master performance qualification.

The compiler emits bounded through-f values/derivatives alongside the complete
g programs from the existing scalar mathematics. Producers supply the maximum
physical shell angular momentum; unknown metadata defaults conservatively to g.
Expansion stride is not an angular bound. Separate CUDA translation units avoid
loading the unused high-angular module on the first low-angular consumer call.
Private shared launch headers retain one weighting/sign/traversal implementation.
No initialization moves outside the existing endpoint, and FP64, physical work,
SCF/precision/grid/screening controls and numerical gates are not changed.

The rebase retains current g-shell symmetry lowering and optional consumer-owned
primitive namespaces. Typed derivative bindings now refer to the selected owner;
family outputs and owner-scoped outputs are separate CLI modes. This preserves
the current COSX bounded artifacts rather than reverting their build contract.
The current upstream OpenBLAS LAPACKE declaration adapter already implements the
old experimental CPU fix. Its canonical `openblas_lapacke.hpp` owner remains;
the older duplicate `cpu/build_linked_lapacke.hpp` is not reintroduced.

Independent emitted-header tests cover f/g integral blocks against Libcint and
Cartesian/spherical derivative contractions against independent blocks. The
through-f value tolerances are atol2e-11/rtol3e-12, the g fallback uses
atol2e-9/rtol2e-11, and derivative blocks use atol3e-11/rtol6e-12. These are
numerical parity gates, not promises of bitwise-identical floating-point results.
Historical device tests belong to the measured tag; no new benchmark timings or
new source-matched NVIDIA endpoint qualification are claimed by this port.

Native identity hashes literal input bytes. The measured export has2131 EOL-only
differences from canonical Git blobs. With core.autocrlf=true, nine native files
must retain the LF spelling listed in the tag annotation to reproduce native
identity `e7335217cc4d94a4b450e1455d5164330441a02b1a6da98b822fe5c78e0726f9`.
The qualified clean tag checkout was checked against all2143 original native
inputs; its index tree remains the original tag tree. Ordinary platform-default
or all-LF checkouts have different byte identities. Do not override the identity
or normalize acquired evidence to hide those differences.
