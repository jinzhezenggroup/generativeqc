# Decision: saddle derivatives consume the retained native CPU batch

Status: implemented
Date: 2026-11-12

## Ownership and fixed chemistry

The molecular saddle adapter now requires canonical native B3LYP RKS/UKS and
retains the actual prepared batch for each energy/force/Hessian geometry. The
same native Hessian bridge used by the public Calculator and CLI supplies
coupled response and exactly-once native two-body D3(BJ). A missing prepared
native force capability refuses before SCF rather than using the incumbent
Python force adapter. Missing linked second-order/root-export closure also
refuses explicitly. The obsolete molecular JIT cache argument is removed and
repository callers migrate directly.

The optimizer is still the shared native eigenvector-following owner. Its
100-attempt cap, accepted/rejected iteration accounting, physical model guard,
independent convergence thresholds, final verification and unscaled harmonic
analysis do not change. Rejected trial roots never replace the accepted root.
Each occupied spin channel retains its own integer population and principal
cross-geometry overlap gate. Empty channels remain distinguishable from a
failed or changed population.

## Why a small orbital bridge remains

Root tracking needs occupied subspaces, not a Python electronic response owner.
A separate synchronous context-locked bridge admits the actual native shape
and numeric reservation before frame export/allocation, verifies the current
physical frame, and publishes only canonical C and scalar occupation counts.
It exports no AO values, Fock, density, quadrature or response operator.
Both query and copy authenticate expected atoms/coordinates/charge/spin and
the actual producer; staging and final validation prevent partial publication.
Descriptor/output overlaps and borrowed-source aliases refuse.

The export reservation covers the detached verified frame, staged C and caller
C. It is not a claim about producer verification/eigensolver scratch, Python
copies, process RSS or complete electronic work. Cross-geometry overlap has its
own bounded native consumer. Completed SCF, nuclear response, induced Fock,
integral primitive and D3 work remain separate; partially failed work is not
inferred. Batch owners close on publication and on all failure paths.

## Rejected alternatives and remaining domains

Do not instantiate CompositeRKSResponse/NativeAO merely to obtain root orbitals,
reuse a full-grid snapshot as a hidden response provider, relabel an electronic
Hessian as a corrected total, or add D3 again after native publication.

Compact batch/lifecycle controls and the independent light-molecule saddle
oracles are not installed-wheel/CLI, heavy Ni/I/Cs, or practical25--40atom
complete-electronic qualification. Those remain separate gates; ECP/DF/CUDA
molecular admission stays closed. The Python-free molecular TS CLI integration
also remains separate from the existing native corrector and Hessian CLI.

## Subsequent CLI integration

The separate molecular task and native harmonic-publication owner are now
implemented; see the [native CLI decision](2026-11-12-python-free-molecular-saddle-cli.md).
This supersedes the CLI implementation gap above, not the independent
heavy-element, positive open-shell or practical-endpoint qualification gates.
