# Decision: complete named KS programs belong to the native prepared owner

Status: implemented
Date: 2026-10-25

## Problem

The generated native method catalog described only electronic prefixes. CLI
callers reconstructed those prefixes, while Python retained separate D3/D4
correction owners and combined energy and gradients after native KS execution.
This prevented standalone C/C++ clients from using common complete compositions
and made the public method identity broader than the native execution owner.

## Decision

Add named single-system and batch preparation entry points behind the C API.
Their catalog is generated from compiler MethodIR and its existing electronic
lowering gates. It includes exact semilocal coefficients, range-exchange terms,
nonlocal parameters and supported geometry-only dispersion parameters. There is
no additional handwritten functional parameter registry. CLI and Python reuse
these native entry points; Python selects a complete graph by exact identity.

Keep correction plans inside `KsPreparedState`, sharing the established KS
scheduler, geometry rebuild, force admission and result transaction. Corrections
run only after electronic convergence. Energy-only replay does not compute
correction gradients. A complete force subtracts the correction gradient once.
Electronic final-state tokens continue to identify the SCF Hamiltonian; they do
not incorporate geometry-only energy. Separate correction receipts belong to the
completed result and are revoked at every attempted replay, including invalid
input. Correction diagnostics use an optional batch capability facet.

The prefix-only catalog query refuses corrected methods rather than presenting
an incomplete graph as the selected program. Canonical WB97M-V is admitted only
when its semilocal, range-exchange and nonlocal graph matches the compiler's
canonical graph. Basis/gCP-bound r²SCAN-3c remains outside this entry point.

## Compatibility and resources

Existing ABI layouts remain unchanged. The additive named descriptor owns no
buffers; preparation copies controls and system data. Known method/KS precision
and nonlocal-grid extensions are preserved, while unknown extension sizes fail
closed. The named selector supplies scientific fields; grid, scheduling and byte
budgets remain explicit caller controls, including zero-budget refusals.

The KS scheduler currently retains one correction owner per system. Divide the
batch dispersion budget equally among those owners, accounting for duplicated
tables as well as workspaces. This is conservative and can refuse a heterogeneous
fleet that a future shared ragged owner could admit. Diagnostic counters describe
the current owners; geometry rebuilds retire prior counters.

Python retains its established explicit composition for custom/unrepresented
graphs, older libraries, non-FP64 requests and global `ResourcePlan` requests.
The resource-plan exception is intentional: its D3 inventory models a single
ragged correction fleet and cannot silently account for these per-system owners.
Native admission failures never trigger a retry through compatibility execution.
Electronic force contexts still requiring Python compatibility consumers keep a
separate correction-gradient consumer; they do not add correction energy twice.
The native CLI continues to reject unqualified complete forces. The legacy
PBE-D4 ABI selector keeps its existing compatibility behavior.

## Rejected alternatives

- Compose corrected energies in CLI or Python wrappers: this duplicates the
  scientific endpoint and leaves C callers without complete behavior.
- Wrap a prepared KS owner in an unrelated generic correction owner: existing
  electronic snapshot/gradient ports and geometry lifetime would need another
  forwarding hierarchy. The correction belongs to the same prepared KS state.
- Advertise a method as supported when only its electronic prefix lowers:
  dropping VV10, dispersion or gCP changes the model while preserving its name.
- Reuse the old global resource inventory unchanged: memory-bounded individual
  owners do not prove the old fleet inventory bounds their aggregate storage.

## Evidence

Qualification used a CPU-only Release build with verified sccache 0.16.0. Native
catalog tests cover metadata, exact identity, refusal publication, supported
extension validation and prepared-input lifetime. Python composition tests forbid
standalone Python correction constructors on native paths, exercise energy-only
requests, failed-replay receipt invalidation/recovery and capacity refusal without
fallback. Existing D3/D4 independent reference tests are retained.
Triatomic endpoint comparisons against the separately qualified standalone
providers also require a nonzero ATM contribution, which a diatomic cannot test.

CLI execution tests run without Python in `PATH`. WB97M-V RKS and UKS energies are
compared with independent PySCF 2.14.0 / Libxc 7 results on H2/H2+, STO-3G, an
8×4×8 product grid, original Becke partition without radii adjustment/pruning,
and identical XC/nonlocal grids. The gate is 2e-8 Hartree. Total D3(BJ),
D3(BJ)-ATM, D3(0), PBE0-D4 and B3LYP-D4 forces are checked against reconverged
energy differences at 3e-4 and 1e-4 Bohr steps with a 2e-5 Hartree/Bohr gate.
These small-grid checks establish endpoint assembly, not quadrature convergence.

No CUDA execution qualification or performance improvement is claimed. Slurm
reported only the drained maintenance node `node3`; no real-GPU run was attempted.
The generated CUDA flags retain existing component/backend admission and still
require complete endpoint qualification on an available permitted GPU node.

## Revisit when

- A shared correction fleet can follow KS item rebuilds while preserving bounded
  aggregate storage and existing global resource-plan accounting.
- Native electronic force domains expand enough to retire compatibility consumers.
- A complete basis/gCP binding contract admits r²SCAN-3c without dropping terms.
- Independent CUDA endpoint qualification is available for the new compositions.

## References

- [Native C++ API](../../../../docs/user/native_cpp.md)
- [Native CLI](../../../../docs/user/native_cli.md)
- [Runtime ownership](../../../../docs/developer/architecture.md)
- `tests/native/test_named_ks_catalog.cpp`
- `tests/native/test_cli_dft_df.py`
- `tests/python/test_native_ks_catalog.py`
