# Decision: Make packed CPU ERI values available to the native force CLI

Status: implemented
Date: 2026-11-22

The later [Python force plan](2026-11-23-python-packed-eri-force-plans.md)
adds an independently admitted resource contract for that frontend.

## Decision

Route the opt-in `run --cpu-packed-eri-budget-bytes` and the installed C++
`prepare_named_ks` helper through the existing named KS descriptor, rather than
adding a force-specific integral representation. Only the scalar SCF J/K value
source changes: analytic forces still belong to the existing prepared,
token-bound derivative owner. Keep that owner's capability predicate and the
ordinary default provider unchanged. The exact spherical s/p/d/f restriction
and value-array size gate stay in native preparation; an explicit request never
silently retries with dense values. Reject incompatible providers, including
DF, CUDA, range-separated and nonlocal programs. This is not a lease for force
scratch, and does not admit packed Python forces under a `ResourceBudget`.

## Evidence

On the checkout based on `cae2959e1`, four installed C++ SDK / native CLI /
Python cases agreed with independent PySCF for spherical PBE0 and B3LYP RKS
and UKS. The SDK and CLI ran with an empty `PATH`. The corrected B3LYP UKS
CLI test observes exactly 48 retained bytes at an exact 48-byte cap; the one-byte
miss, incompatible providers and unconverged SCF publish no force result.
The broader validation passed 31 CPU CTests, 115 installed/force/resource
pytest cases and 63 API and source-identity cases. The remaining 208 cases
in the installed/force/resource run were skipped by their capability gates.
On non-axis-aligned CsH/def2-SVP with scalar ECP and B3LYP-D3(BJ), the analytic
H z force agreed with centered differences of the complete packed energy to
`1.3e-9` and `3.6e-10` Hartree/Bohr at `2e-4` and `1e-4` Bohr, respectively.

Nine interleaved process samples per variant used the same shared library,
one AMD EPYC 7K62 core and one OpenBLAS/OpenMP thread. Timing spans process
startup to complete energy and analytic force output. Work is unchanged in
each dense/packed comparison, as measured by SCF iterations, terminal Fock
builds and grid points; force derivatives use the same owner and providers.

| Endpoint | Iterations / Fock builds / grid points | Dense → packed value bytes | Median before dense / after dense / after packed |
| --- | ---: | ---: | ---: |
| Water PBE0 RKS | 10 / 12 / 2,326 | 19,208 → 3,248 | 55.86 / 55.73 / 55.74 ms |
| HeH B3LYP-D3(BJ) UKS | 6 / 8 / 1,264 | 128 → 48 | 31.24 / 31.26 / 31.23 ms |
| CsH B3LYP-D3(BJ) RKS + ECP | 14 / 16 / 33,516 | 5,658,248 → 758,640 | 3.339 / 3.333 / 3.328 s |

The largest observed default slowdown was 0.09%; the timings do not show a
material packed speedup. Between dense and packed, the largest absolute force
difference was `3.1e-14` Hartree/Bohr. The byte reductions concern only retained
ERI values, not full endpoint peak memory. Commands, per-run timings, native
traces, checksums and the finite-difference receipt are retained under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-packed-force-bench`.

## References

- [Packed Hessian host inventory](2026-11-22-cli-packed-eri-host-inventory.md)
- [Packed Python host inventory](2026-11-22-python-packed-eri-host-inventory.md)
