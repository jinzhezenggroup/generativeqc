# Decision: retain larger CPU capacity results without hiding mixed timing

Status: implemented evidence retention; no production algorithm change
Date: 2026-10-03

## Problem

The shell-local spherical ERI projection needed qualification beyond the earlier
12-atom/96-AO workload. Materialized four-index storage also makes careless
larger comparisons an OOM risk on the measured 9.73-GiB/no-swap host.

## Decision

Keep the direct RHF/def2-SVP method and exact primitive inputs fixed while
increasing both atoms and AOs. Admit 17-atom/130-AO pentane in both arms and
20-atom/154-AO hexane in the candidate only. Preserve the statically blocked
hexane baseline and both 23-atom/178-AO heptane arms as capacity boundaries,
without inventing measured failures or speed ratios.

All 44 executed scalar endpoints pass unchanged numerical gates. The balanced
pentane cohort shows 53.10% lower median process RSS. Its pooled warm median is
nearly tied, with three matched losses; different paired summary statistics do
not justify selectively replacing that result. Iteration counts differ and
complete Fock/history/native-state telemetry is absent.

## Oracle and storage boundaries

The four initial strict PySCF incremental-direct attempts failed at 150 cycles.
A separately retained full-density J/K rebuild protocol, with identical inputs,
equations, tolerances, screening and maximum iterations, passes all four states.
The failures are not dropped or reclassified. Independent saved-array audits
support the strict oracle states and all native scalar comparisons, not native
full-array or large-force equivalence.

Retain compact scalar records, exact scientific protocol sources/inputs and
checksums using standard XZ plus a bounded decoder. Full raw arrays/logs/binaries
remain local; a checksum is not recovery. The frozen driver binds absolute paths
and binary identities, so a new build requires a new, explicitly adapted freeze.
This avoids placing large matrices or duplicated expanded text in Git while
preserving the actual scientific procedure and every loss/failure category.

## Evidence and consequences

See `benchmarks/results/cpu-larger-systems-20261003/README.md` and the decoded
independent report. Measured source identities remain e47058a5 and the exact-tree
1378cdbc alias of local 4796d967. Later merged ancestry is not relabeled as a new
measurement. The minor permitted metadata-read exception and hidden cgroup
visibility limit remain explicit. Formal performance/default promotion,
full native state/work parity and repository-wide resource proofs remain open.

Revisit warm throughput with complete semantic work counters and new balanced
endpoints, or larger capacity with a separately qualified storage algorithm.
Do not silently change to density fitting or loosen acceptance gates to make
an otherwise blocked comparison fit.
