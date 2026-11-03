# H100 SM90 supporting qualification, frozen cd2b

Native source `cd2b4514be0a38c4920934481b1f02d157a694b4`, production core SHA256
`37b8bd02c1dddbf42385c494ae21ff51b50d874dae1dd0c01ab2b75fb0eb194e`.
These measurements do not qualify later source changes, including the restricted
Direct-force compiler optimization merged in #2935. They supplement the earlier
[build/DF44/screening packet](../h100-sm90-cd2b-20261102/README.md).

| Retained phase | CPU-replayed result | Boundary |
|---|---|---|
| Native qualification | Four executions, including two unfiltered all-process memchecks | Production and fixture core identities stay distinct; no retained full RSH finite-difference transcript |
| Nsight | Three actual SQLite traces, 12 complete E/F calls | Intrusive attribution; zero clean timing votes |
| Original Standard guards | 84 complete E/F, 7/8 populations qualify | All nine default UKS PBE DF energy results fail the unchanged dense-energy gate |
| Independent CPU attribution | 12 CPU E/F observations | Default 113-function auxiliary approximation explains the failure; not a native-force error |
| Explicit refined UKS PBE DF | Nine E/F calls pass unchanged dense and matched-DF gates | Existing 139-function cc-pVTZ-JKFIT input, separate qualification; original 84 remain 7/8 |
| Fresh dense references | Six known-source populations, 12 E/F calls | Same-level corroboration and dual-grid stability at original quarter-gates; no native or timing claim |

Run from the repository with Python 3.12+ and NumPy:

```sh
python benchmarks/h100_production_cohort.py
```

The receiver first replays the build/DF44/screening packet, then authenticates
all archives and recomputes these five supporting phase results. It uses temporary
source snapshots from local Git objects and makes no network requests or GPU/SCF
calls. A shallow clone must explicitly fetch native `cd2b4514b` and stock-reference
`15ab66a68a061355fda833a93e88484dbfee6360` before replay. Temporary native source
metadata points only to the clone's existing local Git object store.

The Nsight archive is split into two authenticated parts to meet the unchanged
1 MiB per-file limit. The reader reconstructs the original 1,440,220-byte archive,
checks its original SHA256 and member inventory, and independently reduces all
three retained SQLite databases. Large raw QDSTRM/nsys-rep originals stay remote;
the local reader does not inspect or reconstruct those remote originals.

All returned force vectors and original unfavorable results remain. Numerical
references are independent scientific implementations, not dataset ground truth.
The six fresh dense recipes use source `000517f82dc62130459a6fd92a27518a158c49cc`;
they corroborate only those six old targets. Producer bytes for the original 14
historical oracle recipes remain unrecovered. Installed-wheel/binary verification
is a remote execution attestation; local CPU replay checks its records, not the
remote binaries. Review is same-family/provisional, with actual model and effort
unverified.

Public ResourceBudget and broader facade qualification, final three-pair majority
performance, A100, RTX5090, universal MD and custom-RSH UKS are not established by
this packet. The fixed final 30-case denominator and >=20-win rule stay unchanged.

The [fresh delivery audit](audit/EXPERIMENT_AUDIT.md) independently recomputes
the CPU attribution and all five phase results. The public entry authenticates
the twelve CPU diagnostic observations and replays the matched-DF comparisons
used by the refined nine-call qualification. Its dormant budget helper files
add no resource-budget acceptance. The audit input seal includes private control
fixtures; the published audit metadata describes that boundary.

The two Nsight analysis archive parts are retained byte-for-byte in the existing
merged Git objects at `4c7354958468810dc83eb409c493ae49d810e613`.
`historical-snapshot.json` authenticates both SHA256 and Git object identities.
The replay command reads only these explicit entries through the offline history
loader; no network, Release or external backup is used. Full Git history is
required. All three SQLite databases and twelve intrusive E/F remain available;
this storage change adds no timing votes and changes no scientific proof.
