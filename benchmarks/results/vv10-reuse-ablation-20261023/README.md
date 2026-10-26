# Same-binary VV10 reuse ablation

**Selected numerical PASS; original four-owner protocol FAILED / INCOMPLETE.**
#2677 OPEN; no default/performance promotion.
Details: [Agent Note](../../../.agents/notes/proposed/2026-10-23-vv10-same-binary-ablation.md).

WB97M-V/def2-SVP water48/96, implicit grids, complete moving-grid E+F.
Standard/Fast x Direct/DF: default/SCF-off/force-off/both-off, each cold+3 warm
from its own cold density. All Fast candidate/audit/fallback work survives beyond
public iteration/Fock counts. Frozen source4a783a27e; identities in
[publication](publication.json)/raw manifests.

Formal128 = old80 on H100 qb-prod-gpu046/PCI2A + authorized new48 on
qb-prod-gpu095/PCI18, zero retries; pilot64 stays separate. Physical GPU changed. The [addendum](protocol-addendum-single-owner.md) replaces
the entire partial DF96 cohort; three old colds are excluded. Do not pool timing.
Independent PySCF/GPU4PySCF producers/H-O tables and CPU-GPU/grid-stability
quarter gates at3/12/48/96 atoms survive; missing-producer arrays are not oracles.

| Error | Original absolute gate | Worst selected |
| --- | ---: | ---: |
| Energy/atom, Eh | 1e-5 | 3.0313102807e-6 |
| Maximum force, Eh/Bohr | 1e-4 | 2.1186440036e-5 |
| Per-sample RMS force, Eh/Bohr | 3e-5 | 1.0775033025e-5 |

[Timings](timings.md) cover cold construction and synchronized warm public E+F;
[work](work.md) retains histories/consumers/operator scopes. Opaque J/K, isolated
compile time and measured endpoint peak memory remain UNAVAILABLE. Device total
is not peak; consumer sampling is not reservation. First-use/resource plans and
shared SSD/cache overlap prevent cache-only attribution or pooled speedup.

## Offline verification

Linux Python3.11.16/NumPy2.4.6; data only:

```sh
cd benchmarks/results/vv10-reuse-ablation-20261023
python reproduce_recovery.py --output ../../../.artifacts/vv10-replay
python -m unittest discover -p 'test_*.py'
```

Restores715 collection paths (134 whole-file aliases), plus16 retirement records: SHA/bytes/POSIX modes, source/SDK/oracle/
claim/admission/return bindings and retired Job receipts;
frozen replay regenerates reports. Helpers restore from `offline-source.json.xz`.
Numerical, runtime, producer and failure records are unchanged. XZ caps:96MiB decoder/64MiB expansion; reject
truncation/extra streams. Verify Git-LF blobs or a `core.autocrlf=false` archive.

Failures retained: Windows3.13.14 /
NumPy2.3.2 strict RMS differs in32 last bits (max5.083e-21), without changed
values/gates; original DF96 hit10800s, keeping failed claim/83 returns (80 selected)
and no restorable owner/device checkpoint; intrusive both-off diagnostic rc1 lost
E/F/outer timing after a list/tuple postcheck, keeping journal/callback work.
Missing observables remain UNAVAILABLE; later source integrity cannot complete
the failed protocol or establish its cause. Broader DFT is unproven.

## Source identity

`source-identity.json` replaces only the derived 12,222-entry full-repository
source listing. It binds the frozen commit and tree, the original listing's
byte count and SHA-256, and the solver identity consumed by the oracle checks.
The revision/tree pair was verified against the public Git revision. Replay
authenticates this recorded identity through publication/collection checksums
and checks the solver SHA against retained oracle receipts; it does not resolve
the Git tree or provide a self-contained Git-object proof.
Build, runtime, driver, numerical and failed-protocol receipts remain unchanged.
The original full listing is no longer restored by this replay. Its exact
3,192,615 bytes were independently reconstructed from all frozen Git blobs
(JSON indent 2, Git path order, CRLF and final newline), matching its recorded
SHA-256. This removes repeated source metadata, not scientific observations.
