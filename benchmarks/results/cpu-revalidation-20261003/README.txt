Fresh CPU endpoint results, measured locally 2026-10-02
Baseline: 9a5871dca1f371e5f6e9fcf9d6b076c83b894c14
Candidate: a7c118aa3076dbfdc13e47ed2d1056c3b0fe6e80
Later publication/integration commits were not timed; they must not replace these measured identities.

Clean six-pair cohort, baseline -> candidate warm median seconds (descriptive reduction):
HF24: 0.033648905 -> 0.024533925 s (27.088%)
HF96: 12.172592472 -> 7.221002969 s (40.678%)
PBE24: 3.343463445 -> 3.241654162 s (3.045%)
PBE0-24: 0.379665240 -> 0.358991805 s (5.445%)
LDA7: 0.376747534 -> 0.380077768 s (-0.884%) (losing result: slower)
R2SCAN7: 2.407986312 -> 1.899525932 s (21.116%)
HF96 process-first: 12.517738929 -> 7.340049768 s
HF96 changed geometry: 11.860249011 -> 7.200266012 s
HF96 median cumulative process peak RSS: 4623886 -> 1498946 KiB; not per-phase allocator memory.
All 288 clean endpoint energies pass the independent 1e-8 Eh gate (maximum error 3.9221959014e-12 Eh), with bitwise-identical baseline/candidate energies and matching exposed iterations/basis work. DFT Fock counts match; HF Fock counts/full timed histories are unavailable. Warm runs use fresh Calculator ownership without density reuse. These are descriptive timings, not a formal performance/default-promotion pass.
PBE96: both versions fail the same final energy-change gate, 1.3073986338e-12 > 1e-12 Eh, after 25 iterations plus two final Fock builds. Max150 was not exhausted; no PBE96 speed claim. HF96 pilot timing is invalid due to overlapping local archive work. All failed/invalid/losing scalar records remain retained.

Storage and reproduction
Deterministic gzip contains scientific JSON only, not logs/profiler archives/matrices. Original hashes and expanded-payload identity preserve scalar values and order. Historical source deltas must match pre-existing receipt SHA-256.
python decode_cpu_evidence.py verify
python decode_cpu_evidence.py expand --output NEW_DIRECTORY
Expansion writes readable JSON and exact source snapshots without executing them. Copy expanded qualification-harness/*.py.txt to qualification-harness/*.py, prepare pinned builds from retained toolchain/cache inputs, and record fresh actual build identities. Run publication.json's balanced command in an exclusive CPU0 lane and new output directory. Never relabel a new binary as measured a7c118aa. Oracles retain exact primitives/grid/geometry/thresholds.

Source recovery: local measured candidate a7c118aa is available through remote commit b039eea4354b55fa7bcd0d274f14dbad6a11a10c. Its complete Git tree is exactly 0680717ea6cda4e6abb2d3c478defba37a02b755; only commit metadata differs. Use that remote commit for the candidate checkout, and retain a7c118aa as the original measurement identifier. This alias was not separately timed.
