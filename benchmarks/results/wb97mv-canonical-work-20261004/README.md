# Twelve-atom TZVPD canonical value work census

This records one full J/K and one LR K resident source call on n1 RTX 5090,
Slurm 5748, against frozen #1842 source `591a53951`. It is diagnostic work
evidence, with no molecular endpoint timing or speedup claim. The original
geometry and full def2-TZVPD snapshot match the accepted twelve-atom protocol;
the synthetic identity density does not change geometry-only value screening.

Both actual device counters equal the independent host enumeration of resident
screened rows: 407,065,289 admitted AO quartets per operator. For orders 5--8,
487,996,523 primitive-loop products reduce to 2,921,528 when counting common
preparation once per physical shell quartet, or 355,424,818 within the current
32-lane cohorts. These bounds do not remove component contractions, charge a
new schedule, establish FLOPs, or predict speedup. Force counts remain unknown.

`evidence.json` retains exact text bytes and SHA-256/lengths for the C++ probe,
input, build/run scripts, ccache and source/binary receipts, successful run,
and failed diagnostic attempts. Binaries are identified by digest, not checked
in. To reproduce, extract the `probe/*` text members, adjust the recorded
checkout/toolchain paths, compile with verified ccache against the frozen
headers/library, and run through finite GPU Slurm. Preserve assigned visibility.
The frozen library is essential; this internal diagnostic consumes its plan ABI.

Run the offline integrity and arithmetic checks without CUDA:

```bash
python benchmarks/results/wb97mv-canonical-work-20261004/verify.py
```

The verifier authenticates retained bytes and checks census/aggregate consistency;
it does not rerun device work or independently qualify J/K numerical accuracy.
The [Agent Note](../../../.agents/notes/proposed/2026-10-04-canonical-shell-component-reuse.md)
records formulas, memory limits, correctness risks and required endpoint gates.
