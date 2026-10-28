# Native precision-budget controls qualification

PR #2869 moves the incumbent Fast budget-Fock control selection to the shared
native precision policy. Model qualification remains with the selected Direct
or fitted owner. Native thresholds preserve explicit caller ceilings; only a
missing additive protocol permits compatibility. This is a scoped #2227 R4/R6
slice, excludes R3 and does not complete the parent.

Candidate production source is `15d8894c9`, compared with #2859 production
`7adf42cea`. The CPU matrix passes all 1,792 complete calls with identical
scientific work and exact energy/force parity. The maximum median time ratio is
1.0051806418477298, below the unchanged 1.05 gate. Calculator construction is
inside scalar and complete fleet timing, since this is where control selection
executes. Source-isolated workers retain cold, warm, moved and cleanup behavior.

`identities.json` binds sources, binary hashes and worker settings. Raw CPU
samples are retained here. Real-GPU qualification is still running; this CPU
receipt is not CUDA acceptance. Existing precision-admission raw recovery uses
the exact already-reviewed metadata from the sibling #2864 stack, preserving
the unchanged 80 MiB aggregate evidence budget.
