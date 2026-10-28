# Native warm-seed capacity qualification

PR #2865 queries the prepared native owner before any warm-seed export.
Final production source is `115cc3c72`; baseline is `e8de3e4d9` (#2864).
This intermediate evidence commit retains the complete CPU audited endpoint
population. Full control receipts and reproduction instructions follow.

Each backend has 1,792 complete scalar/batch calls in eight ABBA/BAAB groups.
CPU maximum ratio is 1.008437472446882; CUDA is 1.0376111664979546, below the
unchanged 1.05 gate. All E/F, semantic-work, retained-seed and independent audit
count gates pass. Slurm 14843 passes 19 GPU tests with compatibility poisoned.
Slurm 14842 failed test collection because the uploaded benchmark helper was
missing; its installed SDK passed, and the failed receipt remains retained.

`cpu-samples.json.xz` retains the exact original CPU sample JSON. No scientific model, tolerance, seed policy or gate was changed.
This scoped #2227 R2/R4/R6 migration excludes R3 and does not close the parent.
