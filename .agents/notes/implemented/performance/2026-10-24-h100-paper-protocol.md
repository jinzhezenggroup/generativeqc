# Decision: bind the minimal H100 paper protocol before execution

Status: implemented (offline contract only)
Date: 2026-10-24

The paper's RTX 5090 endpoints need a bounded second-device study. Replaying
current master would mix device effects with later WB implementation changes;
replaying the evidence archive would confuse archive and measured identities.

The task-owned `h100-paper-protocol-20261024` reader reconstructs HF/DFT's
provenance patch against its compiled base and compares the changed LF files
with the measured commit. CBS applies its distinct qualified patch and checks
all retained candidate file hashes. This opt-in historical authentication is
not a permanent production-source spelling test. The missing original full
HF source-manifest remains UNAVAILABLE.

Eleven input cells have 90 independent processes and 306 complete calls.
Warm calls stay nested under their own process/cold density. Cost is derived
from actual pilots with finite caps; failed attempts remain in the ledger.
Explicit exceptions retain all reader admission checks under optimized Python.
Receipt-contract completeness is deliberately separate from independent raw
scientific acceptance and device qualification.

Rejected: current-master substitution, sm120 binary reuse, interpolated H100
timing/cost, reconstructed C2 displaced energies from projections, pooling warm
calls into independent repeats, and inventing qz Slurm metadata. A qz transport
adapter records its own hash alongside the frozen original source driver.
Production compatibility changes require a separate paired-version cohort.

Evidence: family `test_protocol.py` behavioral tests and opt-in `protocol.py
verify`; original archive `7522994e`, measured HF `cbd4d807`, CBS base
`976f756b` plus patch `5a1a974b`, independent energy `334e7f48` and directions
`59434367`. Refs #2725, #2724. No GPU/scientific calculation is performed here.
