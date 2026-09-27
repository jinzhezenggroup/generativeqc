# Decision: reuse common AO-to-MO transform prefixes inside CUDA post-HF batches

Status: implemented
Date: 2026-09-27

## Problem

After #1411, #1415, #1419 and #1420, a CUDA batch can share one AO source scan,
one raw-tile upload and one CUDA owner, but every requested MO block still
repeats all four staged GEMMs. RCCSD's seven canonical blocks therefore execute
28 GEMMs per AO source tile even though many requests share occupied/virtual
prefixes.

## Decision

Add a compiler-owned ordered prefix-reuse plan over abstract per-axis keys.
The native provider maps exact MO slot vectors to those keys and passes the
resulting prefix leaders into the generic CUDA batch owner.

The CUDA runtime executes the prefix trie one depth at a time and ping-pongs the
existing per-request scratch buffers. No new numeric arena is required. A child
reads the first request that owns its exact prefix; duplicate leaves fan out
through the existing DAXPY publication accumulators.

For the seven RCCSD blocks the unique prefix counts are 2, 3, 5 and 7 across the
four transform depths, reducing staged GEMMs from 28 to 17 per AO source tile.
The same mechanism applies to any ordered NativeBlockProvider batch with shared
MO-slot prefixes; there is no RCCSD-only runtime branch.

Provider transform-FMA accounting follows the actual CUDA prefix schedule while
the CPU path remains unchanged.

No wall-time speedup is claimed until the allocated RTX 5090 endpoint is rerun.

References: #1401; #1411; #1415; #1419; #1420.

Agent: ChatGPT
Model: GPT-5.6 Sol
