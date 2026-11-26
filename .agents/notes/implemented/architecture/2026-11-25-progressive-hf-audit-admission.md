# Decision: Native admission for the plain progressive HF audit

The fixed-density CPU physical audit already combines the full native HF fleet
inventory and its dense observation buffers. For a plain RHF/UHF target, running
the front-end fleet planner first repeated the same admission and rebuilt basis
topology metadata. Use the selected native audit's capacity and execution gates
instead; do not size from an approximate matrix model or replace the Fock owner.

The independent audit still starts after both SCF owners have closed. Missing
protocols and unsupported topology require the incumbent preflight before any
compatibility audit. Targets with a preliminary-guess reservation also retain
that preflight: today's public budget includes that reservation even though the
audit itself never forms a new seed. A native refusal or shortage is terminal,
not permission to retry a different audit or to loosen the bound.
