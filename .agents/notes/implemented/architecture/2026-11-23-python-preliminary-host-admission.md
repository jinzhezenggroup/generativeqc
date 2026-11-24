# Decision: Share the CLI's explicit CPU cold-start layout with Python

Status: implemented
Date: 2026-11-23

## Decision

Explicit Direct FP64 RHF/RKS resource plans use the same per-item native
layout as the CPU CLI and the same numeric cap as the prepared solver. A
compact query accepts the target's already resolved AO topology; only Hückel
also needs shell-to-atom rows. This avoids constructing a context or target
system just to estimate resources and avoids rebuilding a second Python HF or
KS request. The input excludes coordinates and basis values. The output is
transactional and the Python consumer validates its disposition before
composing it with the target.

Every admitted item gets a seed allowance and source scratch is serialized.
The aggregate host limit never changes the explicit preliminary policy.
A local budget skip or native coarse-LDA source decline contributes no seed
or source workspace for that item. Missing symbols preserve old-library
estimation; refusal from a discovered symbol does not fall back. Automatic,
fitted and CUDA policies retain their separate existing contracts.

## Evidence

On base `cae2959e1`, 152 CPU CTests, 226 installed Python tests, and strict
Sphinx passed. Compact and normalized native queries match all four layouts
and preserve the previously published result on invalid inputs. CLI and
Python match exact host cutoffs for all four H2/PBE force cases, including one
byte below the separate local cap. A ragged He/water plan leaves the skipped
item uncharged and replays moved coordinates. Installed SDK E/F outputs and
work receipts match the previous install at the exact local caps and just
below, with Python and the toolchain off its runtime search path.

Seven interleaved process pairs per fixture after warmup on one EPYC 7K62 core
include construction, plan estimation, preparation, cold/warm/moved force
replays and close. Energies, forces, diagnostic outcomes, SCF iterations, Fock
builds and warm-start selection were identical. Water/PBE0/def2-SVP medians
changed by −1.34% to −0.20%; the ragged Hückel case changed by +0.71%. The
ragged modeled host peak fell from 84,038,968 to 67,256,664 bytes because both
items' local caps skipped preparation. HF/LDA water gained 4,608 modeled bytes
for the retained seed; this is not an increase in observed process memory.
The cold/warm/moved target Fock counts were 12/4/12 for HF/LDA and 13/4/12 for
MINAO/Hückel; cold preliminary Fock counts were 11/17/0/7 respectively.

Raw receipts and compiler-cache statistics are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/python-preliminary-host`.

## References

- [CLI host cold-start admission](2026-11-23-cli-preliminary-host-admission.md)
