# Decision: full automatic Libxc CUDA catalog in release artifacts

Status: proposed
Date: 2026-11-01

## Problem

The source-derived Libxc semilocal catalog was structurally default-allow, but
only the SM120 native CLI release preset explicitly enabled full automatic
CUDA XC AOT. The manylinux wheel did not request automatic XC AOT. A user
could therefore see a selectable Libxc method but encounter a missing static
CUDA provider in the installed binary, depending on packaging rather than
the functional's ingredients.

## Policy

- All CUDA **release presets** (portable, SM120, native CLI) enable automatic
  XC AOT and leave the registration selection empty, thereby packaging the
  entire structural inventory from the compiler. This is not a frozen
  per-functional allowlist.
- The distributable manylinux wheel requests automatic AOT through the
  supported scikit-build-core CMake-define environment mechanism, independent
  of its selected CUDA SM target. The normal development preset remains
  lightweight and AOT opt-in.
- A generic `cuda-release-cli` preset inherits the portable release policy.
  The SM120 native CLI is an optimized specialization of that same policy,
  not a separate catalog owner.
- No change is made to Libxc source equations, runtime SCF/force domain
  admission, derivative qualification, CUDA execution semantics, or the
  existing architecture matrix. Absent/unsupported runtime providers remain
  fail-closed; compilation is not a claim of universal molecular validation.

## Verification and integration

The added host policy tests resolve the actual preset inheritance and inspect
the wheel build's CMake-defines. Full CUDA wheel build/link/payload checks and
architecture-specific smoke remain required in CI. PR #2832 owns the
SM80/SM90/SM120 wheel matrix and host-link reach; this change must preserve
its results when integrated. Do not interpret host-only config tests as a
successful three-target CUDA wheel or a new numerical qualification.

Coding-Agent: ChatGPT
Model: GPT-6
