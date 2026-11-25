# Decision: authenticate the linked CPU Hessian electronic/total pair

Status: implemented
Date: 2026-11-24

## Problem

The Python preflight checked that the two model identities were individually
linked. That cannot distinguish a qualified correction from a total closure
linked to another electronic model. The accepted native owner still had to
reject the mismatch, after frontend work that could have been avoided.

## Decision

The CPU Hessian linker compares the resolved bare owner of the electronic
identity with the owner of the total identity. The check is a preflight
receipt, never an SCF state, basis, property or resource admission promise.
An older library with no paired query retains its current individual checks;
a completed current query never falls back after a mismatch.

Refs #2227 and #932.
