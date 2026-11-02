# Decision: share budget-Fock SCF controls with the native SDK

Status: implemented
Date: 2026-10-29

## Problem

Application precision evidence already has a native admission policy, but Fast
candidate construction still selects energy/density/screening tolerances using
Python budget thresholds. SDK consumers cannot query that same selection.

## Decision

Add a versioned allocation-free policy beside native precision admission. Its
inputs are model traits, availability from the existing Direct or fitted-model
qualification owner, requested allowances and explicit caller ceilings. Native
code owns the incumbent threshold selection and minimum with each caller ceiling.
It returns no override for unqualified traits or tighter budgets; this keeps the
ordinary strict route. It does not select a new provider or qualify a method.

The Python facade marshals those facts and uses the native decision. Only a
missing symbol selects the old policy. A selected refusal or invalid receipt
propagates, so an error cannot silently change scientific controls. Public API
validation remains finite/positive and the native adapter validates independent
SDK inputs transactionally.

## Invariants and alternatives

The three allowance boundaries remain inclusive. One-ULP tighter budgets stay
strict. An omitted RMS budget is distinct from a restrictive RMS request. Explicit
tighter controls survive selection. Independent audit scheduling, model/gradient
qualification, equations, grids and precision arithmetic remain their existing
owners; the selector is not an observable error bound.

Duplicating the fitted method qualification catalog in C++ would create a second
authority and is outside this slice. General candidate/fallback orchestration
requires a separate prepared-owner contract, not additions to a scalar query.

## Evidence and revisit conditions

Always-on Release SDK checks exercise exact/adjacent thresholds, caller ceilings,
unsupported traits, ABI errors and transactional invalid inputs. The compiled
public-adapter Python tests verify real native consumption and absent-only
compatibility. Full endpoint qualification must retain existing independent
numerical, work and 1.05 timing gates before Ready. Revisit this transport boundary
when the native prepared precision controller owns model qualification itself.

Related: #2227 R4/R6; R3 excluded. This does not complete either full slice.
