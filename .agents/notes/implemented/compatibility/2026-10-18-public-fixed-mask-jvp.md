# Decision: explicit non-differentiated primals for public fixed-mask JVP

Status: implemented
Date: 2026-10-18

## Problem

P0 #2137 requires real branch directional derivatives for where. Public JVP
marks every supplied primal differentiable, so an external Boolean selector
cannot be expressed. Canonical generated fixed-mask JVP already owns the rules.

Before exposure, a reference-only reproduction also showed that linearize
accepted declared but disconnected FP32 and AO-annotated direction inputs. Its
live-primal domain proof did not cover those requested input definitions.

## Decision

Keep matching positional tuples. With allow_external_mask_where=True only,
None explicitly marks a non-differentiated primal; all remaining real directions
are validated and at least one is required. Do not infer selection from dtype.
Reuse linearize, including its zero-direction and eager-forward-domain guards.
Extend the existing primal/derivative graph combination to use the generated
execution inputs for JVP as for VJP, preserving the non-differentiable typed
boundary rather than restoring input flags to combine graphs.

Share the existing VJP requested-input domain proof with JVP. This closes the
disconnected-input admission hole without changing accepted derivative equations.
Generic finite FP64 and reference/materializing CPU remain the qualified scope.
Default calls, computed predicates, Boolean AD, scientific/precision policies,
higher-order selection and CUDA remain outside this opt-in.

## Rejected alternatives

Automatically skipping Boolean inputs would silently change call semantics.
Numeric zero mask directions would misrepresent Boolean differentiation. Another
argument-selection implementation would duplicate the VJP work in #2604; an
explicit None placeholder retains the existing tuple alignment. Adding a public
runtime-only domain audit would duplicate compiler-owned qualification.

## Evidence

Independent NumPy analytic derivatives, central finite differences and dot
identities cover nested selections, broadcasting, repeated gather and nonlinear
contractions. Reference/native cases qualify omitted real directions, empty and
disconnected outputs, all supplied seeds, singular unselected forward domains,
bit preservation, serialized execution boundaries, budgets and lifetime. Native
tests forbid the reference interpreter and reference AD evaluators. Analytic
acceptance uses rtol/atol 2e-13 and finite differences 2e-8 with step 1e-5.

## Revisit when

Additional scientific/dtype/backend or higher-order selection qualification is
available. #2137 and its parent #2111 remain broader than this public JVP slice.
