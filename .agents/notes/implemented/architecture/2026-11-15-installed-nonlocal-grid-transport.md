# Decision: publish the existing independent nonlocal-grid transport

Status: implemented
Date: 2026-11-15

## Problem

The prepared native KS owner already copies and executes an independent VV10
quadrature. Python could prescribe that context, but its descriptor was defined
only in private C++ headers. An installed SDK consumer could not prescribe the
same calculation without reverse-engineering a private layout, and the native
CLI exposed only the XC grid. Same-grid force parity does not establish parity
for independently prescribed XC/nonlocal quadratures.

## Decision

Move the existing size-tagged precision/nonlocal transport definitions into the
installed C header, preserving their field layout and the existing discovery
symbol. Remove the private definitions instead of retaining aliases or adding
a competing preparation API. The C++ named helper accepts an optional independent
grid and forwards it through that descriptor. The CLI exposes a level control
for `run`; other commands explicitly refuse the new control until their own
contracts consume it.

Native preparation still owns model identity, quadrature validation, copied
input lifetimes and contextual force admission. The named adapter replaces only
scientific fields from its generated catalog; it preserves both quadratures.
No derivative equation, force capability, precision default or capacity promise
changes. In particular, publishing the precision prefix does not newly qualify
mixed arithmetic or make an error budget an observed error guarantee.

## Alternatives and invariants

- Do not duplicate the private descriptor in an installed downstream example.
- Do not force both grids to be equal merely to pass a three-product test.
- Do not accept an independent grid on a model without a nonlocal term.
- Do not silently ignore the control in resources, Hessian or optimization
  commands that do not yet transport it.
- Independent scientific acceptance, resource/lifecycle gates and matched full
  endpoint performance remain necessary; this transport repair is not complete R1.

## Validation contract

The installed SDK/CLI/Python force matrix includes both equal level-zero grids
and XC level zero with independent nonlocal level one, for both spins,
representations and Direct/DF contexts. PySCF independently reconverges the
specified quadratures and computes moving-grid analytic forces. The native
named tests exercise invalid extension versions, reserved fields, nested ABI
tags and copied-input lifetime. Raw source-matched CPU/CUDA runs are retained
separately from this architectural rationale.

## Revisit when

Additional CLI commands gain complete independent-grid transport and endpoint
qualification, or the application-precision descriptor itself is redesigned.
