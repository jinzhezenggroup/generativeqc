# Capability sources and interpretation

GenerativeQC deliberately avoids a second handwritten support matrix. Public
support depends on method identity **and** execution context, so a static table
of method × CPU/GPU × basis × spin × precision × density fitting × ECP ×
property combinations would become stale quickly.

Use the following sources for different questions.

## Public method discovery

The generated [public method catalog](../public_methods.md) is the documentation
view of the public selector inventory. Sphinx renders it from:

- `manifests/public_methods.json` for stable native ABI/provider registrations;
- compiler-discovered DFT selectors;
- public composite selectors; and
- the automatic pinned-Libxc semilocal inventory and explicit blacklist.

The equivalent runtime discovery command is:

```bash
python -m generativeqc methods --json
```

Discovery answers **which public method names exist**. It is not a blanket claim
that every execution context for that name is qualified.

## Execution-context capabilities

After the method, backend, basis, spin, precision, density-fitting mode, ECP,
grid, and other execution choices are fixed, `Calculator.capabilities` is the
authoritative public capability view for that calculation. Prepared batches
expose the corresponding contextual capability record.

This layer is where properties that depend on the fully resolved execution
context are admitted. Unsupported combinations fail closed rather than being
inferred from a nearby method or backend.

This distinction is intentional: the stable native ABI manifest can remain
conservative while a fully resolved Python calculation advertises an
independently qualified property for its exact context.

## Low-level compiler and backend evidence

`docs/codegen_capabilities.json` records shell/code-generation capability
evidence. `docs/cuda_ownership/` records CUDA semantic ownership. The generated
[bulk Libxc import report](../libxc_bulk_import.md) and
[Maple importer coverage](../libxc_maple_coverage.md) describe importer state.

These are implementation and validation inputs. A low-level generated kernel,
imported functional, compiler representation, or ownership entry does not by
itself establish an end-to-end public method/property capability.

## Documentation rule

Reader-facing pages should link the generated catalog or contextual capability
API instead of copying broad support tables. A narrow page may state the exact
domain it documents, but cross-method support summaries should remain generated
or runtime-derived so backend and derivative gates cannot silently drift from
the implementation.
