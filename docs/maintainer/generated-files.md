# Generated documentation and data

Some paths under `docs/` are machine-readable repository interfaces and should not be moved merely for visual cleanliness.

Current examples include:

- `docs/public_methods.md`, a source shell populated during Sphinx `source-read` by `tools/render_public_methods_doc.py` from the native/composite manifest and automatic Libxc capability sources;
- `python/generativeqc/_generated_methods.py` is intentionally **not** generated; it is a stable runtime loader that derives Python metadata from the canonical public-method manifest, which is bundled into wheels;
- `docs/codegen_capabilities.json`, consumed by validation tooling; and
- `docs/cuda_ownership/`, consumed by ownership/reporting tools;
- `docs/libxc_bulk_import.md`, a registered hashed importer report; and
- `docs/libxc_maple_coverage.md`, the generated importer-coverage snapshot.

Reader-facing pages should link these authoritative artifacts rather than duplicate them. A future relocation should update producers, consumers, CI checks, and docs atomically.

## Public Python API documentation

`tools/render_python_api_doc.py` discovers public `generativeqc` modules from
literal `__all__` declarations and renders `docs/reference/api.md` during Sphinx
`source-read`. Private modules are not independently published.

Run `python tools/check_public_api_docstrings.py --ruff ruff` after installing
`docs/requirements.txt`. This same gate runs in Documentation CI and pre-commit.
Ruff checks module and definition docstrings in every discovered public module.
The import-free AST audit follows first-party re-exports and additionally requires
docstrings on their defining classes/functions and explicitly defined public
methods, constructors, magic methods, properties, and nested public classes.
It excludes private helpers, unrelated implementation definitions, and inherited
or generated methods; a re-export does not make its entire implementation module
public. Add member documentation at the defining source, preserving its behavior.
