# Python API

This source page is populated at Sphinx build time from public Python modules.

A module opts into the generated reference by declaring a literal `__all__`.
The renderer discovers those modules recursively under `python/generativeqc`,
so adding a new public module does not require editing this file or the
documentation navigation. Private modules and modules without `__all__` are
not promoted into the public API by documentation discovery.

Public modules and their re-exported Python declarations are checked without
importing the native runtime. The documentation CI enforces Ruff
`D100,D101,D102,D103,D104,D105,D107` on every `__all__` module, including
new ones, and requires each exported first-party class/function to have a
source docstring. A Python module without `__all__` is not itself a public
module, but its re-exported public declarations are still checked.
