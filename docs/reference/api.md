# Python API

This source page is populated at Sphinx build time from public Python modules.

A module opts into the generated reference by declaring a literal `__all__`.
The renderer discovers those modules recursively under `python/generativeqc`,
so adding a new public module does not require editing this file or the
documentation navigation. Private modules and modules without `__all__` are
not promoted into the public API by documentation discovery.
