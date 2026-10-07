# Python API

This reference is generated at Sphinx build time from the importable public Python
surface. Function and class signatures, type annotations, docstrings, inheritance,
and source links therefore stay synchronized with the code.

For the core package and programmable extension modules, `__all__` is the
authoritative export list. Adding or removing a public symbol there changes this
reference automatically. Private implementation names are intentionally excluded.

## Public namespaces

```{eval-rst}
.. autosummary::

   generativeqc
   generativeqc.torch
   generativeqc.extensions.method
   generativeqc.extensions.tensor
   generativeqc.extensions.xc
```

## Core API

```{eval-rst}
.. automodule:: generativeqc
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```

## PyTorch integration

`generativeqc.torch` is an optional integration. The documentation build mocks
the optional `torch` import so API generation does not require installing
PyTorch.

```{eval-rst}
.. automodule:: generativeqc.torch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Programmable method API

See the [extension guide](../developer/extensions.md) for the semantic contract
and activation rules.

```{eval-rst}
.. automodule:: generativeqc.extensions.method
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```

## Programmable TensorIR API

```{eval-rst}
.. automodule:: generativeqc.extensions.tensor
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```

## Programmable XC API

```{eval-rst}
.. automodule:: generativeqc.extensions.xc
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```
