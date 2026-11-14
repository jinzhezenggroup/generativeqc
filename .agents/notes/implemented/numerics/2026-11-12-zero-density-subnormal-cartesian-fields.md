# Decision: admit raw subnormal gradients at a rounded zero spin density

Status: implemented
Date: 2026-11-12

The actual public native H3 UKS control on its eight-radial-point moving grid
exposed a derivative-source admission false positive. At point254, beta density
rounded to zero while differentiated AO products retained Cartesian gradients
`[0, 9.8813129168249309e-324, -9.8813129168249309e-324]`. The linked production
XC graph already owns its lazy vacuum branch, but the native source rejected
every nonzero gradient at an exactly zero represented density before reaching
that graph. The same guard also affected fixed-grid response and mixed/column XC.

A shared field-admission helper accepts only zero/subnormal Cartesian gradients
at zero density. It preserves all raw density, gradient and tangent inputs to the
generated graph: no clipping, floor, altered scalar algebra or residual gate.
Negative/nonfinite density, nonfinite gradients, and normal-scale gradients at
zero density still refuse. Extreme tail cases beyond this bounded admission
are not silently projected into a physical field.

The native C/API control checks invalid-field refusal as well as the observed
underflow class. The original actual H3 UKS public/independent PySCF+D3 multistep
matrix/HVP controls and their original gates remain in place. The Python-free
CLI exercises the same actual RKS/UKS geometry instead of avoiding its tail.
This is not a heavy-element, large-system or saddle qualification.
