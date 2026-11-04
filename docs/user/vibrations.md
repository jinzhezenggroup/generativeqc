(harmonic-analysis-inputs)=
# Analyze a supplied Cartesian Hessian

`generativeqc.analyze_harmonic_modes` analyzes a **supplied** Cartesian energy
Hessian on the host. It performs mass weighting and translation/rotation
projection, not SCF, Hessian generation or transition-state optimization. Its
array identity does not establish that a Hessian belongs to a geometry, method,
basis, ECP or electronic state. It is not a TSopt+freq task.

```python
from pathlib import Path
import numpy as np
from generativeqc import analyze_harmonic_modes

analysis = analyze_harmonic_modes(
    np.load("hessian-hartree-bohr2.npy"),
    coordinates=np.load("coordinates-bohr.npy"),
    masses=np.load("physical-isotope-masses-dalton.npy"),
    hessian_units="hartree/bohr^2",
    coordinate_units="bohr",
    mass_units="dalton",
)
print(analysis.frequencies_cm1)
print(dict(analysis.diagnostics))
```

Inputs must be a complete `(3N, 3N)` energy Hessian, `(N, 3)` coordinates and
`N` positive physical atomic/isotopic masses in the same atom-major x/y/z order.
Only the explicit units shown are accepted. A force Jacobian has the opposite
sign and is not interchangeable with an energy Hessian. ECP core removal never
changes the physical nuclear mass: for example, do not use Cs ionic charge 9
as its mass. The consumer does not infer isotopes or masses from electron counts.

(harmonic-analysis-admission)=
## Projection and integrity

Invalid units, shape, masses, symmetry controls and geometry raise `ValueError`
or `TypeError`; a failed NumPy factorization raises `numpy.linalg.LinAlgError`.
No partial result is returned. No native library, SCF or GPU is invoked.

Raw asymmetry, translation and rotation residuals are reported before mass
weighting and projection. Asymmetry above `raw_symmetry_tolerance=1e-7` fails.
Any nonzero admitted asymmetry requires explicit `symmetrize=True`; the default
does not silently average a raw matrix. Projection can remove rigid-coordinate
contamination for analysis, but cannot qualify the input Hessian or prove
stationarity. Residual diagnostics remain available even after projection.

Translations and rotations are constructed in mass-weighted coordinates about
the center of mass. Rotation rank uses the singular-value ratio and declared
`linear_relative_tolerance=1e-8`. Linear molecules have `3N-5` modes; nonlinear
molecules have `3N-6`. A single atom has no vibrations, distinct diatomics are
linear, and coincident-only multiatom geometries are rejected. Near-linear
decisions retain the ratio, singular values and threshold in diagnostics.

Dense analysis rejects inputs exceeding `maximum_coordinates=600` before
converting the Hessian. This dimension bound is not a process-memory budget.

(harmonic-analysis-modes)=
## Signed modes and visualization

Frequencies are **unscaled**, signed real values in cm⁻¹; negative values denote
imaginary frequencies. No negative or soft mode is deleted. `negative_mode_count`
includes every negative projected eigenvalue. The separate
`significant_imaginary_mode_count` uses the explicit `soft_mode_threshold_cm1=1`
band; this does not classify a transition state or imply IRC connectivity.
Conversion uses the pinned 2022 CODATA SI values.

Mode-major arrays have shape `(number_of_modes, N, 3)`. Mass-weighted vectors
have Euclidean norm one. Cartesian vectors satisfy
`sum(mass_atom * displacement_component**2) = 1` for dalton masses; their vector
units are dalton⁻¹ᐟ². Degenerate modes should be compared as subspaces rather
than individual signed vectors. Result arrays are detached and read-only.

To visualize a mode, supply matching element labels and a zero-based index:

```python
Path("mode.xyz").write_text(
    analysis.xyz_animation(["Cs", "H"], 0, amplitude_angstrom=0.2)
)
```

This emits multi-frame XYZ in Angstrom. The display amplitude sets only the
maximum atom excursion; it does not change the normalized result vectors or
frequencies. Supply labels for the actual input molecule, not the example pair.
Invalid labels, mode indices, nonpositive/nonfinite amplitudes or a frame count
outside 4–1000 raise `ValueError` (nonnumeric amplitudes raise `TypeError`).
The returned string is caller-owned; the exporter does not write files itself.
