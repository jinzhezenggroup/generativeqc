# Shared FP32

WB97M-V complete E/force medians; RTX5090, grid48/16/32, three repeats:
- 24 AO: FP64 6.783615s / Fast 4.987146s = 1.360220x.
- 96 AO: FP64 24.434694s / Fast 18.176775s = 1.344281x.

Summary: identities, samples, errors, work, slower baseline and sanitizer failures.
96 AO means 12 atoms, not 96. Run decompressed script via finite `srun`.
Raw: `../evidence-vv10-range-fp32-20261015/`.
