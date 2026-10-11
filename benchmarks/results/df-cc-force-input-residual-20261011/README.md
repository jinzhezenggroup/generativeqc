# Incoming-residual mixing for complete native CUDA DF analytic forces

Measured base: `34dd51ed3ebc5b03b17f095de2cd7b39903923c8` plus the retained
production overlay. Fresh cold ethane230 force ABBA medians: control
275.829710826 s; candidate 250.339765314 s, 9.2412% shorter. This is a scoped
descriptive result with two observations per arm, not a statistical/global claim.

Every original native output, process timing/RSS receipt, independent small
force record, JUnit/log receipt, failed staging attempt, separate validation
pilot, compiler/cache receipt and passing reproduction script is retained
losslessly in `raw-receipts.json.xz`. The pilot is never pooled with ABBA.
Read the selected envelope with
`tools.generativeqc_validation.record.load_publication_record`.

Restore raw members by their UTF-8/base64 encodings and check each original
byte length/SHA-256. Rebuild the exact base with its recorded CMake options,
apply `gzip -dc production-source.patch.gz | git apply`, and build the candidate
with the same compiler/provider settings. Retain the library SONAME alias in
each arm directory. Restore the scripts from the raw bundle, adjust only local
artifact paths, and run all real-device work through finite node2 Slurm jobs.
The matching endpoint/probe binaries consume normalized metadata only; no
reference orbitals, factors, amplitudes or response seeds enter production.

All original energy, physical residual and independent force gates pass.
Complete logical numeric capacity is identical, 19,509,839,897 bytes; it is
not physical peak VRAM. Actual native per-graph queries reconstruct CC GEMM,
contraction, packing, Q and accumulation counts. CUDA scientific artifacts,
Lambda and source/orbital/nuclear response are unchanged. Broad production
and statistical performance promotion remain not-run.
