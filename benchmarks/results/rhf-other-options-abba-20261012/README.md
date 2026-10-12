# Other complete-force option ablations

Frozen source `531364e133ae145097b402d66fec5036ed051365` plus the retained
shell-bilinear patch. These are **not** the new default PR's frozen
`a50eb54f` measurements. Each cohort is four fresh complete ethane230 E+F
processes in ABBA order, on the same physical RTX 5090 under Slurm; strict FP64,
CCSD Q32, Lambda Q32, immutable core reuse and single-shell-pass derivatives
are held fixed. The scalar cohort requests no gap output, its admitted domain.

| Existing option | Off mean E+F | On mean E+F | Observation |
| --- | ---: | ---: | --- |
| Triples multi-output scalar fusion | 227.393698 s | 219.492646 s | 3.475% less endpoint time |
| Lambda serial scratch reuse | 227.397166 s | 227.398488 s | No identifiable time benefit |

Fusion is actually selected with no resource fallback. Triples phase falls
39.459958 -> 31.642801 s; generated scalar kernels fall 1320 -> 165 per endpoint.
Its planned scalar workspace grows 604456216 -> 1036210656 bytes, while sampled
whole-endpoint VRAM peaks do not increase because another phase dominates.

Scratch reuse reduces reported Lambda capacity 14127345193 -> 13266849577
bytes and sampled whole-endpoint VRAM 26500 -> 25678 MiB (822 MiB saved).
This verifies effective capacity reduction, not just a request flag. Expensive
work is unchanged: CCSD 19 iterations, Lambda 21 iterations/22 solver actions,
416 reported auxiliary batches, 14307 Lambda GEMMs and one shell derivative pass.

All eight endpoints separately pass the retained independent energy/force,
scoped two-component/two-step large FD, translation and response residual gates.
Energy span is zero; force span is 5.151435e-14. The retained force is a complete
qualified reference; the large independent FD receipt is not an all-component
FD audit. Raw outputs, controls, process timings, sampled memory traces and telemetry summaries/hashes, reference inputs,
Slurm allocation and binary hashes are retained in the selected receipts. Full telemetry remains in the ignored local run directory.

These are scoped two-sample-per-side observations, not generalized statistical
performance qualification. Neither option is promoted here. This is the
existing **triples scalar** consumer fusion, not a new Lambda Q-batch
producer-consumer fusion. No native DIIS-history preconditioner or f-shell
component tiling is implemented by these experiments.

Reconstruct the exact source and patch, build the strict Release/sm120 CUDA
12.9.1 endpoint with verified compiler caching, then place it in the layout in
`reproduce.sh`. Run only through the retained finite Slurm command, preserving
assigned device visibility. Checked-in ethane230 and reference inputs suffice;
no external release/archive is required.

Already tracked ethane230/reference inputs are retained by exact repository
paths, source revisions and byte hashes in the receipts instead of duplicating
their payloads. All endpoint outputs, numerical gates and source patches remain
in this bundle; reproduction depends on Git inputs, not an expiring archive.
