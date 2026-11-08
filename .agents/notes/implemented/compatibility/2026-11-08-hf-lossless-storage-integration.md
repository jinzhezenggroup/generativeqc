# Decision: retain complete HF evidence through exact-FP64 storage

Status: implemented; storage only, no new scientific qualification
Date: 2026-11-08

Integrating the derivative-AD publication into the shared-Rys branch exceeded
the unchanged 85-MiB checkout evidence budget. Increasing that budget, dropping
failed force populations, discarding compiler stdout, or hiding observations in
another directory would not fix the underlying duplicated/decimal storage.

Reuse the existing authenticated exact-FP64 v3 decoder instead. The five related
2026-11-08 HF populations share metadata and FP64 streams while retaining all
459 exact original UTF-8 texts, original per-member hashes and separate cohort
populations. Each cohort pins the shared stream manifest. The two older HF cold
work and ordered-Gram publications independently use the same codec for four
complete original JSON texts; their historical source and qualification claims
do not change. Compress two derived summaries without changing their bytes.

The format separates reversible FP64 words from text; it does not round values,
filter outcomes or substitute regenerated references. Existing independent
oracles, full force arrays, work counts, compiler output and failure evidence
remain available offline, including in source exports without `.git`. The
existing decoder verifies every recovered original byte hash and full FP64
consumption. New corruption controls exercise both metadata and FP64 streams.

No current-head molecular qualification follows from storage equivalence.
In particular, the historical default HF48 force diagnostic still fails its
original gate; the source-matched numerical fix and complete endpoint evidence
must be assessed separately. The frozen-VV10 rejection is also unchanged.
