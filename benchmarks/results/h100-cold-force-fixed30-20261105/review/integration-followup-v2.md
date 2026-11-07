# Bounded integration follow-up — upstream #3011

**PASS: no blocking integration concern.** Original WARN and supplementv1 remain unchanged.

Head `728e17c1ece72967a033cbc2910e05d55233c0ee` normally merges published `a5c12e2c176f82deea2ef5faddd25db8d15a0e52` and upstream `051a157c64061eac4fd9ee90d7becfa982f80161`.

- **CUDA constructor:** only the imported KS symbol and second-order eligibility assignment change. The original KS-present/composite predicates remain identical. The new OR arm begins `self._device_name == "cpu"` (Calculator lines1261–1271), so CUDA never evaluates the new full-range helper. The CUDA flag value, its auxiliary/ECP guard uses and first-E/F routing stay the same. Extra Python symbol/Boolean bookkeeping exists; no zero-overhead or new timing claim is inferred.
- **Endpoint scope:** all other changed Calculator methods are explicitly Hessian/HVP helpers. Primal SCF diagnostic export and composite `public_calculator_endpoint` fields affect second-order results. The KS module adds only a function definition. No native RKS, force/default, preparation, benchmark timer or batch/precision-batch source changes occur.
- **Frozen evidence:** exactly the two scientific record parts, summary and replay match publisheda5c12e2c byte-for-byte. Original report and v1 supplement hashes also match. No member unpack,720/72 reduction, compiled/device gate or test-suite replay ran.
- **Libxc:** comparison with both actual parents changes only checked source/native identities; all scientific/capability/execution/evidence fields remain equal and `stale:true` is preserved. Documentation changes only these identities.

This source review does not qualify upstream CPU Hessian science or measure head728e17c1e. Existing H100 source/device boundaries and original WARN limitations persist. Normal CI remains responsible for merged interfaces. Same-family/provisional assurance; actual backend identity unknown.

`integration-followup-v2.json` records18 directly relevant input hashes, actual parents and precise scope evidence. No blocker remains in this bounded follow-up.
