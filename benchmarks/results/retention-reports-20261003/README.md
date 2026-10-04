# Historical report retention, 2026-10-03

Eight finished campaign payloads leave the normal checkout: six ZIP archives
and two JSON reports, totaling **1,669,507 bytes** before the small documentation
and inventory updates. Their exact bytes remain recoverable from published
master commit `1a4acc519eb881cc19d418de65ecca72324359d4`. This reduces current
checkout size, not existing Git history or full-clone size. No history rewrite,
external upload, new scientific run, budget change, or tolerance change occurs.

## Why Markdown is sufficient here

These are historical reports, not present-day scientific inputs. The only
archive consumers were tests of ZIP hashes/member counts; those now exercise
the same unpack/restore contract on tiny synthetic archives and temporary Git
repositories. Source search found no active scientific consumer of either
removed JSON report. No active publication manifest references these payloads.
All current numerical reference fixtures, regression inputs, and live
publication/recomputation datasets remain unchanged.

| Family | Preserved historical result and limits |
| --- | --- |
| [RCCSD A](../rccsd-148-a/README.md) | Energy/physical T1 oracle agreement; CPU scope only; no T2 solver, GPU, or performance claim |
| [RCCSD B](../rccsd-148-b/README.md) | Physical R2 and intermediate parity; independent determinant/PySCF checks; provenance repair retained |
| [RCCSD C](../rccsd-148-c/README.md) | Ten small-system CPU endpoints; independent physical residual and FCI checks; failure handling and later provider-preflight correction retained |
| [RCCSD GPU A](../rccsd-149-a/README.md) | Eight fixed-amplitude H100 records, minimum-budget rejection and memcheck; no resident iterative solver or full molecular GPU acceptance |
| [Fixed-density XC](../xc-integration-162/README.md) | 24 CPU LDA/PBE comparisons plus directional checks; controlled quadrature only, no SCF/quadrature convergence or GPU claim |
| [DF matrix](../issue206-df-a/README.md) | Four converged endpoints; numerical parity; unequal iteration branches and incomplete component accounting preclude a speed claim |
| [SCF proposals](../scf-proposals-186/README.md) | 115 converged/15 deliberate failures; extra Fock work, diagnostic timings, unverified intended state, no learned-acceleration claim |
| [Tensor CUDA](../tensor-cuda-146/README.md) | 18 fixed-tensor cases; rejected plans, provider allocation allowance and scope limits retained; historical selection only |

The family READMEs retain the measured identities, commands, acceptance gates,
failed/rejected outcomes and limitations. ZIP member manifests stay beside them.
The two reports' older trace/tuning archives retain their existing separate
recovery manifests. Exact timing samples are available for historical analysis;
Markdown summaries must not be treated as recomputable statistical datasets or
as new/current performance qualifications.

## Recovery

The [snapshot manifest](snapshot.manifest.json) uses the existing
`generativeqc.git-snapshot.v1` schema: full source revision, original paths,
byte lengths, SHA-256 digests, file count and total bytes. All eight files were
restored and matched against their original bytes before removal, and all six
ZIPs passed their retained per-member manifests. The source revision is the
storage anchor; original experiment revisions/dirty-source records inside the
payloads are unchanged and identify the measurements.

From a Git checkout containing that revision, restore all eight stored files:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-reports-20261003/snapshot.manifest.json \
  --output .artifacts/retention-reports-20261003
```

The command verifies every file before creating the destination and refuses to
overwrite one. To inspect a ZIP without extracting or running historical code:

```bash
python -m tools.unpack_evidence benchmarks/results/issue206-df-a \
  --archive .artifacts/retention-reports-20261003/benchmarks/results/issue206-df-a/raw-evidence.zip
```

Add `--output <new-directory>` to unpack verified members. The family READMEs
also show single-file restoration. Neither helper fetches implicitly.

A source distribution without `.git`, or a shallow/partial clone lacking the
recorded objects, cannot restore these optional historical records by itself.
Use a full project clone, or explicitly fetch the pinned revision in a Git
checkout before retrying:

```bash
git fetch origin 1a4acc519eb881cc19d418de65ecca72324359d4
```

Normal archive-format/recovery unit tests create their own tiny local fixtures;
they do not depend on this repository's old objects. Historical restoration is
an explicit audit and fails if required objects are unavailable. Successful
byte recovery is distinct from rerunning or independently validating science.
