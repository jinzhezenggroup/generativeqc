# Shell-component value-consumer qualification

This retains the default-off orders 5--8 experiment in #1848, frozen at
`64780b7355f628892886d5e08c32efe7c38ca34b`. All measurements use finite n1
RTX 5090 Slurm allocations. Node3 was not used. Source and library identities,
assigned devices, scripts and exact raw text are in `evidence.json.gz`;
`summary.json` records its digest and derived results. Binaries are identified
by digest and are not embedded.

The three-atom full-def2-TZVPD energy/force ABBA comparison (job 5761) accepts
all 72 calls, including the bracketing CUDA-LibXC GPU4PySCF references. Each
process measures cold, five warm calls, displaced geometry and five displaced
warm calls. Candidate/baseline ratios of the two process medians are 1.077713
for cold, 1.051273 for warm, 1.101970 for displaced geometry and 1.050345 for
displaced warm. This is a regression, so the experiment remains off and no
README performance graph or default promotion is justified.

The twelve-atom census (job 5762) is a separate identity-density resident-value
measurement, with no SCF or force timing. It preserves the 407,065,289 full/LR
radial admissions per operator. Each separate high-order call executes
5,308,640 common preparations and 487,996,523 primitive-component contractions;
the joint call shares geometry across both radial operators. Preparation reuse
does not imply a speedup or FLOP count. No twelve-atom endpoint improvement,
larger-size crossover or combined benefit with another PR is established.

Qualification job 5760 passes retained values/derivatives, the independent
component matrix and allocation gates, six sanitizer runs and nine checkpoint
cases. The [Agent Note](../../../.agents/notes/proposed/2026-10-04-shell-component-value-consumer.md)
also preserves the two earlier synchronization failures and the corrected SASS
observation. The successful qualification does not erase those failed revisions.

From the repository root, the following offline check authenticates embedded
text and repeats the existing complete-endpoint acceptance checks. It neither
loads a GPU nor executes the embedded diagnostic scripts:

```bash
PYTHONPATH=python:. python - <<'PY'
import gzip, hashlib, json
from pathlib import Path
from tools.render_omol25_benchmarks import validate

root = Path('benchmarks/results/wb97mv-shell-components-20261004')
summary = json.loads((root / 'summary.json').read_text())
packed = (root / 'evidence.json.gz').read_bytes()
if hashlib.sha256(packed).hexdigest() != summary['evidence_sha256']:
    raise ValueError('evidence digest differs')
files = json.loads(gzip.decompress(packed))['files']
for name, member in files.items():
    data = member['text'].encode()
    if len(data) != member['bytes'] or hashlib.sha256(data).hexdigest() != member['sha256']:
        raise ValueError('embedded bytes differ: ' + name)
reference = json.loads(files['endpoint3/reference-a.json']['text'])
for name in summary['endpoint3']['processes']:
    validate(json.loads(files['endpoint3/' + name + '.json']['text']), reference)
print('Evidence integrity and all 72 endpoint calls pass')
PY
```

To rerun real-device work, extract the recorded scripts and probe/input text,
adjust toolchain paths, and build with verified ccache against the exact frozen
headers/library. The census consumes internal plan ABI and must not be linked
against an arbitrary newer build. Use finite GPU Slurm and preserve its assigned
visibility. New timings require a fresh same-source, same-GPU comparison.
