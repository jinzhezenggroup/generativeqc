"""Replay the current source's emitted CPU stencils without running a solver."""
from pathlib import Path
import ast
import hashlib
import json
import math
import subprocess

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1] / 'generativeqc-h100-current-evidence'
REVISION = 'db3657f08d983791c45b7d949cb9028fe0055e3f'


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    admission = json.loads((TASK / 'native-qualification-v1-local-replay-v3.json').read_text())
    require(admission['status'] == 'PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK' and admission['source_revision'] == REVISION, 'Native replay required')
    path = 'benchmarks/results/a100-native-stages-20261102/report.py'
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', REVISION + ':' + path])
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['distance', 'rsh_stencils']]
    require(len(functions) == 2, 'Exact source-owned stencil reader required')
    namespace = {'json': json, 'math': math, 'require': require}
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, 'exec'), namespace)
    rows = {}
    for mode in ['normal', 'memcheck']:
        raw = TASK / 'native-qualification-v1' / (mode + '-generativeqc_cuda_fock_composition_tests.stdout')
        rows[mode] = {'stdout_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), **namespace['rsh_stencils'](raw.read_text())}
    result = {'status': 'PASS_CURRENT_SOURCE_INDEPENDENT_RSH_STENCIL_REPLAY', 'source_revision': REVISION, 'reader_source_path': path, 'reader_source_sha256': hashlib.sha256(source).hexdigest(), 'executions': rows, 'scope': 'Exact source-owned reader reused for new H100 raw outputs: two CPU steps, 240 stencils per execution, host/resident/borrowed full vectors and pending-work failure witnesses. This does not transfer A100 device acceptance or supply clean timing votes.'}
    with (TASK / 'native-rsh-stencils-v1-local-replay.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
