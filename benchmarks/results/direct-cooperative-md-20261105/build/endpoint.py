"""Run one scheduled current-default population, keeping oracles out of native."""

import argparse
import importlib.abc
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', action='store_true')
parser.add_argument('--inputs', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--method', default='r2scan-3c')
parser.add_argument('--warm-repeats', default='1')
args = parser.parse_args()


class NoOracle(importlib.abc.MetaPathFinder):
    """Reject an accidental production dependency on either reference engine."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'pyscf', 'gpu4pyscf'}:
            raise ImportError('native endpoint attempted reference import: ' + fullname)


if not args.reference:
    sys.meta_path.insert(0, NoOracle())
from benchmarks import readme_defaults

readme_defaults.main([
    'gpu4pyscf' if args.reference else 'standard', '--method', args.method,
    '--mode', 'direct', '--atoms', '96', '--warm-repeats', args.warm_repeats,
    '--inputs', args.inputs, '--source-revision', args.source, '--output', args.output,
])
