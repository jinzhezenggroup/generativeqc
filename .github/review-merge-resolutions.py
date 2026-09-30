"""Resolve only fixed-head, reviewed conflict hunks in PRs 1510/1589/1597."""
import ast
import hashlib
import re
import sys
from pathlib import Path

CONFLICT = re.compile(
    r"^<<<<<<< .*?\n(.*?)^\|\|\|\|\|\|\| .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n",
    re.M | re.S,
)


def resolve(root, relative, count, choose):
    path = root / relative
    source = path.read_text()
    matches = list(CONFLICT.finditer(source))
    if len(matches) != count:
        raise RuntimeError(f"unexpected conflict count in {relative}: {len(matches)}")
    output = source
    for index, match in reversed(list(enumerate(matches))):
        replacement = choose(index, match[1], match[2], match[3])
        output = output[:match.start()] + replacement + output[match.end():]
    if '<<<<<<< ' in output or '>>>>>>> ' in output:
        raise RuntimeError(f"unresolved conflict in {relative}")
    if path.suffix == '.py':
        ast.parse(output)
    path.write_text(output)


def apply(pr, root):
    if pr == 1510:
        header = root / 'src/dft/stationary_gradient_cuda.cuh'
        digest = hashlib.sha256(header.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for path in ('tests/python/test_dft_mp_v1_capacity.py',
                     'tools/dft_mp_v1/qualify_capacity.py'):
            def choose(index, ours, base, theirs):
                old = 'c4327728312261019983a95120ef3f34a0910a0f134068c596c2bb8a91b30f0b'
                if ours.count(old) != 1:
                    raise RuntimeError('unexpected capacity fingerprint hunk')
                return ours.replace(old, digest)
            resolve(root, path, 1, choose)
        print('Audited merged native header SHA256:', digest)
    elif pr == 1597:
        resolve(root, 'docs/developer/hessian.md', 2, lambda i,o,b,t:o)
        def response(index, ours, base, theirs):
            property_source = '''    @property
    def basis(self) -> NativeAO:
        """Borrowed AO owner shared with response and Hessian consumers."""
        return self._basis
'''
            if property_source not in ours or 'def basis(self)' in theirs:
                raise RuntimeError('unexpected Hessian basis property hunk')
            if 'self._plan._ensure_open()' not in theirs or 'real_spherical' not in theirs:
                raise RuntimeError('missing upstream response safeguards')
            if not theirs.endswith('            raise\n'):
                raise RuntimeError('unexpected response class tail')
            return theirs + '\n' + property_source
        resolve(root, 'python/generativeqc/rks_response.py', 1, response)
    elif pr == 1589:
        def schedule(index, ours, base, theirs):
            if index == 0:
                generic = theirs.replace('_compiled_profitability_fields', '_compiled_fact_fields')
                generic = generic.replace(
                    "Every native stage must be present before the result is eligible for\n    ProgramIR region selection.",
                    "Every native stage must be present. This unbound pressure summary is\n    not a substitute for the source-bound native evidence used by candidates.",
                )
                bound = ours.replace(
                    '    payload = evidence.profitability.to_payload()\n'
                    '    return typing.cast("dict[str, typing.Any]", payload["compiled"])',
                    '    return _compiled_fact_fields(evidence.profitability)',
                )
                return generic + '\n\n' + bound
            if index == 2:
                return ('''        if self.compiled_evidence is not None and not isinstance(
            self.compiled_evidence, GridXcCompiledRegionEvidence
        ):
            raise TypeError(
                "grid/XC compiled evidence requires GridXcCompiledRegionEvidence"
            )
''' + ours)
            return ours
        resolve(root, 'python/generativeqc_compiler/dft/xc_schedule.py', 6, schedule)
        def schedule_tests(index, ours, base, theirs):
            if index == 0:
                return ours + theirs
            marker = 'def test_compiled_region_evidence_requires_every_native_device_stage()'
            if theirs.count(marker) != 1:
                raise RuntimeError('expected upstream stage-aggregation regression')
            return (ours + '        )\n\n'
                '    with pytest.raises(TypeError, match="GridXcCompiledRegionEvidence"):\n'
                '        GridXcScheduleCandidate(DEVICE_FUSED, shape, GpuProfitability())\n\n\n'
                + theirs[theirs.index(marker):])
        resolve(root, 'tests/python/test_dft_xc_schedule.py', 2, schedule_tests)
        resolve(root, 'tests/python/test_dft_xc_program_region.py', 4, lambda i,o,b,t:o)
    else:
        raise RuntimeError('unapproved resolution target')


if __name__ == '__main__':
    apply(int(sys.argv[1]), Path(sys.argv[2]).resolve())
