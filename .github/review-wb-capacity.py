"""Refresh only the five audited WB97M owner/ABI source contracts."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def cpp_block(source, marker):
    if source.count(marker) != 1:
        raise RuntimeError('ambiguous native source marker: ' + marker)
    begin = source.index(marker)
    opening = source.index('{', begin)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == '{':
            depth += 1
        elif source[index] == '}':
            depth -= 1
            if depth == 0:
                return digest(source[begin:index + 1].encode())
    raise RuntimeError('unterminated native block')


def apply(root):
    qualifier = root / 'tools/dft_mp_v1/qualify_capacity.py'
    tests = root / 'tests/python/test_dft_mp_v1_capacity.py'
    python_source = (root / 'python/generativeqc/_stationary_cuda.py').read_text()
    source_tree = ast.parse(python_source)
    owners = [node for node in source_tree.body
              if isinstance(node, ast.ClassDef) and node.name == '_CudaSources']
    if len(owners) != 1:
        raise RuntimeError('ambiguous stationary source owner')
    initializers = [node for node in owners[0].body
                    if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    if len(initializers) != 1:
        raise RuntimeError('ambiguous source initializer')
    native = (root / 'src/dft/stationary_gradient_cuda.cuh').read_text()
    updates = {
        'STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256': digest(ast.get_source_segment(python_source, initializers[0]).encode()),
        'STATIONARY_SOURCES_OWNER_CONTRACT_SHA256': digest(ast.get_source_segment(python_source, owners[0]).encode()),
        'NATIVE_STATIONARY_GEOMETRY_EXTERNAL_CONTRACT_SHA256': cpp_block(native, 'int stationary_geometry_external('),
        'NATIVE_STATIONARY_GEOMETRY_ENQUEUE_CONTRACT_SHA256': cpp_block(native, 'int stationary_geometry_enqueue('),
        'NATIVE_STATIONARY_HEADER_CONTRACT_SHA256': digest(native.encode()),
    }
    source = qualifier.read_text()
    expected = tests.read_text()
    assignments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node
    report = {}
    for name, new in updates.items():
        if name not in assignments:
            raise RuntimeError('missing audited contract: ' + name)
        old = ast.literal_eval(assignments[name].value)
        if not isinstance(old, str) or not re.fullmatch('[0-9a-f]{64}', old):
            raise RuntimeError('unexpected old contract: ' + name)
        if old == new:
            continue
        if source.count(old) != 1 or expected.count(old) != 1:
            raise RuntimeError('unexpected fingerprint occurrence: ' + name)
        source = source.replace(old, new, 1)
        expected = expected.replace(old, new, 1)
        report[name] = {'old': old, 'new': new}
    ast.parse(source)
    ast.parse(expected)
    qualifier.write_text(source)
    tests.write_text(expected)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    apply(Path(sys.argv[1]).resolve())
