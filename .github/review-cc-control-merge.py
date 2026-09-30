"""Compose the audited CC control-VJP and split-output implementations."""
import ast
import re
import subprocess
import sys
from pathlib import Path

CONFLICT = re.compile(r'^<<<<<<< .*?\n(.*?)^\|\|\|\|\|\|\| .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n', re.M | re.S)


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('unexpected source anchor: ' + old[:90])
    return source.replace(old, new, 1)


def section(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError('ambiguous section: ' + begin)
    start = source.index(begin)
    stop = source.index(end, start)
    return source[start:stop]


def compose_cpp(ours, theirs):
    source = once(theirs, '#if GENERATIVEQC_HAS_CUDA\nCudaParameterResponseView cuda_parameter_view',
        'struct ControlWeights {\n  std::vector<double> stationarity, orbital_rhs;\n};\n\n#if GENERATIVEQC_HAS_CUDA\nCudaParameterResponseView cuda_parameter_view')
    old = section(source, 'SmallResponseWeights fock_small_pullback(', 'double max_abs(')
    new = section(ours, 'ControlWeights hamiltonian_control_pullback(', 'double max_abs(')
    source = once(source, old, new)
    old = section(source, '  auto fock_dispatch = ', '  double minimum_same_space_gap = ')
    new = section(ours, '  auto control_dispatch = ', '  double minimum_same_space_gap = ')
    new = new.replace('detach_cuda_response(', 'detach_cuda_small(').replace('cuda_response->hamiltonian(', 'cuda_response->hamiltonian_small(')
    source = once(source, old, new)
    old = section(source, '  for (std::size_t index = 0; index < bar_fock.size(); ++index)\n    accumulated_fock_seed[index]', '  double same_space_stationarity = ')
    new = section(ours, '  add_same_space_fock_seed(parameters, bar_fock, o, v);\n  correlation = ControlWeights{};', '  double same_space_stationarity = ')
    source = once(source, old, new)
    release = section(ours, '  // The control response and dense curvature oracle are dead', '  for (std::size_t index = 0; index < dimension; ++index)\n    parameters.fov[index] -= ')
    source = once(source, '  add_fock_seed(parameters, accumulated_fock_seed, o, v);\n', release)
    source = once(source,
        '  const auto fock_small_arena = bytes(generated::fock_small_weights_arena_elements(o, v));',
        '  const auto control_arena = bytes(generated::hamiltonian_control_arena_elements(o, v));')
    source = once(source, '                     fock_small_arena}),', '                     control_arena}),')
    source = once(source,
        '  // Correlation and canonicalization both remain live through the final\n  // pullbacks. The ERI cotangent is isolated, but the response owners are not freed.',
        '  // Retain the conservative split-response envelope while introducing the\n  // control-only CPU arena. The control owner is smaller than this bound and\n  // is released before final small/ERI publication; no admission cap is relaxed.')
    source = once(source,
        '  // Z solution and independent residual survive the solve; basis/action are\n  // already included in response_base. Moving final weights does not free them.',
        '  // This remains an upper bound after early control-workspace release.\n  // Moving final weights into the derivative consumer does not free their data.')
    for required in ('force_source_matches_system(system, source.orbital())',
                     "tensor::cpu_congruence('T'", 'rank2_transform_phase',
                     'auto total_eri = eri_dispatch(parameters, 1.0);',
                     'weights.two_electron = std::move(total_eri);',
                     'cuda_response->hamiltonian_small(', 'run_hamiltonian_control_cpu'):
        if required not in source:
            raise RuntimeError('lost upstream/control contract: ' + required)
    if 'accumulated_fock_seed' in source or 'fock_dispatch' in source or 'detach_cuda_response(' in source:
        raise RuntimeError('obsolete response path remains')
    return source


def resolve_file(path, expected, choice):
    source = path.read_text()
    matches = list(CONFLICT.finditer(source))
    if len(matches) != expected:
        raise RuntimeError(f'unexpected conflict count in {path}: {len(matches)}')
    for index, match in reversed(list(enumerate(matches))):
        source = source[:match.start()] + choice(index, match[1], match[2], match[3]) + source[match.end():]
    ast.parse(source)
    path.write_text(source)


def apply(root, ours, theirs):
    relative = 'src/cc/rccsdt_force.cpp'
    (root / relative).write_text(compose_cpp(ours, theirs))
    def generator(index, own, base, master):
        if index == 1:
            control = '''    hamiltonian_control = _prepare_production(
        Program(
            {
                name: hamiltonian.weights.outputs[name]
                for name in ("stationarity", "orbital_rhs")
            },
            provenance={
                "parent": hamiltonian.weights.logical_hash,
                "scope": "orbital-control response without retained ERI cotangent",
            },
        ),
        "cpu",
    )
'''
            return control + master
        if index == 4:
            return own + '            ),\n            _required_function(\n' + master
        if index == 5:
            return own + '                },\n            ),\n            _cpu_function(\n' + master
        return own + master
    resolve_file(root / 'tools/generate_rccsd_native.py', 6, generator)
    resolve_file(root / 'tests/python/test_rccsd_native_hamiltonian_codegen.py', 1,
                 lambda index, own, base, master: own + '\n\n' + master)


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    own = subprocess.run(['git', 'show', '0e77a96aec26a93b4d3707c13182e1b7165718bb:src/cc/rccsdt_force.cpp'], cwd=root, check=True, capture_output=True, text=True).stdout
    master = subprocess.run(['git', 'show', '268e7112ebb81db46755054ac1cc42374379040c:src/cc/rccsdt_force.cpp'], cwd=root, check=True, capture_output=True, text=True).stdout
    apply(root, own, master)
