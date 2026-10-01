"""Bounded repairs for eight inspected PRs; each result requires fresh tests."""
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HEADS = {
    1651: ('a40e2981572a87d601d7b4beebc1008436998f22', 'perf/through-f-dft-readme-20261001'),
    1652: ('a9b9ffa0f855e89f91595cf23c8c47e7f7a59976', 'refactor/stationary-composite-cuda'),
    1653: ('3b3fceb41a65d46a522d989cb7f02e54f825156a', 'agent/retire-stale-scientific-ownership'),
    1654: ('1aafacd205c49ccf4365c259c755674a5c33783c', 'chatgpt/dft-df-forces-cpu-cuda-20261001'),
    1655: ('2c4c5a466148ac7a6b95088c1fcb4a3c35815b21', 'agent/retire-vv10-local-scale'),
    1657: ('beeb397465175b2690d2636cc74b0f11b47c2727', 'agent/retire-gfn2-mulliken-population'),
    1659: ('ce0b2a7028a2ee05e5cf0805d93f1754f601f7fd', 'perf/1569-stationary-resident-grid'),
    1660: ('83e434fed994628ff499b46797779c6b7ab3c086', 'chatgpt/issue-246-cosx-pbe0-scf'),
}
TESTS = {
    1651: ['test_dft_mp_v1_capacity.py','test_stationary_cuda_merge_boundary.py','test_direct_shell_derivative_owner.py','test_cuda_rsh_derivative_weights.py','test_pbe0_benchmark.py','test_omol25_benchmark.py','test_evidence_legacy_review.py','test_evidence_retention.py','test_df_evidence_retention.py'],
    1652: ['test_dft_mp_v1_capacity.py','test_stationary_aot_no_compiler.py','test_stationary_composite_cuda_routing.py','test_wb97mv_work_route_evidence.py'],
    1653: ['test_direct_hf_retirement.py','test_cuda_ownership.py','test_cuda_ownership_pairing.py','test_cuda_ownership_publication.py'],
    1654: ['test_dft_mp_v1_capacity.py','test_libxc_snapshot_domain.py','test_public_cpu_item_hamiltonian.py','test_ks_snapshot_provider_proof.py','test_stationary_df_provider_failure.py'],
    1655: ['test_nonlocal_local_scale_codegen.py','test_nonlocal_pair_codegen.py'],
    1657: ['test_gfn2_population_scalar_fma.py','test_gfn2_cuda_provenance.py','test_gfn2_electronic_native_codegen.py','test_gfn2_electronic_population.py','test_gfn2_fused_electronic_codegen.py'],
    1659: ['test_dft_mp_v1_capacity.py','test_stationary_resident_grid.py'],
    1660: [],
}
TITLES = {
    1651:'fix(review): preserve through-f audits and historical evidence',
    1652:'fix(tests): bind composite routing to current AOT and audit contracts',
    1653:'fix(audit): retain native scientific composition in retirement metadata',
    1654:'fix(dft): reject unavailable fitted derivatives instead of exact fallback',
    1655:'fix(codegen): preserve CUDA math and raw local-scale validation order',
    1657:'fix(codegen): preserve population FMA accumulator and source provenance',
    1659:'fix(tests): bind resident-grid execution to capacity mutation guards',
    1660:'fix(build): instantiate COSX KS only with its CUDA provider',
}


def replace(path, old, new, count=1):
    source = path.read_text()
    if old == new:
        return
    if source.count(old) == 0 and new in source:
        return
    if source.count(old) != count:
        raise RuntimeError(f'unexpected source anchor count in {path}: {old[:80]}')
    path.write_text(source.replace(old, new))


def copy_test(root, name):
    source = Path(__file__).parent / 'review_regressions' / name
    target = root / 'tests/python' / name
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError('concurrent regression file exists: ' + name)
    shutil.copyfile(source, target)


def append_once(path, marker, text):
    source = path.read_text()
    if marker not in source:
        path.write_text(source.rstrip() + '\n\n\n' + text.strip() + '\n')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def apply(number, root):
    if number == 1651:
        base = '6ccecb1b41d604018d738f814a6cb44d2e191996'
        paths = git(root, 'diff', '--name-only', '--diff-filter=M', base, HEADS[number][0], '--', '*.patch').decode().splitlines()
        if len(paths) != 45 or any(not p.startswith('benchmarks/results/') for p in paths):
            raise RuntimeError('unexpected historical patch inventory')
        for relative in paths:
            old = git(root, 'show', base + ':' + relative)
            changed = git(root, 'show', HEADS[number][0] + ':' + relative)
            target = root / relative
            if target.read_bytes() not in (old, changed):
                raise RuntimeError('concurrent historical evidence changed: ' + relative)
            target.write_bytes(old)
        replace(root/'.pre-commit-config.yaml', '      - id: trailing-whitespace\n', '      - id: trailing-whitespace\n        # Historical patches retain significant diff-context whitespace.\n        exclude: ^benchmarks/results/.*\\.patch$\n')
        replace(root/'tests/python/test_stationary_cuda_merge_boundary.py', 'lambda _basis: (\n            primitives,', 'lambda _basis, *, integral_derivatives=True: (\n            primitives,')
        path = root/'tests/python/test_cuda_rsh_derivative_weights.py'
        replace(path, 'kernel = source.index("__global__ void independent_rsh_derivative_kernel(")', 'kernel = source.index("__device__ unsigned contract_rsh_quartet(")')
        replace(path, 'source.index("    double j_weight =", kernel)', 'source.index("  double j_weight =", kernel)')
        replace(path, 'source.index("    if (j_weight ==", begin)', 'source.index("  if (j_weight ==", begin)')
        path = root/'tests/python/test_direct_shell_derivative_owner.py'
        replace(path, 'assert "cuda_execution::one_electron_view(shared.batch)" in bridge', 'assert (\n        "cuda_execution::one_electron_view(generated_owner ? exchange->shared->batch : source->batch)"\n        in bridge\n    )')
        replace(path, 'assert "auto* output = exchange->force" in bridge', 'assert "auto* output = generated_owner ? exchange->force : source->derivative" in bridge')
        path = root/'tests/python/test_pbe0_benchmark.py'
        block = '''    figure([point], tmp_path, title="PBE0 / def2-SVP", filename="pbe0.svg")
    svg = (tmp_path / "pbe0.svg").read_text()
    assert "PBE0 / def2-SVP" in svg
    assert "OMol25" not in svg and "through-f" not in svg
'''
        if 'def test_pbe0_plot_has_its_own_method_label' not in path.read_text():
            replace(path, block, '')
            append_once(path, 'def test_pbe0_plot_has_its_own_method_label', '''def test_pbe0_plot_has_its_own_method_label(tmp_path: Path) -> None:
    """Only rendering needs the optional plotting package, not evidence checks."""
    pytest.importorskip("matplotlib")
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-def2-svp-20261001/water3.json"
    )
    point = json.loads(path.read_text())
''' + block)
    elif number == 1652:
        path = root/'tests/python/test_stationary_aot_no_compiler.py'
        replace(path, '        method_ir=resolve_method("PBE"),\n', '        method_ir=resolve_method("PBE"),\n        nonlocal_density_policy=None,\n', 2)
        replace(path, '_stationary_cuda_execution=object(),', '_stationary_cuda_execution=None,', 2)
        replace(root/'tests/python/test_wb97mv_work_route_evidence.py', '''    assert '"full-range"' in radial''', '''    assert any(
        isinstance(item, ast.Constant) and item.value == "full-range"
        for item in ast.walk(fields["two_electron_radial_operators"])
    )''')
    elif number == 1653:
        reasons = {
            'src/scf/cuda/direct_bounded_dddd.cu':'Bounded dddd traversal delegates ERI/force/scatter algebra to generated and shared owners, but retains the initial physical Schwarz acceptance criterion in the kernel. Keep that remaining scientific policy visible; launch/control code is not a second ERI implementation.',
            'src/xtb/native/src/backends/cuda/gfn2_geometry.cu':'Coordination-pair scalar value/derivative mathematics is compiler-generated. Native geometry still owns distance/radius composition and the radial-to-Cartesian coordination VJP, as well as physical cutoffs; these remaining scientific pieces are not retired by the scalar cutover.',
            'src/xtb/native/src/backends/cuda/gfn2_pairlist.cu':'Coordination-pair scalar value/derivative mathematics is compiler-generated. Native pair-list execution still owns distance/radius composition, physical cutoff decisions and radial-to-Cartesian coordination VJP composition; topology and transport surround this remaining scientific work.',
            'src/xtb/native/src/backends/cuda/gfn2_repulsion.cu':'Repulsion-pair scalar energy and radial derivative mathematics is compiler-generated. Native execution still composes element-pair alpha/charge and light-pair policy and maps radial derivatives into Cartesian forces; keep these remaining scientific operations visible.',
        }
        for relative, reason in reasons.items():
            path = root/'docs/cuda_ownership/files'/(relative+'.json')
            data = json.loads(path.read_text()); data['role'] = 'scientific'; data['reason'] = reason
            path.write_text(json.dumps(data, indent=2)+'\n')
        path = root/'docs/cuda_ownership/direct_hf_retirement.json'
        data = json.loads(path.read_text())
        family = next(f for f in data['families'] if f['id']=='bounded-direct-native-exceptions')
        retired = 'src/scf/cuda/direct_bounded_exact_force.cu'
        if retired in family['files']:
            family['files'].remove(retired)
        family['capability'] = 'Initial physical Schwarz acceptance in the bounded dddd scheduler. Its ERI/force/scatter algebra is delegated to existing compiler-generated/shared owners; the separate low-order exact-force page file is runtime-only.'
        family['retirement_condition'] = 'Retire the remaining dddd scientific-policy classification when its initial Schwarz acceptance delegates to the shared scientific screening owner. Compiler-owned ERI/force algebra is not duplicate formula debt; scheduler-only runtime may remain native.'
        path.write_text(json.dumps(data, indent=2)+'\n')
        append_once(root/'tests/python/test_direct_hf_retirement.py', 'def test_retired_exact_force_scheduler_is_absent_from_scientific_overlay', '''def test_retired_exact_force_scheduler_is_absent_from_scientific_overlay() -> None:
    retirement, ownership = _inputs()
    path = "src/scf/cuda/direct_bounded_exact_force.cu"
    roles = {row["path"]: row["role"] for row in ownership["files"]}
    assert roles[path] == "runtime"
    assert all(path not in family["files"] for family in retirement["families"])
    validate_retirement_ledger(ROOT, retirement, ownership)


def test_native_radial_and_screening_composition_is_not_hidden() -> None:
    _, ownership = _inputs()
    roles = {row["path"]: row["role"] for row in ownership["files"]}
    for path in (
        "src/scf/cuda/direct_bounded_dddd.cu",
        "src/xtb/native/src/backends/cuda/gfn2_geometry.cu",
        "src/xtb/native/src/backends/cuda/gfn2_pairlist.cu",
        "src/xtb/native/src/backends/cuda/gfn2_repulsion.cu",
    ):
        assert roles[path] == "scientific"
''')
    elif number == 1654:
        path = root/'python/generativeqc/_stationary_cuda.py'
        old = '        native_complete_integrals = native_integral_components is not None\n'
        if 'Direct derivative fallback would change the Hamiltonian' not in path.read_text():
            replace(path, old, '''        if use_fitted_integrals and native_integral_components is None:
            raise NotImplementedError(
                "density-fitted stationary derivative provider is unavailable; "
                "Direct derivative fallback would change the Hamiltonian"
            )
''' + old)
        path = root/'tests/python/test_libxc_snapshot_domain.py'
        replace(path, '_calculator = SimpleNamespace(_method_name="test-selector")', '_calculator = SimpleNamespace(_method_name="test-selector", _density_fitting_mode=0)')
        replace(path, '"_native": SimpleNamespace(check=check),', '"_native": SimpleNamespace(check=check, DENSITY_FITTING_NONE=0),')
        replace(root/'tests/python/test_public_cpu_item_hamiltonian.py', '        _device_name="cpu",', '        _device_name="cpu",\n        _density_fitting_mode=0,')
        copy_test(root, 'test_stationary_df_provider_failure.py')
    elif number == 1655:
        path = root/'python/generativeqc_compiler/method/nonlocal_pair.py'
        source = path.read_text(); start = source.index('def native_local_scale_cpp()'); stop = source.index('\ndef build_nonlocal_pair_program', start)
        part = source[start:stop].replace('std::sqrt','::sqrt').replace('std::pow','::pow')
        old = '''  if constexpr (Variant == Vv10Variant::rvv10) {
    out.omega /= out.kappa;
    out.kappa *= ::sqrt(out.kappa);
  }
  return out;
}
'''
        new = '''  return out;
}

// The runtime validates raw scales before this representation change. In
// particular, a positive raw kappa may underflow to zero after preconditioning.
template <Vv10Variant Variant>
GENERATIVEQC_NONLOCAL_PAIR_HD inline void precondition_local_scales_cuda(
    double& omega, double& kappa) noexcept {
  if constexpr (Variant == Vv10Variant::rvv10) {
    omega /= kappa;
    kappa *= ::sqrt(kappa);
  }
}
'''
        if part.count(old)!=1:
            raise RuntimeError('local-scale source changed')
        part = part.replace(old,new).replace('preconditions rVV10 omega/kappa for the ordered pair kernel. Runtime callers\n    continue to own finite/domain checks and signed-weight publication policy.', 'emits rVV10 preconditioning separately. Runtime callers validate the raw\n    scales before converting representation, preserving the prior failure domain.')
        path.write_text(source[:start]+part+source[stop:])
        path = root/'src/dft/nonlocal_correlation/vv10_runtime_cuda.cu'
        source = path.read_text(); start = source.index('__global__ void local_scales_kernel'); stop = source.index('\n}\n',start)
        source = source[:stop]+'''\n  generated::precondition_local_scales_cuda<Variant>(omega[i], kappa[i]);
  if constexpr (Variant == Vv10Variant::rvv10) {
    if (!isfinite(omega[i]) || !isfinite(kappa[i])) atomicExch(failed, 1);
  }'''+source[stop:]
        path.write_text(source)
        copy_test(root, 'test_nonlocal_local_scale_codegen.py')
    elif number == 1657:
        path = root/'tools/generate_gfn2_electronic_cuda.py'
        replace(path, '            input_order=("density", "integral", "accumulator"),\n            output_order=("updated",),\n            fused_accumulation=True,', '            input_order=("density", "integral", "accumulator"),\n            output_order=("updated",),\n            fused_accumulation=True,\n            ordered_native_sums=True,')
        path = root/'src/xtb/native/CUDA_SOURCE_PROVENANCE.json'
        data = json.loads(path.read_text()); name='src/backends/cuda/gfn2_mulliken.cu'
        data['files'][name]['vendored_sha256'] = hashlib.sha256((root/'src/xtb/native'/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(data,indent=2)+'\n')
        copy_test(root, 'test_gfn2_population_scalar_fma.py')
    elif number == 1659:
        path = root/'tests/python/test_dft_mp_v1_capacity.py'
        replace(path, 'old = "for begin in range(0, len(grid.points), tile_points):"', 'old = "for begin in range(0, grid_points, tile_points):"')
        replace(path, '"for begin in range(0, len(grid.points), 2 * tile_points):"', '"for begin in range(0, grid_points, 2 * tile_points):"')
    elif number == 1660:
        for spin in ('rks','uks'):
            path = root/f'src/dft/{spin}.cpp'; source=path.read_text(); marker=f'ScfResult run_pbe0_cosx_{spin}('
            start=source.index(marker); stop=source.index('\n}\n',start)+3
            source=source[:start]+'#if GENERATIVEQC_HAS_CUDA\n'+source[start:stop]+'#endif\n'+source[stop:]
            path.write_text(source)
        path=root/'src/dft/cosx_scf.hpp';source=path.read_text();start=source.index('namespace generativeqc::scf {')
        source=source[:start]+'// The prepared COSX provider is linked only in CUDA-enabled builds.\n#if GENERATIVEQC_HAS_CUDA\n'+source[start:]
        stop=source.rindex('\n#endif');source=source[:stop]+'\n#endif  // GENERATIVEQC_HAS_CUDA\n'+source[stop:];path.write_text(source)
    else:
        raise RuntimeError('unreviewed target')


def source_node(root, relative, owner, name):
    source=(root/relative).read_text(); body=ast.parse(source).body
    if owner:
        nodes=[n for n in body if isinstance(n,ast.ClassDef) and n.name==owner]
        if len(nodes)!=1: raise RuntimeError('ambiguous class')
        body=nodes[0].body
    nodes=[n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    if len(nodes)!=1: raise RuntimeError('ambiguous source node')
    return source,nodes[0]


def hash_node(source,node):
    return hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()


def bind(number,root):
    changes={}
    def item(const,expected,relative,owner,name):
        actual=hash_node(*source_node(root,relative,owner,name))
        allowed=expected if isinstance(expected,tuple) else (expected,)
        if actual not in allowed: raise RuntimeError('unreviewed source drift: '+const+' '+actual)
        changes[const]=actual
    stationary='python/generativeqc/_stationary_cuda.py'
    if number==1651:
        for const,expected,owner,name in (
            ('STATIONARY_LAYOUT_CONTRACT_SHA256','fa8c4ff2a644fd45ab4eb828a995c4e42c49c80adbe712b32d50f90b3d98fb74',None,'_layout'),
            ('STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256','34bc6b3c49fdb6661587b46f5aa374e911a2c921b5a6940661f5464347d64cf9','_CudaSources','__init__'),
            ('STATIONARY_PAGE_BULK_CONTRACT_SHA256','b7bc1344bd86447cd6c9efcdfef944bb22c8b92b5ed5327d2028cf787d6a1729','_CudaSources','integral_page'),
            ('STATIONARY_PAGE_COMPONENT_INTEGRAL_CONTRACT_SHA256','d3f61e820c8bcf0df4bf4fce639f342936b79aceb956cb6e9f43caa3d13cdaa3','_CudaSources','integral'),
        ): item(const,expected,stationary,owner,name)
    if number==1652:
        item('PUBLIC_CUDA_FORCE_METHOD_CONTRACT_SHA256','432072ea6ce50303e4e855bc29585fe00dc3b74a3dbee00f4490f15af1b15c3a','python/generativeqc/batch.py','PreparedBatch','_public_dft_cuda_force')
    if number==1654:
        source,ctor=source_node(root,'python/generativeqc/calculator.py','Calculator','__init__')
        nodes=[n for n in ctor.body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=='semilocal_force' for x in ast.walk(n.test))]
        if len(nodes)!=1: raise RuntimeError('ambiguous promotion')
        actual=hash_node(source,nodes[0]); expected='49f903598301e16b11be96d1b24eb084aa7bee3194942702b174b41e59d4b01c'
        if actual!=expected: raise RuntimeError('unreviewed force promotion')
        changes['PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256']=actual
        item('NATIVE_KS_SNAPSHOT_INIT_CONTRACT_SHA256',('0b7d7d0513c4c8f8004653017169cd766ad182310ae401e53174be72bda62f01','a185c806e3c61e65c1bf19ab8cca5e8ee2dd82934226cccefd6535d959cdb6da'),'python/generativeqc/_ks_snapshot.py','NativeKsSnapshot','__init__')
        item('NATIVE_KS_SNAPSHOT_DECODE_CONTRACT_SHA256','ebba6922b0b7e47a25dd76455f0f87c9fdfc053555d5d81ef95903313e9d802c','python/generativeqc/_ks_snapshot.py','NativeKsSnapshot','decode')
        item('STATIONARY_ENDPOINT_OWNER_CONTRACT_SHA256','0cabfc6f7334c84a119eaba2f844f9f4b4fa7fe6a32b7d4c9049d157082ed050',stationary,None,'_complete_rks_cuda_gradient_diagnostic')
    if number==1659:
        item('STATIONARY_ENDPOINT_OWNER_CONTRACT_SHA256','f1546a355c19016fcda39a51e2370696087a9e679f2765093bda2b9d0a2cc1d2',stationary,None,'_complete_rks_cuda_gradient_diagnostic')
    if not changes: return {}
    q=root/'tools/dft_mp_v1/qualify_capacity.py'; t=root/'tests/python/test_dft_mp_v1_capacity.py'
    source=q.read_text(); tests=t.read_text()
    original=git(root,'show',HEADS[number][0]+':tools/dft_mp_v1/qualify_capacity.py').decode()
    def constants(text):
        return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    old=constants(source); base=constants(original); report={}
    for name,new in changes.items():
        if old[name]!=new:
            if source.count(old[name])!=1: raise RuntimeError('ambiguous qualifier fingerprint')
            source=source.replace(old[name],new,1)
        if tests.count(new)!=1:
            candidates=[value for value in {old[name],base[name]} if tests.count(value)==1]
            if len(candidates)!=1: raise RuntimeError('ambiguous expected fingerprint: '+name)
            tests=tests.replace(candidates[0],new,1)
        report[name]={'previous':old[name],'new':new}
    q.write_text(source);t.write_text(tests)
    return report


if __name__=='__main__':
    number=int(sys.argv[1]);root=Path(sys.argv[2]).resolve()
    if len(sys.argv)>3 and sys.argv[3]=='bind':
        print(json.dumps(bind(number,root),indent=2))
    else:
        apply(number,root)
