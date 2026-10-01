"""Apply only residual repairs after the read-only concurrent-head review."""
import json
import subprocess


def configure(module):
    prior_apply = module.apply
    module.HEADS.update({
        1651: ('674542ede09cc2b9a254b18f0514f616827cd9cc', 'perf/through-f-dft-readme-20261001'),
        1652: ('5aa36757d2f5855ac5ba116b7c61795461dc06ad', 'refactor/stationary-composite-cuda'),
        1653: ('57c30c42507e3e9eb1843035c7031ce43458d05e', 'agent/retire-stale-scientific-ownership'),
        1657: ('16a6f98eeca67842926936d81d2fa5c82a98175f', 'agent/retire-gfn2-mulliken-population'),
        1661: ('a6fcd114ffbccb690c32b067869178e009dd17a3', 'perf/ri-k-final-projection-handoff'),
    })
    module.TESTS[1661] = ['test_ks_warm_orbital_failure.py']
    module.TITLES[1661] = 'test(dft): extract the complete final orbital publication block'

    def replace(path, old, new):
        source=path.read_text()
        if old not in source and new in source:return
        if source.count(old)!=1:raise RuntimeError('changed residual anchor: '+str(path)+' '+old[:60])
        path.write_text(source.replace(old,new,1))

    def git(root,ref,path):
        return subprocess.check_output(['git','show',ref+':'+path],cwd=root)

    def apply(number,root):
        if number==1651:
            base='6ccecb1b41d604018d738f814a6cb44d2e191996'
            reviewed='a40e2981572a87d601d7b4beebc1008436998f22'
            paths=subprocess.check_output(['git','diff','--name-only','--diff-filter=M',base,reviewed,'--','*.patch'],cwd=root,text=True).splitlines()
            if len(paths)!=45:raise RuntimeError('historical patch inventory changed')
            for relative in paths:
                old=git(root,base,relative); changed=git(root,reviewed,relative); target=root/relative
                if not target.exists() or target.read_bytes() not in (old,changed):
                    raise RuntimeError('concurrent historical patch changed: '+relative)
                target.write_bytes(old)
            path=root/'.pre-commit-config.yaml'
            replace(path,r'exclude: ^benchmarks/results/.*\\.patch$',r'exclude: ^benchmarks/results/.*\.patch$')
            # Restore the removed inventory entry only for an existing historical
            # source.patch whose original bytes were just verified/restored.
            relative='benchmarks/results/bounded-direct-945-20260923/publication.json'
            original=json.loads(git(root,base,relative)); current=json.loads((root/relative).read_text())
            old_entry=next(x for x in original['files'] if x['path']=='source.patch')
            if not any(x['path']=='source.patch' for x in current['files']):
                expected=dict(original);expected['files']=[x for x in original['files'] if x['path']!='source.patch']
                if current!=expected:raise RuntimeError('unreviewed publication metadata change')
                (root/relative).write_bytes(git(root,base,relative))
            replace(root/'tests/python/test_direct_shell_derivative_owner.py',
                    'assert "auto* output = exchange->force" in bridge',
                    'assert "auto* output = generated_owner ? exchange->force : source->derivative" in bridge')
            path=root/'tests/python/test_pbe0_benchmark.py'
            old='''    pytest.importorskip("matplotlib")
    figure([point], tmp_path, title="PBE0 / def2-SVP", filename="pbe0.svg")
    svg = (tmp_path / "pbe0.svg").read_text()
    assert "PBE0 / def2-SVP" in svg
    assert "OMol25" not in svg and "through-f" not in svg
'''
            if 'def test_pbe0_plot_has_its_own_method_label' not in path.read_text():
                replace(path,old,'')
                module.append_once(path,'def test_pbe0_plot_has_its_own_method_label','''def test_pbe0_plot_has_its_own_method_label(tmp_path: Path) -> None:
    """Rendering alone needs matplotlib; timeout/evidence validation always runs."""
    pytest.importorskip("matplotlib")
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/pbe0-def2-svp-20261001/water3.json"
    )
    point = json.loads(path.read_text())
    figure([point], tmp_path, title="PBE0 / def2-SVP", filename="pbe0.svg")
    svg = (tmp_path / "pbe0.svg").read_text()
    assert "PBE0 / def2-SVP" in svg
    assert "OMol25" not in svg and "through-f" not in svg
''')
        elif number==1652:
            path=root/'tests/python/test_stationary_aot_no_compiler.py'
            source=path.read_text()
            if source.count('_stationary_cuda_execution=object(),')!=2:raise RuntimeError('AOT fixture changed')
            path.write_text(source.replace('_stationary_cuda_execution=object(),','_stationary_cuda_execution=None,'))
            path=root/'tests/python/test_wb97mv_work_route_evidence.py'
            replace(path,'    assert "full-range" in radial','''    assert any(
        isinstance(item, ast.Constant) and item.value == "full-range"
        for item in ast.walk(fields["two_electron_radial_operators"])
    )''')
            path=root/'tests/python/test_stationary_composite_cuda_routing.py'
            replace(path,'from generativeqc._stationary_composite_cuda import requires_composite_stationary_cuda',
                'from generativeqc._stationary_composite_cuda import _plan_for_state, requires_composite_stationary_cuda')
            replace(path,'''    assert "source.nonlocal_density_policy == MOLECULAR_VV10_DENSITY_POLICY" in DRIVER
    assert "else SCF_POINT_MODEL" in DRIVER''','''    state = _fake_state("PBE")
    assert _plan_for_state(state).mean_field.point_model == SCF_POINT_MODEL
    del state._source.nonlocal_density_policy
    assert _plan_for_state(state).mean_field.point_model == SCF_POINT_MODEL''')
        elif number==1653:
            path=root/'docs/cuda_ownership/direct_hf_retirement.json'
            current=json.loads(path.read_text())
            if not any(f['id']=='bounded-direct-native-exceptions' for f in current['families']):
                baseline=json.loads(git(root,'3b3fceb41a65d46a522d989cb7f02e54f825156a','docs/cuda_ownership/direct_hf_retirement.json'))
                family=next(f for f in baseline['families'] if f['id']=='bounded-direct-native-exceptions')
                current['families'].insert(1,family)
                path.write_text(json.dumps(current,indent=2)+'\n')
            prior_apply(number,root)
        elif number==1657:
            prior_apply(number,root)
        elif number==1661:
            path=root/'tests/python/test_ks_warm_orbital_failure.py'
            replace(path,'import shutil\n','import re\nimport shutil\n')
            replace(path,'    end = legacy.index("    previous_energy =")',
                    '    end = legacy.rindex("\\n    previous_energy =") + 1')
            replace(path,'''    begin = legacy.rfind("    try {\\n", 0, end)
    assert begin >= 0''','''    publication_blocks = list(re.finditer(r"^    try \\{\\n", legacy[:end], re.MULTILINE))
    assert publication_blocks
    begin = publication_blocks[-1].start()''')
        else:raise RuntimeError('unreviewed residual target')
    module.apply=apply
