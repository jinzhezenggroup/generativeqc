from pathlib import Path
import sys
root=Path(sys.argv[1]); stage=int(sys.argv[2])
p=root/'tests/python/test_stationary_cuda_merge_boundary.py'
s=p.read_text(); marker='        "stationary_geometry_drain",'
extra='        "stationary_geometry_molecular_enqueue",\n        "stationary_geometry_external_device_molecular_enqueue",\n'
if stage>=3: extra+='        "stationary_geometry_molecular_resident_weights_enqueue",\n        "stationary_geometry_external_device_molecular_resident_weights_enqueue",\n'
assert '"stationary_geometry_molecular_enqueue"' not in s
p.write_text(s.replace(marker,extra+marker))
p=root/'tests/python/test_stationary_nonlocal_timing.py';s=p.read_text()
if stage>=2:
    s=s.replace('        assert args[1] is None and kwargs["defer_error_to_consumer"]', '        assert args == (8192, 1, None, ("rho", "gradient", "tau"))\n        assert not kwargs')
extra='''    def geometry(task: Any, begin: int, points_per_atom: int,
                 {weight_arg}weights: Any, raw: Any, **kwargs: Any) -> None:
        assert task.view.stream == 31
        assert (begin, points_per_atom) == (0, 1)
        {weight_assert}np.testing.assert_array_equal(weights, np.ones(1))
        np.testing.assert_array_equal(raw, np.ones(1))
        assert kwargs == {{"functional": 4}}

    def nonlocal_geometry(task: Any, begin: int, points_per_atom: int,
                          {weight_arg}weights: Any, raw: Any, pointer: int,
                          stride: int, offset: int) -> None:
        geometry(task, begin, points_per_atom, {weight_pass}weights, raw, functional=4)
        assert (pointer, stride, offset) == (4096, 1, 0)

'''.format(weight_arg='device_weights: int, ' if stage>=3 else '',weight_assert='assert device_weights == 16384\n        ' if stage>=3 else '',weight_pass='device_weights, ' if stage>=3 else '')
s=s.replace('    parts = {',extra+'    parts = {')
if stage>=3:
 s=s.replace('        geometry=lambda *args, **kwargs: None,\n        geometry_external_device=lambda *args: None,','        geometry_molecular_resident_weights=geometry,\n        geometry_external_device_molecular_resident_weights=nonlocal_geometry,\n        natom=1,')
else:
 s=s.replace('        geometry=lambda *args, **kwargs: None,\n        geometry_external_device=lambda *args: None,','        geometry_molecular=geometry,\n        geometry_external_device_molecular=nonlocal_geometry,\n        natom=1,')
if stage>=2:
 s=s.replace('        _source=object(),','        _source=SimpleNamespace(\n            cuda_resident_grid=lambda: SimpleNamespace(\n                device=0, point_count=1, points=8192, weights=16384\n            )\n        ),')
 s=s.replace('        grid=SimpleNamespace(feature_task=feature_task),','        grid=SimpleNamespace(feature_task_device_points=feature_task, device_id=0),')
p.write_text(s)
if stage>=2:
 p=root/'tests/python/test_grid_view_publication.py';s=p.read_text()
 s=s.replace('  double *points = storage, *ao = storage, *features = storage;','  double *points = storage, *ao = storage, *features = storage;\n  const double* current_points = storage + 16;')
 s=s.replace('extern "C" int sections()', '''extern "C" int view_uses_selected_points() {
  generativeqc::dft::GridTaskView view{};
  if (grid_cuda_view_v1(&plan, &view, nullptr, 0)) return -1;
  return view.points == plan.current_points && view.points != plan.points;
}
extern "C" int sections()''')
 s=s.replace('    subprocess.run(\n        [\n            compiler,','    compiled = subprocess.run(\n        [\n            compiler,',1).replace('        check=True,\n        timeout=30,','        check=False,\n        timeout=30,',1)
 s=s.replace('    native = ct.CDLL(str(library))','    assert compiled.returncode == 0, compiled.stdout + compiled.stderr\n    native = ct.CDLL(str(library))')
 s += '''\n\n@pytest.mark.parametrize("deferred", [0, 1])
def test_view_exports_selected_points_not_owned_scratch(publication: ct.CDLL, deferred: int) -> None:
    assert publication.run(deferred, 0, 0, 0) == 0
    assert publication.view_uses_selected_points() == 1
'''
 p.write_text(s)
p=root/'.github/workflows/ci.yml';s=p.read_text()
old='          curl --fail --silent --show-error --location \\\n            --output "/tmp/${archive}" \\\n            "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/${archive}"'
new=old.replace(' --location \\\n',' --location \\\n            --retry 5 --retry-all-errors --retry-delay 2 \\\n            --connect-timeout 20 --max-time 120 --retry-max-time 240 \\\n')
assert s.count(old)>=1, 'ccache download recipe changed'
p.write_text(s.replace(old,new))
