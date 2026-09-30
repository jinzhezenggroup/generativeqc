from pathlib import Path
import sys

root = Path(sys.argv[1])
kind = sys.argv[2]
if kind == '1510':
    path = root / 'tests/python/test_stationary_cuda_lowering.py'
    text = path.read_text()
    old = "    geometry=s.split('__global__ void geometry_kernel',1)[1].split(\n        '}  // namespace vibeqc_stationary_cuda',1\n    )[0]"
    new = "    geometry, closing, _ = s.split('__global__ void geometry_kernel',1)[1].partition(\n        '}  // namespace generativeqc_stationary_cuda'\n    )\n    assert closing, 'generated stationary geometry namespace boundary is missing'"
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
else:
    path = root / 'python/generativeqc/_stationary_nonlocal_cuda.py'
    text = path.read_text()
    old = '    return components, seconds, work\n'
    new = '''    # Detailed profiling deliberately uses the retained explicit-owner / host
    # weight route. Each geometry consumer visits the complete grid exactly once,
    # including the two-pass feature fallback; count transfers per consumer, not
    # per collocation pass. Points and nonlocal seeds remain resident either way.
    profiled = sum(
        bool(getattr(owner, "profile_device", False))
        for owner in (sources, nonlocal_sources)
    )
    if profiled:
        work["grid_owner_source"] = (
            "profile-host-explicit-owners"
            if profiled == 2
            else "mixed-implicit-and-profile-host-owners"
        )
        work["grid_weight_source"] = (
            "profile-host-partition-weights"
            if profiled == 2
            else "mixed-resident-and-profile-host-weights"
        )
        work["grid_owner_h2d_bytes"] = profiled * count * 8
        work["grid_weight_h2d_bytes"] = profiled * count * 8
        if "grid_atomic_measure_source" in work:
            work["grid_atomic_measure_source"] = (
                "profile-host-atomic-measures"
                if profiled == 2
                else "mixed-resident-and-profile-host-atomic-measures"
            )
            work["grid_atomic_measure_h2d_bytes"] = profiled * count * 8
    return components, seconds, work
'''
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    test = root / 'tests/python/test_stationary_resident_nonlocal_join.py'
    text = test.read_text()
    text += '''

@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("local_profile", [False, True])
@pytest.mark.parametrize("nonlocal_profile", [False, True])
def test_profile_fallback_reports_actual_grid_uploads(
    fallback: bool, local_profile: bool, nonlocal_profile: bool
) -> None:
    args, _events = fallback_fixture() if fallback else fixture()
    args["sources"].profile_device = local_profile
    args["nonlocal_sources"].profile_device = nonlocal_profile
    grid_lease = args["state"]._source.cuda_resident_grid()
    atomic_resident = hasattr(grid_lease, "atomic_weights")
    if atomic_resident:
        # The existing fixture validates the resident-pointer fast path. Also
        # validate the host raw measure supplied to either profiling consumer,
        # then let that fixture check every pointer/offset and source mapping.
        def wrap(original: typing.Any) -> typing.Any:
            def invoke(*values: typing.Any, **keywords: typing.Any) -> None:
                host_raw = values[6]
                if local_profile or nonlocal_profile:
                    np.testing.assert_array_equal(host_raw, np.ones(len(values[4])))
                else:
                    assert host_raw is None
                original(*values[:6], None, *values[7:], **keywords)

            return invoke

        for owner, name in (
            (args["sources"], "geometry_molecular_resident_weights"),
            (
                args["nonlocal_sources"],
                "geometry_external_device_molecular_resident_weights",
            ),
        ):
            setattr(owner, name, wrap(getattr(owner, name)))
    _parts, _seconds, work = MODULE.resident_nonlocal_geometry(**args)
    profiled = int(local_profile) + int(nonlocal_profile)
    expected = profiled * grid_lease.point_count * 8
    assert work["grid_owner_h2d_bytes"] == expected
    assert work["grid_weight_h2d_bytes"] == expected
    assert work["grid_point_h2d_bytes"] == 0
    assert work["nonlocal_seed_h2d_bytes"] == 0
    assert work["ao_collocation_point_visits"] == (2 if fallback else 1) * 6
    assert ("profile-host" in work["grid_owner_source"]) == bool(profiled)
    assert ("profile-host" in work["grid_weight_source"]) == bool(profiled)
    if atomic_resident:
        assert work["grid_atomic_measure_h2d_bytes"] == expected
        assert ("profile-host" in work["grid_atomic_measure_source"]) == bool(profiled)
'''
    test.write_text(text)
