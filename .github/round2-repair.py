"""Apply the inspected test/API repairs to explicitly selected PR checkouts."""
from pathlib import Path
import sys


def edit(root, path, old, new, count=1):
    target = root / path
    text = target.read_text()
    if text.count(old) != count:
        raise RuntimeError(f"unexpected before-image in {path}: {text.count(old)}")
    target.write_text(text.replace(old, new))


def apply(number, root):
    if number == 1633:
        edit(root, 'tests/python/test_dft_mp_v1_capacity.py',
             '        "force_capability_promotion_sha256": (',
             '        "global_hybrid_force_predicate_sha256": (\n'
             '            "dee0b5dfd30d6ddefcf12e7f62e0b6d570e111fb084e385f2ec00cfa200ca8fd"\n'
             '        ),\n        "force_capability_promotion_sha256": (')
    elif number == 1634:
        edit(root, 'tests/python/test_xc_matrix_schedule.py',
             '    assert "spins*work_jets), block" in source',
             '    assert "spins*work_jets);" in source\n'
             '    assert "const dim3 block(16,16);" in source')
    elif number == 1635:
        edit(root, 'tests/python/test_ks_warm_orbital_failure.py',
             '  std::uint64_t final_generation = 0, generation = 7;\n'
             '  struct { bool converged = false; double energy = 0.0; } output;',
             '  std::uint64_t final_generation = 0, generation = 7, solve_epoch = 11;\n'
             '  struct {\n    bool converged = false;\n    double energy = 0.0;\n'
             '    struct {\n      std::uint64_t returned_solve_epoch = 0, returned_state_generation = 0;\n'
             '    } precision_work;\n  } output;')
        edit(root, 'tests/python/test_ks_warm_orbital_failure.py',
             '    assert(owner.final_state_ready && owner.final_generation == owner.generation);',
             '    assert(owner.final_state_ready && owner.final_generation == owner.generation);\n'
             '    assert(owner.output.precision_work.returned_solve_epoch == owner.solve_epoch);\n'
             '    assert(owner.output.precision_work.returned_state_generation == owner.final_generation);')
    elif number == 1637:
        edit(root, 'tests/python/test_stationary_cuda_d_shell.py',
             '    owner.aos = aos', '    owner.integral_derivatives = True\n    owner.aos = aos')
        edit(root, 'tests/python/test_stationary_cuda_merge_boundary.py',
             '        "_layout",\n        lambda _basis: (',
             '        "_layout",\n        lambda _basis, *, integral_derivatives=True: (', count=2)
        # Fixed before/after hashes of the four inspected AST source blocks.
        # No automatic acceptance of arbitrary source drift is performed here.
        fingerprints = (
            ('2f1bb49d43cbfd93e65f69c769ec26c9d04b84bfe5e4be2d705b1262a386b030', 'fa8c4ff2a644fd45ab4eb828a995c4e42c49c80adbe712b32d50f90b3d98fb74'),
            ('20b479949538cd216c5d914aae2787a44b9f5c36def2e284129aff8a46464a4b', '959d14895ac3a65c14c10ebb5d1c8a6be6b5a5965e2496b0b13d22bb01953188'),
            ('c0eb9658bb707083073c9ea57b5021825691c35c0cc2ff6e2afa326c09159dc3', 'd3f61e820c8bcf0df4bf4fce639f342936b79aceb956cb6e9f43caa3d13cdaa3'),
            ('d5f2d214d89a6c714edc52d81b6909e14b5c1b962234c6892c91c9d076e9d63b', 'b7bc1344bd86447cd6c9efcdfef944bb22c8b92b5ed5327d2028cf787d6a1729'),
        )
        for path in ('tools/dft_mp_v1/qualify_capacity.py', 'tests/python/test_dft_mp_v1_capacity.py'):
            for before, after in fingerprints:
                edit(root, path, before, after)
        with (root / 'tests/python/test_stationary_cuda_d_shell.py').open('a') as stream:
            stream.write('''\n\n@pytest.mark.parametrize("method", ["integral", "integral_page"])
def test_geometry_only_owner_rejects_integral_tasks_before_packing(method: str) -> None:
    from generativeqc._stationary_cuda import _CudaSources

    owner = object.__new__(_CudaSources)
    owner.integral_derivatives = False
    with pytest.raises(NotImplementedError, match="geometry-only"):
        getattr(owner, method)(0, "overlap", ())
''')
    elif number in (1640, 1641):
        edit(root, 'tests/python/test_posthf_cuda_transform_validation.py',
             '''    assert batch_add.count("cudaMemcpyAsync(p.raw, values") == 1
    assert "for (auto& state : p.states)" in batch_add
    assert "state.prefix_leader[k] != request" in batch_add
    assert "state.prefix_leader[k - 1]" in batch_add
    assert batch_add.count("ctx.section(true, ctx.metrics.input_ms") == 1
    assert batch_add.count("ctx.section(true, ctx.metrics.library_ms") == 1''',
             '''    device_add = _function_body(
        source, "posthf_cuda_batch_add_device_v1", "posthf_cuda_batch_add_v1"
    )
    shared_begin = source.index("void accumulate_batch_device(")
    shared = source[shared_begin : source.index("}  // namespace", shared_begin)]

    assert batch_add.count("cudaMemcpyAsync(p.raw, values") == 1
    assert "cudaMemcpyAsync" not in device_add
    assert "for (auto& state : p.states)" in shared
    assert "state.prefix_leader[k] != request" in shared
    assert "state.prefix_leader[k - 1]" in shared
    assert batch_add.count("ctx.section(true, ctx.metrics.input_ms") == 1
    assert shared.count("ctx.section(true, ctx.metrics.library_ms") == 1
    for entrypoint in (batch_add, device_add):
        assert entrypoint.count("accumulate_batch_device(p, begin, shape, elements)") == 1
        assert entrypoint.index("batch_tile(p, begin, counts)") < entrypoint.index(
            "accumulate_batch_device(p, begin, shape, elements)"
        )''')
    elif number == 1642:
        edit(root, 'python/generativeqc/rks_hessian_integrals.py',
             '"second-integral budget must be a positive int64"',
             '"integral_budget_bytes (second-integral budget) must be a positive int64"')
    else:
        raise RuntimeError('unapproved repair target')


if __name__ == '__main__':
    apply(int(sys.argv[1]), Path(sys.argv[2]).resolve())
