"""Regression guards for one-gate stationary nuclear batching."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER = (ROOT / "python/generativeqc_compiler/method/stationary_cuda.py").read_text()
RUNTIME = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
PYTHON = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
WB97MV = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()


def test_generated_single_and_batched_routes_share_one_pair_helper() -> None:
    assert "__device__ bool nuclear_pair(" in COMPILER
    assert "__global__ void nuclear_kernel(" in COMPILER
    assert "__global__ void nuclear_all_kernel(" in COMPILER
    single = COMPILER[
        COMPILER.index("__global__ void nuclear_kernel(") : COMPILER.index(
            "__global__ void nuclear_all_kernel("
        )
    ]
    batch = COMPILER[
        COMPILER.index("__global__ void nuclear_all_kernel(") : COMPILER.index(
            "__global__ void validate_centers("
        )
    ]
    assert "nuclear_pair(" in single
    assert "for (size_t a = 0; a < na; ++a)" in batch
    assert "for (size_t b = 0; b < a; ++b)" in batch
    assert "nuclear_pair(" in batch


def test_native_nuclear_batch_has_one_upload_launch_and_final_gate() -> None:
    begin = RUNTIME.index("int stationary_nuclear_all(")
    end = RUNTIME.index("int stationary_geometry_external(", begin)
    body = RUNTIME[begin:end]
    assert body.count("upload(*p, p->scratch, charges") == 1
    assert body.count("nuclear_all_kernel<<<1, 1") == 1
    assert body.count("finished(*p, stream)") == 1
    assert "p->count_primitive_work(p->atoms * (p->atoms - 1) / 2)" in body


def test_python_batch_preserves_legacy_single_pair_api() -> None:
    assert "def nuclear(self," in PYTHON
    assert "def nuclear_all(self," in PYTHON
    begin = PYTHON.index("    def nuclear_all(self,")
    end = PYTHON.index("    def geometry(", begin)
    body = PYTHON[begin:end]
    assert '"stationary_nuclear_all"' in body
    assert "_checked(charges, (self.natom,))" in body


def test_wb97mv_uses_one_nuclear_batch_submission() -> None:
    assert "self.sources.nuclear_all(charges)" in WB97MV
    assert '"stationary_nuclear"' not in WB97MV
    assert "for other in range(atom)" not in WB97MV
