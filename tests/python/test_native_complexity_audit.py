"""Regression coverage for the native high-order source audit."""

from pathlib import Path

from tools.audit_native_complexity import (
    DEFAULT_EXCLUDES,
    audit_text,
    audit_tree,
)

ROOT = Path(__file__).resolve().parents[2]


def test_audit_detects_quartic_rank2_matrix_chain() -> None:
    source = r"""
void transform(std::size_t n, const double* c, const double* a, double* out) {
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          out[p * n + q] += c[mu * n + p] * a[mu * n + nu] * c[nu * n + q];
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "matrix-chain-candidate"
    assert findings[0].lhs == "out[p * n + q]"


def test_audit_does_not_call_true_four_index_source_reducible() -> None:
    source = r"""
void contract(std::size_t n, const double* eri, const double* d, double* fock) {
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const auto quartet = ((p * n + q) * n + r) * n + s;
          fock[p * n + q] += eri[quartet] * d[r * n + s];
        }
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "high-rank-source-contraction"


def test_audit_does_not_promote_scalar_or_vector_accumulators() -> None:
    sources = (
        r"""
void xc(std::size_t n, const double* phi, const double* jets, double* out) {
  for (std::size_t point = 0; point < n; ++point)
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          for (unsigned axis = 0; axis < 3; ++axis) {
            double value = 0.0;
            value += phi[mu] * phi[nu] * jets[axis * n + mu] * jets[axis * n + nu];
          }
}
""",
        r"""
void primitive_sum(std::size_t n, const double* first, const double* second, double* gradient) {
  for (std::size_t pa = 0; pa < n; ++pa)
    for (std::size_t pb = 0; pb < n; ++pb)
      for (unsigned ti = 0; ti < 4; ++ti)
        for (unsigned tj = 0; tj < 4; ++tj)
          for (unsigned axis = 0; axis < 3; ++axis)
            gradient[axis] += first[pa * 4 + ti] * second[pb * 4 + tj];
}
""",
    )
    for source in sources:
        findings = audit_text(source, path="synthetic.cpp")
        assert findings
        assert all(
            finding.classification == "fixed-extent-inner-loop"
            for finding in findings
        )
        assert all(finding.effective_depth < finding.depth for finding in findings)


def test_audit_surfaces_high_rank_materialization_pass() -> None:
    source = r"""
void symmetrize(std::size_t o, std::size_t v, double* target, const double* source) {
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t b = 0; b < v; ++b) {
          const auto first = ((i * v + a) * o + j) * v + b;
          const auto second = ((j * v + b) * o + i) * v + a;
          target[first] += 0.5 * (source[first] + source[second]);
        }
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "high-rank-output-materialization"
    assert findings[0].effective_depth == 4
    assert "fusion" in findings[0].recommendation


def test_production_has_no_avoidable_rank2_quartic_scalar_reduction() -> None:
    scanned, findings = audit_tree(ROOT / "src", excludes=DEFAULT_EXCLUDES)
    assert scanned > 0
    candidates = [
        finding
        for finding in findings
        if finding.classification == "matrix-chain-candidate"
    ]
    assert not candidates, (
        "production source reintroduced avoidable high-order rank-2 contractions: "
        + ", ".join(f"{item.path}:{item.line}" for item in candidates)
    )
