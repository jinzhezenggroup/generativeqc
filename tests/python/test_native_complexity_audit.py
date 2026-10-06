"""Regression coverage for the native high-order source audit."""

from pathlib import Path

import pytest

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


def test_audit_does_not_reassociate_nonlinear_pairwise_expression() -> None:
    source = r"""
void nonlinear(std::size_t n, const double* a, const double* b, const double* c, double* out) {
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          out[p * n + q] += std::exp(a[p * n + k]) * b[k * n + l] * c[l * n + q];
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification != "matrix-chain-candidate"


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
            finding.classification == "fixed-extent-inner-loop" for finding in findings
        )
        assert all(finding.effective_depth < finding.depth for finding in findings)


def test_fixed_extent_does_not_hide_remaining_high_order_work() -> None:
    source = r"""
void still_high_order(std::size_t n, const double* data, double* out) {
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d)
          for (unsigned axis = 0; axis < 3; ++axis)
            out[a * n + b] += data[((c * n + d) * 3) + axis];
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].effective_depth == 4
    assert findings[0].classification != "fixed-extent-inner-loop"


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


def test_audit_keeps_high_rank_materialization_visible_through_helper_call() -> None:
    source = r"""
double source_weight(const double* weights, std::size_t n, std::size_t p,
                     std::size_t q, std::size_t r, std::size_t s);

void pullback(std::size_t n, const double* c, const double* weights, double* first) {
  for (std::size_t iu = 0; iu < n; ++iu)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          for (std::size_t p = 0; p < n; ++p)
            first[((iu * n + q) * n + r) * n + s] +=
                c[iu * n + p] * source_weight(weights, n, p, q, r, s);
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "high-rank-output-materialization"
    assert findings[0].lhs == "first[((iu * n + q) * n + r) * n + s]"


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


@pytest.mark.parametrize(
    "body",
    [
        (
            "// out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q];\n"
            "out[p*n+q] += a[((p*n+q)*n+k)*n+l];"
        ),
        ("out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q] / (1.0 + d[p*n+k] + d[l*n+q]);"),
        (
            "const auto nonlinear = std::exp(a[p*n+k]);\n"
            "out[p*n+q] += nonlinear*b[k*n+l]*c[l*n+q];"
        ),
    ],
    ids=["commented-example", "coupled-denominator", "opaque-alias"],
)
def test_ambiguous_algebra_remains_report_only(body: str) -> None:
    source = (
        "void f(size_t n, double* out, double* a, double* b, double* c, double* d) {\n"
        "for (size_t p=0;p<n;++p) for(size_t q=0;q<n;++q) "
        "for(size_t k=0;k<n;++k) for(size_t l=0;l<n;++l) {\n" + body + "\n}}"
    )
    findings = audit_text(source, path="ambiguous.cpp")
    assert len(findings) == 1
    assert findings[0].classification != "matrix-chain-candidate"


def test_audit_preserves_pure_product_alias_detection() -> None:
    source = """
void f(size_t n, double* out, double* a, double* b, double* c) {
  for (size_t p=0;p<n;++p) for(size_t q=0;q<n;++q)
    for(size_t k=0;k<n;++k) for(size_t l=0;l<n;++l) {
      const auto product = a[p*n+k] * b[k*n+l];
      out[p*n+q] += product * c[l*n+q];
    }
}
"""
    assert audit_text(source)[0].classification == "matrix-chain-candidate"


@pytest.mark.parametrize(
    "domain,statement",
    [
        ("0", "if (p + q > k + l) out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q];"),
        ("p + q", "out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q];"),
    ],
)
def test_conditional_or_coupled_domain_is_not_ci_blocking(
    domain: str, statement: str
) -> None:
    source = (
        "void f(size_t n, double* out, double* a, double* b, double* c) {"
        "for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q) "
        + f"for(size_t k={domain};k<n;++k) for(size_t l=0;l<n;++l) {{"
        + statement
        + "}}"
    )
    assert audit_text(source)[0].classification != "matrix-chain-candidate"
