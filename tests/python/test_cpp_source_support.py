"""Regression tests for the host-only C++ contract extractor."""

from __future__ import annotations

import pytest

from _cpp_source_support import (
    cpp_function_declaration,
    cpp_function_definition,
    cpp_record_definition,
)


def test_definition_ignores_comments_strings_and_raw_strings() -> None:
    source = r'''
// void target(int x) { /* wrong */ }
const char *example = R"delim(void target() { ) }; } )delim";
const char ch = '}';
template <class Value>
inline int target(Value value) noexcept {
  const char *ignored = "} // not end";
  /* } */ if (value > 0) { return 4; }
  return 2;
}
'''
    definition = cpp_function_definition(source, "target", include_template=True)
    assert definition.startswith("template <class Value>")
    assert definition.endswith("  return 2;\n}")
    assert "/* } */" in definition


def test_declaration_adapts_to_added_parameters() -> None:
    header = '''
// int execute(...);
std::vector<RhfBucketItem> execute(
    CudaRhfBucketPlan& plan, const std::vector<core::System>& systems,
    const ScfOptions& options);
'''
    signature = cpp_function_declaration(header, "execute")
    assert signature.startswith("std::vector<RhfBucketItem> execute(")
    assert "const std::vector<core::System>& systems" in signature
    assert not signature.endswith(";")


def test_record_uses_definition_not_forward_declaration() -> None:
    source = 'struct Lease;\nstruct Lease { char x; const char* text = "}"; };'
    assert cpp_record_definition(source, "Lease").startswith("struct Lease {")
    assert cpp_record_definition(source, "Lease").endswith("}")


def test_duplicate_or_missing_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="found 2"):
        cpp_function_definition("void f() {}\nvoid f(int x) {}", "f")
    with pytest.raises(ValueError, match="found 0"):
        cpp_function_declaration("void f() {}", "absent")


def test_missing_template_and_unbalanced_body_are_errors() -> None:
    with pytest.raises(ValueError, match="missing template"):
        cpp_function_definition("void f() {}", "f", include_template=True)
    with pytest.raises(ValueError, match="unclosed C\\+\\+ '\\{'"):
        cpp_function_definition("void f() {", "f")
