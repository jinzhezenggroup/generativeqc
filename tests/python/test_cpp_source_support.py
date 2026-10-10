"""Regression tests for the host-only C++ contract extractor."""

from __future__ import annotations

import pytest
from _cpp_source_support import (
    cpp_function_declaration,
    cpp_function_definition,
    cpp_if_block,
    cpp_record_definition,
)


def test_definition_ignores_comments_strings_and_raw_strings() -> None:
    source = r"""
// void target(int x) { /* wrong */ }
const char *example = R"delim(void target() { ) }; } )delim";
const char ch = '}';
template <class Value>
inline int target(Value value) noexcept {
  const char *ignored = "} // not end";
  /* } */ if (value > 0) { return 4; }
  return 2;
}
"""
    definition = cpp_function_definition(source, "target", include_template=True)
    assert definition.startswith("template <class Value>")
    assert definition.endswith("  return 2;\n}")
    assert "/* } */" in definition


def test_declaration_adapts_to_added_parameters() -> None:
    header = """
// int execute(...);
std::vector<RhfBucketItem> execute(
    CudaRhfBucketPlan& plan, const std::vector<core::System>& systems,
    const ScfOptions& options);
"""
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


def test_if_block_skips_unbraced_guards_and_string_braces() -> None:
    source = """
if (retain_resident) additional += 10;
if (retain_resident /* guarded lease */) {
  const char* marker = "}";
  if (ready) { ++count; }
}
"""
    block = cpp_if_block(source, "retain_resident")
    assert block.startswith("if (retain_resident /* guarded lease */) {")
    assert block.endswith("\n}")
    with pytest.raises(ValueError, match="found 2"):
        cpp_if_block("if (ready) {} if (ready) {}", "ready")


@pytest.mark.parametrize(
    "call",
    [
        "if (target()) { return; }",
        "if (ready && target()) { return; }",
        "if (ready &&\n target()) { return; }",
        "while (target()) { break; }",
        "target();",
        "return target();",
        "auto value = target();",
        "object.target();",
        "ns::target();",
    ],
)
def test_function_calls_are_not_contracts(call: str) -> None:
    source = "void caller() {\n" + call + "\n}"
    for extract in (cpp_function_definition, cpp_function_declaration):
        with pytest.raises(ValueError, match="found 0"):
            extract(source, "target")
    definition = "bool target() noexcept { return true; }"
    assert cpp_function_definition(definition + "\n" + source, "target") == definition
    assert cpp_function_declaration("bool target() noexcept;\n" + source, "target") == (
        "bool target() noexcept"
    )


def test_function_suffix_does_not_consume_enclosing_expression() -> None:
    # Even a type-looking line prefix cannot turn an enclosing call into a
    # definition by scanning forward until an unrelated opening brace.
    with pytest.raises(ValueError, match="found 0"):
        cpp_function_definition("bool target()) { return true; }", "target")
    assert cpp_function_definition(
        "bool target() noexcept(true) { return true; }", "target"
    ) == ("bool target() noexcept(true) { return true; }")


@pytest.mark.parametrize(
    "use",
    [
        "void f(struct Target* p) { return; }",
        "struct Target* p = [] { return nullptr; }();",
        "struct Target* factory() { return nullptr; }",
        "struct Target object {};",
    ],
)
def test_elaborated_type_uses_are_not_record_definitions(use: str) -> None:
    with pytest.raises(ValueError, match="found 0"):
        cpp_record_definition(use, "Target")
    definition = "struct Target final : public Base<int> { int value; }"
    assert cpp_record_definition(use + "\n" + definition + ";", "Target") == definition
