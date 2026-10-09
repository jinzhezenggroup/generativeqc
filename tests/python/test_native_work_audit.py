"""Conservative source-work audit regression fixtures (no native build needed)."""

import json
from pathlib import Path

import pytest

from tools.audit_native_work import audit_native, fingerprint, main


def rules(source: str, **kwargs: int) -> list[str]:
    return [finding["rule_id"] for finding in audit_native(source, **kwargs)]


def test_sized_vector_and_cuda_allocation_in_loop() -> None:
    source = """
void replay(int n) {
  for (int tile = 0; tile < n; ++tile) {
    std::vector<double> values(n);
    cudaMalloc(&pointer, bytes);
  }
}
"""
    result = audit_native(source)
    assert {item["rule_id"] for item in result} == {
        "loop-host-allocation",
        "loop-device-allocation",
    }
    assert all(
        item["details"]["count_kind"] == "static-site-not-runtime-count"
        for item in result
    )
    assert result[0]["details"]["requested_elements"] == "n"


def test_no_allocation_claim_for_default_zero_stack_or_prepared_capacity() -> None:
    source = """
void replay(int n) {
  std::vector<double> prepared;
  prepared.reserve(n);
  for (int tile = 0; tile < n; ++tile) {
    std::vector<double> empty;
    std::vector<double> zero(0);
    std::array<double, 64> stack{};
    prepared.resize(n);
  }
}
"""
    assert not audit_native(source)


def test_fresh_vector_growth_inside_loop_but_repeated_capacity_unknown() -> None:
    source = """
void replay(int n) {
  for (int tile = 0; tile < n; ++tile) {
    std::vector<double> values;
    values.reserve(n);
    values.resize(n);
  }
}
"""
    result = audit_native(source)
    assert len(result) == 1
    assert "reserve" in result[0]["evidence"][0]


def test_same_file_callee_allocation_has_cross_call_witness() -> None:
    source = """
std::vector<double> producer(int n) {
  std::vector<double> result(n);
  return result;
}
std::vector<double> wrapper(int n) { return producer(n); }
void replay(int n) {
  for (int tile = 0; tile < n; ++tile) consume(wrapper(n));
}
"""
    result = audit_native(source)
    assert len(result) == 1
    finding = result[0]
    assert finding["rule_id"] == "loop-callee-host-allocation"
    assert finding["function"] == "replay"
    assert finding["details"]["allocation_site"]["function"] == "producer"
    assert [edge["function"] for edge in finding["details"]["call_path"]] == [
        "wrapper",
        "producer",
    ]
    assert not audit_native(source, max_call_depth=1)


def test_ambiguous_overload_and_member_dispatch_not_resolved() -> None:
    source = """
void allocate(int n) { std::vector<double> data(n); }
void allocate(double n) { std::vector<double> data(5); }
void other(int n) { std::vector<double> data(n); }
void replay(int n) {
  for (int i = 0; i < n; ++i) { allocate(n); object.other(n); }
}
"""
    assert not audit_native(source)


def test_fresh_local_aggregate_member_allocation_cross_call() -> None:
    source = """
struct Scratch { unsigned dim{}; std::vector<double> data; };
Scratch build(int n) {
  Scratch scratch;
  scratch.dim = n;
  scratch.data.assign(n*n, 0.0);
  return scratch;
}
void replay(int n) { for (int i = 0; i < n; ++i) consume(build(n)); }
"""
    result = audit_native(source)
    assert len(result) == 1
    assert "scratch.data.assign(n*n, 0.0)" in result[0]["evidence"][0]


def test_setup_boundaries_not_hot_but_loop_transfers_and_sync_are_inventory() -> None:
    source = """
void replay(int n) {
  cudaMalloc(&pointer, bytes);
  cudaMemcpy(pointer, data, bytes, cudaMemcpyHostToDevice);
  for (int i = 0; i < n; ++i) {
    cudaMemcpyAsync(data, pointer, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
  cudaDeviceSynchronize();
}
"""
    assert rules(source) == ["loop-transfer", "loop-synchronization"]


def test_recursive_call_graph_is_bounded() -> None:
    source = """
void inner(int n) { std::vector<double> data(n); inner(n-1); }
void replay(int n) { for (int i = 0; i < n; ++i) inner(n); }
"""
    assert len(audit_native(source)) == 1
    with pytest.raises(ValueError):
        audit_native(source, max_call_depth=9)


def test_scalar_call_reuse_is_not_proved_without_cpp_binding_analysis() -> None:
    source = "double producer(double x){return std::exp(x);} void f(int n,const double x){for(int i=0;i<n;++i) consume(producer(x));}"
    assert not audit_native(source)


@pytest.mark.parametrize(
    "expression", ["tile + source", "data[source]", "helper(source)"]
)
def test_dependent_opaque_or_pointer_producers_not_proved(expression: str) -> None:
    source = f"""
double producer(double x) {{ return std::exp(x); }}
void replay(int tiles, int sources) {{
  for (int tile = 0; tile < tiles; ++tile)
    for (int source = 0; source < sources; ++source) {{
      const double argument = {expression};
      consume(producer(argument));
    }}
}}
"""
    assert not audit_native(source)


@pytest.mark.parametrize(
    "body",
    [
        "return std::exp(x) + external;",
        "return unknown(x);",
        "counter += 1; return x;",
    ],
)
def test_external_state_or_unproven_purity_not_reusable(body: str) -> None:
    source = f"""
double producer(double x) {{ {body} }}
void replay(int n, double x) {{ for (int i = 0; i < n; ++i) consume(producer(x)); }}
"""
    assert not audit_native(source)


def test_mutating_input_and_conditional_producer_not_hoist_candidates() -> None:
    for statement in (
        "x += 1; consume(producer(x));",
        "if (i > 2) consume(producer(x));",
    ):
        source = f"""
double producer(double x) {{ return std::exp(x); }}
void replay(int n, double x) {{ for (int i = 0; i < n; ++i) {{ {statement} }} }}
"""
        assert not audit_native(source)


def test_comments_literals_and_line_motion() -> None:
    source = """
void replay(int n) {
  // for (int i = 0; i < n; ++i) malloc(n);
  const char* message = "for (int i = 0; i < n; ++i) malloc(n);";
  for (int i = 0; i < n; ++i) std::vector<double> scratch(n);
}
"""
    first = audit_native(source, "source.cpp")
    second = audit_native("\n\n" + source, "source.cpp")
    assert len(first) == len(second) == 1
    assert fingerprint(first[0]) == fingerprint(second[0])
    assert first[0]["line"] + 2 == second[0]["line"]


def test_json_receipt_and_advisory_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "case.cpp"
    source.write_text("void f(int n) { for(int i=0;i<n;++i) cudaDeviceSynchronize(); }")
    assert main(["--root", str(tmp_path), "--path", "src", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["advisory_only"] is True
    assert report["counts"] == {"loop-synchronization": 1}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report))
    source.write_text("void f() { cudaDeviceSynchronize(); }")
    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--path",
                "src",
                "--format",
                "json",
                "--compare",
                str(baseline),
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)["comparison"]
    assert len(comparison["removed"]) == 1
    assert not comparison["added"]


@pytest.mark.parametrize(
    "declaration",
    [
        "namespace A { void allocate(int n) { std::vector<double> v(n); } } void allocate(double);",
        "struct A { static void allocate(int n) { std::vector<double> v(n); } }; void allocate(double);",
        "void allocate(int n) { std::vector<double> v(n); } void allocate(double);",
    ],
)
def test_unresolved_namespace_class_and_declared_overloads(declaration: str) -> None:
    assert not audit_native(
        declaration + "void replay(int n) {for(int i=0;i<n;++i) allocate(1.0);}"
    )


def test_callable_shadowing() -> None:
    assert not audit_native(
        "void allocate(int n){std::vector<double> v(n);} void f(int n, void (*allocate)(int)){for(int i=0;i<n;++i) allocate(n);}"
    )
    assert not audit_native(
        "void f(int n){auto cudaDeviceSynchronize=[](){}; for(int i=0;i<n;++i) cudaDeviceSynchronize();}"
    )


@pytest.mark.parametrize(
    "mutation",
    ["++x;", "mutate(&x);", "auto& y=x; y+=1;", "x <<= 1;", "x |= 1;", "x ^= 1;"],
)
def test_mutated_scalar_arguments_fail_closed(mutation: str) -> None:
    source = (
        "double producer(double x){return std::exp(x);} void f(int n,int x){for(int i=0;i<n;++i){"
        + mutation
        + " consume(producer(x));}}"
    )
    assert not audit_native(source)


def test_outer_dependent_source_domain_is_not_replay() -> None:
    source = "double producer(double x){return std::exp(x);} void f(int n){for(int tile=0;tile<n;++tile) for(int source=10*tile;source<10*tile+10;++source) consume(producer(source));}"
    assert not audit_native(source)


@pytest.mark.parametrize(
    "body",
    [
        "std::vector<double> v(std::move(prepared));",
        "std::vector<double> v; v=std::move(prepared); v.resize(n);",
        "std::vector<double> v; fill(v); v.resize(n);",
        "std::vector<double> v(prepared.begin(),prepared.end());",
        "std::vector<double> v; v.assign(prepared.begin(),prepared.end());",
    ],
)
def test_move_escape_and_iterator_overloads_not_sized_allocation(body: str) -> None:
    assert not audit_native("void f(int n){for(int i=0;i<n;++i){" + body + "}}")


def test_loop_external_capacity_not_per_iteration_allocation() -> None:
    assert not audit_native(
        "void f(int n){std::vector<double> v; for(int i=0;i<n;++i) v.resize(n);}"
    )


def test_identical_producer_loops_are_candidates_without_execution_proof() -> None:
    block = "for(int p=0;p<n;++p){for(int q=0;q<n;++q){double value=0; for(int r=0;r<n;++r) value+=a[p*n+r]*b[r*n+q]; out[p*n+q]=value;}}"
    result = audit_native("void f(int n){" + block + "consume();" + block + "}")
    assert len(result) == 1
    assert result[0]["rule_id"] == "repeated-producer-loop"
    assert result[0]["details"]["executed_to_logical_ratio"] is None


def test_distinct_increment_token_streams_not_identical_blocks() -> None:
    first = "for(int i=0;i<n;++i) for(int j=0;j<n;++j) out[i] += left[i] + ++right[j];"
    second = "for(int i=0;i<n;++i) for(int j=0;j<n;++j) out[i] += left[i]++ + right[j];"
    assert not audit_native("void f(int n){" + first + second + "}")


def test_same_line_sites_preserved_with_distinct_receipt_fingerprints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "case.cpp").write_text(
        "void f(int n){for(int i=0;i<n;++i){cudaDeviceSynchronize(); cudaDeviceSynchronize();}}"
    )
    main(["--root", str(tmp_path), "--path", "src", "--format", "json"])
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert len(findings) == len({f["fingerprint"] for f in findings}) == 2


@pytest.mark.parametrize(
    "mutation", ["mutate((x));", "obj.mutate(x);", "obj.mutate((x));"]
)
def test_parenthesized_and_member_reference_escapes(mutation: str) -> None:
    source = (
        "double producer(double x){return std::exp(x);} void f(int n,double x){for(int i=0;i<n;++i){"
        + mutation
        + " consume(producer(x));}}"
    )
    assert not audit_native(source)


def test_multiple_vector_declarators_are_distinct_allocation_sites() -> None:
    source = (
        "void f(int n){for(int i=0;i<n;++i){std::vector<double> first(n), second(n);}}"
    )
    result = audit_native(source)
    assert len(result) == 2
    assert result[0]["column"] != result[1]["column"]
    assert result[0]["evidence"][0] != result[1]["evidence"][0]


def test_qualified_overload_prototype_and_inherited_method_are_unresolved() -> None:
    assert not audit_native(
        "void allocate(int n){std::vector<double> v(n);} void allocate(double) noexcept; void f(int n){for(int i=0;i<n;++i) allocate(1.0);}"
    )
    assert not audit_native(
        "struct A : B {void allocate(int n){std::vector<double> v(n);}}; void f(int n){for(int i=0;i<n;++i) allocate(n);}"
    )


@pytest.mark.parametrize(
    "body",
    [
        "std::vector<double> v; obj.prepare(v); v.resize(n);",
        "auto first=prepared.begin(); auto last=prepared.end(); std::vector<double> v(first,last);",
        "std::vector<double> v(allocator);",
    ],
)
def test_more_unresolved_vector_overloads_and_escapes(body: str) -> None:
    assert not audit_native("void f(int n){for(int i=0;i<n;++i){" + body + "}}")


def test_static_and_assigned_aggregate_members_not_fresh() -> None:
    assert not audit_native(
        "struct A {static\nstd::vector<double> v;}; void f(int n){for(int i=0;i<n;++i){A a; a.v.resize(n);}}"
    )
    assert not audit_native(
        "struct A {std::vector<double> v;}; void f(int n){for(int i=0;i<n;++i){A a; a.v=std::move(prepared); a.v.resize(n);}}"
    )


def test_loop_header_mutation_and_shadowed_alias_not_invariant() -> None:
    assert not audit_native(
        "double producer(double x){return std::exp(x);} void f(int n,double x){for(int i=0;i<n;++i,++x) consume(producer(x));}"
    )
    assert not audit_native(
        "double producer(double x){return std::exp(x);} void f(int n,double x){{const double x=0;} for(int i=0;i<n;++i){x+=1;consume(producer(x));}}"
    )


def test_callee_same_line_allocations_are_separate_sites() -> None:
    result = audit_native(
        "void alloc(int n){malloc(n); malloc(n);} void f(int n){for(int i=0;i<n;++i) alloc(n);}"
    )
    assert len(result) == 2
    assert len({item["details"]["allocation_site"]["column"] for item in result}) == 2


def test_ambiguous_aggregate_type_not_resolved() -> None:
    source = "namespace A {struct Scratch{std::vector<double> data;};} struct Prepared{void resize(int);}; struct Scratch{Prepared data;}; void f(int n){for(int i=0;i<n;++i){Scratch s; s.data.resize(n);}}"
    assert not audit_native(source)


def test_hoisted_workspace_before_after_regression() -> None:
    before = "void f(int n){for(int i=0;i<n;++i){std::vector<double> work(n); consume(work);}}"
    after = "void f(int n){std::vector<double> work(n); for(int i=0;i<n;++i) consume(work);}"
    assert len(audit_native(before)) == 1
    assert not audit_native(after)


def test_repeated_source_fused_schedule_before_after_regression() -> None:
    block = "for(int p=0;p<n;++p){for(int q=0;q<n;++q){double value=0; for(int r=0;r<n;++r) value+=a[p*n+r]*b[r*n+q]; out[p*n+q]=value;}}"
    before = (
        "void f(int n){" + block + "first_consumer();" + block + "second_consumer();}"
    )
    after = "void f(int n){" + block + "first_consumer();second_consumer();}"
    assert rules(before) == ["repeated-producer-loop"]
    assert not audit_native(after)


def test_python_fingerprint_preserves_line_motion_and_detects_support_change() -> None:
    from tools.work_audit_python import audit_python

    source = "import numpy as np\ndef build(n, o):\n    result = np.zeros((n, n))\n    result[:o, :o] = 1\n    return result\n"
    first = audit_python(source, "example.py")[0]
    shifted = audit_python("\n\n" + source, "example.py")[0]
    changed = audit_python(source.replace(":o, :o", ":o, :"), "example.py")[0]
    assert fingerprint(first) == fingerprint(shifted)
    assert fingerprint(first) != fingerprint(changed)


@pytest.mark.parametrize(
    "body",
    [
        "std::vector<double> v; auto& w{v}; w.reserve(n); v.resize(n);",
        "std::vector<double> v; {Prepared v; v.resize(n);}",
    ],
)
def test_brace_alias_and_vector_shadow_invalidate_freshness(body: str) -> None:
    assert not audit_native("void f(int n){for(int i=0;i<n;++i){" + body + "}}")


def test_class_and_struct_name_collision_invalidates_freshness() -> None:
    source = "namespace A {struct Scratch{std::vector<double> data;};} class Scratch {public: Prepared data;}; void f(int n){for(int i=0;i<n;++i){Scratch s; s.data.resize(n);}}"
    assert not audit_native(source)
