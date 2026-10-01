"""Exact reviewed follow-up repairs for the native audit and fused geometry PRs."""
from pathlib import Path
import re
import sys


def edit(root, path, before, after):
    file = root / path
    text = file.read_text()
    if text.count(before) != 1:
        raise RuntimeError(f"unexpected before-image in {path}: {before[:60]}")
    file.write_text(text.replace(before, after, 1))


def scanner(root):
    path = 'tools/audit_native_complexity.py'
    product = '''def _pure_product(
    expression: str,
    aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Admit only products; coupled denominators and opaque aliases stay reports."""
    if _CALL.search(expression):
        return False
    # Arithmetic inside one tensor index does not change the product algebra.
    scalar = _ACCESS.sub("value", expression)
    if re.fullmatch(r"[\\w\\s.*()]+", scalar) is None:
        return False
    for name in set(re.findall(r"\\b[A-Za-z_]\\w*\\b", scalar)):
        if name in aliases:
            if name in seen or not _pure_product(aliases[name], aliases, seen | {name}):
                return False
    return True


'''
    edit(root, path, 'def _matrix_chain_lhs(\n', product + 'def _matrix_chain_lhs(\n')
    edit(root, path, '        if _CALL.search(rhs):\n            continue',
         '        if not _pure_product(rhs, aliases):\n            continue')
    before = '    aliases = {match.group(1): match.group(2) for match in _ALIAS.finditer(context)}\n    for reduction in _REDUCTION.finditer(body):\n'
    after = '''    # Conditional/coupled iteration domains are not an ordinary matrix chain.
    if re.search(r"\\b(if|while|switch|do|break|continue|return|goto)\\b", context):
        return None
    for loop in _FOR.finditer(context):
        opening = context.find("(", loop.start())
        closing = _matching(context, opening, "(", ")")
        header = context[opening + 1 : closing]
        variable = _LOOP_VAR.search(header)
        if closing < 0 or variable is None:
            return None
        if any(
            name != variable.group(1) and re.search(rf"\\b{re.escape(name)}\\b", header)
            for name in loop_variables
        ):
            return None
''' + before
    edit(root, path, before, after)
    edit(root, path, '    _, loops = _loop_spans(text)', '    clean, loops = _loop_spans(text)')
    edit(root, path, 'body = text[inner.body_start : inner.body_end]', 'body = clean[inner.body_start : inner.body_end]')
    edit(root, path, 'context = text[chain[-minimum_depth].start : inner.body_end]', 'context = clean[chain[-minimum_depth].start : inner.body_end]')
    tests = root / 'tests/python/test_native_complexity_audit.py'
    text = tests.read_text().replace('from pathlib import Path', 'from pathlib import Path\n\nimport pytest', 1)
    text += '''\n\n@pytest.mark.parametrize(
    "body",
    [
        "// out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q];\\n"
        "out[p*n+q] += a[((p*n+q)*n+k)*n+l];",
        "out[p*n+q] += a[p*n+k]*b[k*n+l]*c[l*n+q] / "
        "(1.0 + d[p*n+k] + d[l*n+q]);",
        "const auto nonlinear = std::exp(a[p*n+k]);\\n"
        "out[p*n+q] += nonlinear*b[k*n+l]*c[l*n+q];",
    ],
    ids=["commented-example", "coupled-denominator", "opaque-alias"],
)
def test_ambiguous_algebra_remains_report_only(body: str) -> None:
    source = (
        "void f(size_t n, double* out, double* a, double* b, double* c, double* d) {\\n"
        "for (size_t p=0;p<n;++p) for(size_t q=0;q<n;++q) "
        "for(size_t k=0;k<n;++k) for(size_t l=0;l<n;++l) {\\n"
        + body
        + "\\n}}"
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
'''
    tests.write_text(text)


def geometry(root):
    conflict = re.compile(r'^<<<<<<< .*?\n(.*?)^\|\|\|\|\|\| .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n', re.M | re.S)
    for path in ('tools/dft_mp_v1/qualify_capacity.py', 'tests/python/test_dft_mp_v1_capacity.py'):
        file = root / path
        text = file.read_text()
        matches = list(conflict.finditer(text))
        if len(matches) != 1 or 'finish_span' not in matches[0][3].lower():
            raise RuntimeError('unexpected capacity-contract merge conflict')
        match = matches[0]
        # Retain master's deliberately narrowed finish-span guard, not the
        # obsolete whole-header guard. The audited geometry guards remain.
        text = text[:match.start()] + match[3] + text[match.end():]
        file.write_text(text)
        for before, after in (
            ('2874742b4a3cf4910edf6a2b38a294dea9aa0fbd72ae5c8785969328436f9a4d', '658dcb1c0dc16e46f01f99f9902bb92a68fcb25c7969a9f1db600506c666b810'),
            ('830d8f4a28806ca4b378f0cd55bc16d91cf5695fc82f311e4ef38336978d721e', '75da921d9ccd76d29b2623b29f7f57a4c939112b1b3c48a024e6020caec5d52f'),
        ):
            edit(root, path, before, after)
    edit(root, 'python/generativeqc_compiler/method/stationary_cuda.py', 'owners ? owner\n', 'owners ? owners[p]\n')
    edit(root, 'tests/python/test_stationary_owner_kernel_abi.py', 'len(arguments(definition)), 16', 'len(arguments(definition)), 17')
    path = 'tests/python/test_stationary_geometry_kernel_host.py'
    edit(root, path, '#include <algorithm>', '#include <algorithm>\n#include <atomic>\n#include <barrier>\n#include <string>\n#include <thread>')
    edit(root, path, 'struct { size_t x{}; } threadIdx;', 'thread_local struct { size_t x{}; } threadIdx;')
    edit(root, path, 'constexpr size_t workers = 32;', 'constexpr size_t workers = 32;\nstd::barrier geometry_barrier(workers);\nvoid __syncthreads() { geometry_barrier.arrive_and_wait(); }')
    edit(root, path, 'int atomicExch(int* p, int v) { const int old=*p; *p=v; return old; }', 'int atomicExch(int* p, int v) { return std::atomic_ref<int>(*p).exchange(v); }')
    edit(root, path, '  *error=1; return fallback;', '  atomicExch(error, 1); return fallback;')
    edit(root, path, 'int main(int argc,char**) {', 'int main(int argc,char** argv) {')
    edit(root, path, '  const bool external=argc==2 || argc==4, arbitrary=argc>=3;', '''  bool external=false, arbitrary=false;
  std::string fault;
  for(int a=1;a<argc;++a) {
    const std::string option=argv[a];
    external |= option=="external";
    arbitrary |= option=="arbitrary";
    if(option=="bad-weight" || option=="producer-error") fault=option;
  }''')
    edit(root, path, '        int error=0,producer_error=0;', '''        int error=0,producer_error=fault=="producer-error" ? 1 : 0;
        if(fault=="bad-weight") weights.back()=NAN;
        const auto unpublished=result;''')
    edit(root, path, '''        for(size_t lane=0;lane<workers;++lane) {
          threadIdx.x=lane;
          kernel(view,work.data(),ao_atoms,implicit?nullptr:owners.data(),
                 implicit?begin:0,implicit?ppa:0,centers,na,weights.data(),raw.data(),
                 external?seeds.data():nullptr,total+7,begin+3,
                 partial.data(),scratch.data(),&error);
        }
        if(error) return 1;
        for(size_t lane=0;lane<workers;++lane)
          for(size_t j=0;j<9*na;++j) result[j]+=partial[lane*9*na+j];''', '''        std::vector<std::thread> lanes;
        for(size_t lane=0;lane<workers;++lane) {
          lanes.emplace_back([&,lane] {
            threadIdx.x=lane;
            kernel(view,work.data(),ao_atoms,implicit?nullptr:owners.data(),
                   implicit?begin:0,implicit?ppa:0,centers,na,weights.data(),raw.data(),
                   external?seeds.data():nullptr,total+7,begin+3,
                   partial.data(),scratch.data(),result.data(),&error);
          });
        }
        for(auto& lane : lanes) lane.join();
        if(!fault.empty()) {
          if(!error || result!=unpublished) return 4;
        } else if(error) return 1;''')
    edit(root, path, '      if(reference.empty()) reference=result;', '      if(!fault.empty()) continue;\n      if(reference.empty()) reference=result;')
    edit(root, path, '    end = source.index("__global__ void geometry_reduce(", begin)', '    end = source.index("}  // namespace generativeqc_stationary_cuda", begin)')
    edit(root, path, '            "-std=c++17",', '            "-std=c++20",\n            "-pthread",')
    with (root / path).open('a') as stream:
        stream.write('''\n\n@pytest.mark.parametrize("fault", ["bad-weight", "producer-error"])
@pytest.mark.parametrize("external", [False, True])
def test_fused_geometry_failure_reaches_barrier_without_publication(
    geometry_probe: Path, fault: str, external: bool
) -> None:
    result = subprocess.run(
        [str(geometry_probe), fault, *(["external"] if external else [])],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
''')


if __name__ == '__main__':
    operation = {1623: scanner, 1510: geometry}[int(sys.argv[1])]
    operation(Path(sys.argv[2]).resolve())
