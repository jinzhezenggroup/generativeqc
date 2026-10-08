"""Small scalar FP64 C++ lowering for compiler-owned runtime kernels.

This backend is intentionally shape-zero-dimensional: it lowers an ordinary
TensorIR Program whose live values are all scalar float64 values into one
inline C++ function. Runtime loops/topology stay outside the generated
scientific expression.
"""

from __future__ import annotations

import re
import typing
from fractions import Fraction
from math import isfinite

from .ir import TRANSCENDENTALS, Node
from .optimize import prepare_for_backend
from .program import Program, _topological

SCALAR_CPP_PRIMITIVES = frozenset(
    {
        "input",
        "constant",
        "add",
        "multiply",
        "divide",
        "scaled_bilinear",
        *TRANSCENDENTALS,
    }
)


def _literal(pair: typing.Any) -> str:
    value = float(Fraction(*pair))
    if not isfinite(value):
        raise ValueError("scalar C++ literal is not finite FP64")
    return value.hex()


def _identifier(name: str, label: str) -> str:
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_]\w*", name) is None:
        raise ValueError(f"{label} must be a C++ identifier")
    return name


def _binary_expression(op: str, left: str, right: str) -> str:
    """Shared native spelling for an existing binary scalar IR operation."""
    operator = {"multiply": "*", "divide": "/", "add": "+", "subtract": "-"}[op]
    return f"{left} {operator} {right}"


def _sqrt_expression(value: str, namespace: str = "std::") -> str:
    return f"{namespace}sqrt({value})"


def native_scalar_read(value: str, *, accessors: tuple[str, ...] = ()) -> str:
    """Admit a closed name/member/index read, optionally a declared accessor.

    Index expressions contain only integer literals or names joined by +, - or
    *. Calls, commas, assignments, nested reads and arbitrary C++ expressions
    are excluded. A method may explicitly bind a known pure one-index accessor;
    its argument must be one identifier, never consumer-provided arithmetic.
    This validates physical spelling, not the implementation of that accessor.
    """
    if not isinstance(accessors, tuple) or len(set(accessors)) != len(accessors):
        raise ValueError("native read accessors must be unique declared names")
    for accessor in accessors:
        _identifier(accessor, "native read accessor")
    identifier = r"[A-Za-z_]\w*"
    name = rf"{identifier}(?:(?:\.|::){identifier})*"
    atom = rf"(?:{name}|(?:0|[1-9][0-9]*)(?:u|U|ul|UL|ull|ULL)?)"
    index = rf"{atom}(?:\s*[+*-]\s*{atom})*"
    pattern = rf"(?P<name>{name})(?:\((?P<argument>{identifier})\))?(?:\[{index}\])*"
    match = re.fullmatch(pattern, value) if isinstance(value, str) else None
    if match is None or (
        match["argument"] is not None and match["name"] not in accessors
    ):
        raise ValueError(
            "native scalar read requires a declared name/member/index/accessor"
        )
    return value


def emit_scalar_cpp_statement(
    program: Program,
    *,
    bindings: typing.Mapping[str, str],
    target: str,
    form: str = "=",
    math_namespace: str = "std::",
    read_accessors: tuple[str, ...] = (),
) -> str:
    """Render one closed FP64 graph as a native assignment without temporaries.

    This is the existing scalar algebra's caller-owned-checks and ordered-native-
    sums execution contract. It deliberately admits only single-output trees of
    input/constant, binary unit signed sums, multiply, divide and sqrt. It neither
    adds checks nor promises unfused arithmetic: the enclosing translation unit
    retains its compiler contraction policy. Compound assignment requires the
    exact first operand to be the target input. Read bindings must be pure C++
    names/member/index reads or explicitly declared one-index accessors, never
    arithmetic supplied by the consumer. Targets cannot be accessor calls.

    It is a spelling/scheduling mode, not a second scalar graph or evaluator.
    Existing emit_scalar_cpp's default helper output is unchanged.
    """
    if not isinstance(program, Program) or len(program.outputs) != 1:
        raise ValueError("scalar statement requires one TensorIR output")
    if form not in ("=", "+=", "-=", "/=") or math_namespace not in ("", "std::"):
        raise ValueError("unsupported native scalar statement spelling")
    program = prepare_for_backend(program, "scalar", preserve_reduction_order=True)
    nodes = program.live_nodes
    if any(n.spec.shape or n.spec.dtype != "float64" for n in nodes):
        raise ValueError("scalar statement requires scalar float64 values")
    inputs = {n.attrs["name"] for n in nodes if n.op == "input"}
    if set(bindings) != inputs:
        raise ValueError("scalar statement bindings must name every input exactly")

    bound = {
        name: native_scalar_read(value, accessors=read_accessors)
        for name, value in bindings.items()
    }
    target = native_scalar_read(target)

    def expression(node: Node) -> tuple[str, int]:
        if node.op == "input":
            return bound[node.attrs["name"]], 4
        if node.op == "constant":
            return _literal(node.attrs["values"][0]), 4
        if node.op == "sqrt":
            return _sqrt_expression(expression(node.inputs[0])[0], math_namespace), 4
        if node.op not in ("add", "multiply", "divide"):
            raise ValueError("unsupported native scalar statement primitive")
        op = node.op
        if op == "add":
            coefficients = node.attrs["coefficients"]
            if len(node.inputs) != 2 or coefficients not in (
                ((1, 1), (1, 1)),
                ((1, 1), (-1, 1)),
            ):
                raise ValueError("native statement requires binary unit signed sum")
            op = "add" if coefficients[1] == (1, 1) else "subtract"
        precedence = 1 if op in ("add", "subtract") else 2
        left, lp = expression(node.inputs[0])
        right, rp = expression(node.inputs[1])
        if lp < precedence:
            left = f"({left})"
        # Equal-precedence right subtrees must retain the original grouping.
        if rp <= precedence:
            right = f"({right})"
        return _binary_expression(op, left, right), precedence

    root = next(iter(program.outputs.values()))
    if form != "=":
        expected = {
            "+=": ("add", ((1, 1), (1, 1))),
            "-=": ("add", ((1, 1), (-1, 1))),
            "/=": ("divide", None),
        }[form]
        if (
            root.op != expected[0]
            or len(root.inputs) != 2
            or (expected[1] is not None and root.attrs["coefficients"] != expected[1])
            or root.inputs[0].op != "input"
            or bound[root.inputs[0].attrs["name"]] != target
        ):
            raise ValueError("compound assignment requires its exact target seed")
        # Validate the complete tree even when the seed is implicit in spelling.
        expression(root)
        code = expression(root.inputs[1])[0]
    else:
        code = expression(root)[0]
    return f"{target} {form} {code};"


def _scalar_constant(node: typing.Any) -> Fraction | None:
    if node.op != "constant":
        return None
    values = node.attrs["values"]
    return Fraction(*values[0]) if len(values) == 1 else None


def _scaled_bilinear_helper(function_name: str) -> str:
    helper = f"{function_name}_scaled_bilinear"
    return f"""inline bool {helper}(
    double a, double b, double c, double d, double e, double f,
    double& out) noexcept {{
  if (e == 0.0 || f == 0.0) return false;
  int ea, eb, ec, ed, ee, ef;
  const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb);
  const double mc = std::frexp(c, &ec), md = std::frexp(d, &ed);
  const double me = std::frexp(e, &ee), mf = std::frexp(f, &ef);
  double p = ma * mb, q = mc * md;
  double pe = std::fma(ma, mb, -p), qe = std::fma(mc, md, -q);
  const int ep = ea + eb, eq = ec + ed;
  const int exponent = p == 0.0 ? eq : (q == 0.0 ? ep : (ep > eq ? ep : eq));
  constexpr int limit = 110;
  const int dp = ep - exponent, dq = eq - exponent;
  if (dp < -limit) {{ p = 0.0; pe = 0.0; }}
  else {{ p = std::scalbn(p, dp); pe = std::scalbn(pe, dp); }}
  if (dq < -limit) {{ q = 0.0; qe = 0.0; }}
  else {{ q = std::scalbn(q, dq); qe = std::scalbn(qe, dq); }}
  const double difference = p - q;
  const double tail = difference - p;
  const double residual = (p - (difference - tail)) - (q + tail);
  const double numerator = difference + ((pe - qe) + residual);
  out = std::scalbn(numerator / (me * mf), exponent - ee - ef);
  return std::isfinite(out);
}}"""


def emit_scalar_cpp(
    program: Program,
    *,
    function_name: str,
    input_order: typing.Iterable[str] | None = None,
    output_order: typing.Iterable[str] | None = None,
    direct_scaled_bilinear: bool = False,
    check_intermediates: bool = True,
    fused_accumulation: bool = False,
    caller_owned_checks: bool = False,
    ordered_native_sums: bool = False,
    output_dependency_order: bool = False,
) -> str:
    """Lower a scalar FP64 Program to one checked inline C++ function.

    direct_scaled_bilinear is an explicit admission for callers that have
    already proved bounded nonzero denominator domains. check_intermediates
    may likewise be disabled only for a bounded domain; inputs and published
    outputs remain finite-checked. Defaults retain the conservative behavior.
    fused_accumulation explicitly contracts single-use product addends with
    coefficient +/-1 into std::fma, retaining the addend order and finite result
    checks. It is opt-in because fused rounding is part of the caller contract.

    caller_owned_checks transfers all runtime finite/domain checks to the caller.
    Exceptional arithmetic and outputs are preserved, including zero times
    infinity and zero divided by zero; the returned bool is always true. This
    mode requires direct_scaled_bilinear for that primitive, whose robust helper
    otherwise owns its own checks. ordered_native_sums starts each sum with its
    first term and spells unit coefficients as identity/unary minus. This is an
    explicit signed-zero lowering contract rather than TensorIR's initial +0
    accumulator for the sums remaining after normal production preparation;
    constant-only folds retain TensorIR semantics. Neither option enables
    reassociation or FMA contraction.

    output_dependency_order schedules definitions depth-first from output_order
    (or sorted output names), visiting dependencies in their existing operand
    order and emitting shared values once. It changes only the emission schedule;
    the default retains the canonical depth/hash order of Program.live_nodes.
    """

    if not isinstance(program, Program):
        raise TypeError("scalar C++ lowering requires a TensorIR Program")
    program = prepare_for_backend(program, "scalar")
    function_name = _identifier(function_name, "function_name")
    nodes = program.live_nodes
    if any(node.spec.shape != () or node.spec.dtype != "float64" for node in nodes):
        raise ValueError("scalar C++ lowering requires scalar float64 values")
    unsupported = sorted({node.op for node in nodes} - SCALAR_CPP_PRIMITIVES)
    if unsupported:
        raise ValueError(f"unsupported scalar C++ primitives: {unsupported}")
    if (
        caller_owned_checks
        and not direct_scaled_bilinear
        and any(node.op == "scaled_bilinear" for node in nodes)
    ):
        raise ValueError("caller_owned_checks requires direct_scaled_bilinear")
    check_runtime = not caller_owned_checks
    check_values = check_runtime and check_intermediates
    simplify_bounded = check_runtime and not check_intermediates

    inputs: dict[str, Node] = {}
    for node in nodes:
        if node.op == "input":
            name = _identifier(node.attrs["name"], "input name")
            inputs.setdefault(name, node)
    ordered_inputs = tuple(sorted(inputs) if input_order is None else input_order)
    if set(ordered_inputs) != set(inputs) or len(ordered_inputs) != len(inputs):
        raise ValueError("input_order must name every scalar input exactly once")

    ordered_outputs = tuple(
        sorted(program.outputs) if output_order is None else output_order
    )
    if set(ordered_outputs) != set(program.outputs) or len(ordered_outputs) != len(
        program.outputs
    ):
        raise ValueError("output_order must name every scalar output exactly once")
    for name in ordered_outputs:
        _identifier(name, "output name")

    if output_dependency_order:
        nodes = _topological(program.outputs[name] for name in ordered_outputs)

    uses: dict[Node, list[tuple[Node, int]]] = {}
    for parent in nodes:
        for slot, child in enumerate(parent.inputs):
            uses.setdefault(child, []).append((parent, slot))
    fused_products: set[Node] = set()
    if fused_accumulation:
        for node in nodes:
            consumers = uses.get(node, [])
            if (
                node.op != "multiply"
                or len(consumers) != 1
                or node in program.outputs.values()
            ):
                continue
            parent, slot = consumers[0]
            if parent.op == "add" and Fraction(*parent.attrs["coefficients"][slot]) in (
                -1,
                1,
            ):
                fused_products.add(node)

    index = {node: position for position, node in enumerate(nodes)}

    def ref(node: Node) -> str:
        return f"v{index[node]}"

    # Keep user-visible TensorIR labels out of the generated local namespace.
    input_parameters = {
        name: f"tensor_input_{i}" for i, name in enumerate(ordered_inputs)
    }
    output_parameters = {
        name: f"tensor_output_{i}" for i, name in enumerate(ordered_outputs)
    }
    parameters = [f"double {input_parameters[name]}" for name in ordered_inputs]
    parameters += [f"double& {output_parameters[name]}" for name in ordered_outputs]
    lines: list[str] = []
    if not direct_scaled_bilinear and any(
        node.op == "scaled_bilinear" for node in nodes
    ):
        lines.append(_scaled_bilinear_helper(function_name))
    lines.append(f"inline bool {function_name}({', '.join(parameters)}) noexcept {{")
    if ordered_inputs and check_runtime:
        condition = " || ".join(
            f"!std::isfinite({input_parameters[name]})" for name in ordered_inputs
        )
        lines.append(f"  if ({condition}) return false;")

    for node in nodes:
        if node in fused_products:
            continue
        name = ref(node)
        attrs = node.attrs
        if node.op == "input":
            lines.append(f"  const double {name} = {input_parameters[attrs['name']]};")
            continue
        if node.op == "constant":
            values = attrs["values"]
            if len(values) != 1:
                raise ValueError("scalar constant must contain exactly one value")
            lines.append(f"  const double {name} = {_literal(values[0])};")
            continue
        if node.op == "add":
            terms = list(zip(node.inputs, attrs["coefficients"], strict=True))
            if simplify_bounded:
                terms = [
                    (child, coefficient)
                    for child, coefficient in terms
                    if Fraction(*coefficient) != 0 and _scalar_constant(child) != 0
                ]
            if not ordered_native_sums or not terms:
                lines.append(f"  double {name} = 0.0;")
            for term_index, (child, coefficient) in enumerate(terms):
                first_native_term = ordered_native_sums and term_index == 0
                factor = Fraction(*coefficient)
                if child in fused_products:
                    left, right = child.inputs
                    sign = "-" if factor == -1 else ""
                    if first_native_term:
                        lines.append(
                            f"  double {name} = {sign}{ref(left)} * {ref(right)};"
                        )
                    else:
                        lines.append(
                            f"  {name} = std::fma({sign}{ref(left)}, {ref(right)}, {name});"
                        )
                    if check_values:
                        lines.append(f"  if (!std::isfinite({name})) return false;")
                elif ordered_native_sums:
                    term = (
                        ref(child)
                        if factor == 1
                        else f"-{ref(child)}"
                        if factor == -1
                        else f"{_literal(coefficient)} * {ref(child)}"
                    )
                    if first_native_term:
                        lines.append(f"  double {name} = {term};")
                    elif factor == -1:
                        lines.append(f"  {name} -= {ref(child)};")
                    else:
                        lines.append(f"  {name} += {term};")
                else:
                    lines.append(f"  {name} += {_literal(coefficient)} * {ref(child)};")
        elif node.op == "multiply":
            left, right = node.inputs
            if simplify_bounded and (
                _scalar_constant(left) == 0 or _scalar_constant(right) == 0
            ):
                lines.append(f"  const double {name} = 0.0;")
            elif simplify_bounded and _scalar_constant(left) == 1:
                lines.append(f"  const double {name} = {ref(right)};")
            elif simplify_bounded and _scalar_constant(right) == 1:
                lines.append(f"  const double {name} = {ref(left)};")
            else:
                expression = _binary_expression("multiply", ref(left), ref(right))
                lines.append(f"  const double {name} = {expression};")
        elif node.op == "divide":
            numerator, denominator_node = node.inputs
            denominator = ref(denominator_node)
            if check_runtime:
                lines.append(f"  if ({denominator} == 0.0) return false;")
            if simplify_bounded and _scalar_constant(numerator) == 0:
                lines.append(f"  const double {name} = 0.0;")
            elif simplify_bounded and _scalar_constant(denominator_node) == 1:
                lines.append(f"  const double {name} = {ref(numerator)};")
            else:
                expression = _binary_expression("divide", ref(numerator), denominator)
                lines.append(f"  const double {name} = {expression};")
        elif node.op == "scaled_bilinear":
            if direct_scaled_bilinear:
                a_node, b_node, c_node, d_node, _e_node, _f_node = node.inputs
                a, b, c, d, e, f = (ref(child) for child in node.inputs)
                if check_runtime:
                    lines.append(f"  if ({e} == 0.0 || {f} == 0.0) return false;")
                left_zero = (
                    _scalar_constant(a_node) == 0 or _scalar_constant(b_node) == 0
                )
                right_zero = (
                    _scalar_constant(c_node) == 0 or _scalar_constant(d_node) == 0
                )
                if simplify_bounded and left_zero and right_zero:
                    expression = "0.0"
                elif simplify_bounded and left_zero:
                    expression = f"-({c} * {d}) / ({e} * {f})"
                elif simplify_bounded and right_zero:
                    expression = f"({a} * {b}) / ({e} * {f})"
                else:
                    expression = f"({a} * {b} - {c} * {d}) / ({e} * {f})"
                lines.append(f"  const double {name} = {expression};")
            else:
                args = ", ".join(ref(child) for child in node.inputs)
                lines.append(f"  double {name} = 0.0;")
                lines.append(
                    f"  if (!{function_name}_scaled_bilinear({args}, {name})) return false;"
                )
        elif node.op in TRANSCENDENTALS:
            child = ref(node.inputs[0])
            if node.op == "sqrt":
                if check_runtime:
                    lines.append(f"  if ({child} < 0.0) return false;")
                expression = _sqrt_expression(child)
            elif node.op == "log":
                if check_runtime:
                    lines.append(f"  if (!({child} > 0.0)) return false;")
                expression = f"std::log({child})"
            elif node.op == "power":
                if check_runtime:
                    lines.append(f"  if (!({child} > 0.0)) return false;")
                expression = f"std::pow({child}, {_literal(attrs['exponent'])})"
            else:
                expression = f"std::exp({child})"
            lines.append(f"  const double {name} = {expression};")
        else:
            raise AssertionError(node.op)
        if check_values:
            lines.append(f"  if (!std::isfinite({name})) return false;")

    if check_runtime:
        for output_name in ordered_outputs:
            value = ref(program.outputs[output_name])
            lines.append(f"  if (!std::isfinite({value})) return false;")
    # The default validates all outputs transactionally; caller-owned checking
    # instead publishes the native exceptional values for the enclosing runtime.
    for output_name in ordered_outputs:
        value = ref(program.outputs[output_name])
        lines.append(f"  {output_parameters[output_name]} = {value};")
    lines.append("  return true;")
    lines.append("}")
    return "\n".join(lines) + "\n"
