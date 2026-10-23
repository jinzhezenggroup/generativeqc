// Traversal only: opcodes, scalar rules and Boys chain rules are compiler-owned.
// Included inside integral_ad after generated scalar helpers. Scratch is lent
// by the enclosing contraction schedule, with two nodes per tape instruction.
// Keep instruction counts as runtime data. Specializing this traversal per
// primitive graph causes CUDA to optimize hundreds of copies of the same
// interpreter. In particular, p-shell ERIs must not expand it at every call.
#ifdef GQC_INTEGRAL_AD_CUDA
#define GQC_AD_TAPE_FN __device__ __noinline__
#else
#define GQC_AD_TAPE_FN static
#endif
template <unsigned BoysOrder, class Scalar>
GQC_AD_TAPE_FN void evaluate_tape(const TapeInstruction* instructions, unsigned nodes,
                                  unsigned inputs, unsigned root, const Scalar* x, Scalar* out,
                                  bool derivative, Scalar* scratch) {
  Scalar* values = scratch;
  Scalar* adjoints = scratch + nodes;
  double boys[BoysOrder + 1];
  bool boys_ready = false;
  for (unsigned i = 0; i < nodes; ++i) {
    const auto& op = instructions[i];
    if (op.opcode == 0) {
      values[i] = op.parameter;
      continue;
    }
    if (op.opcode == 1) {
      values[i] = x[op.left];
      continue;
    }
    if (op.opcode == 2) {
      if (!boys_ready) {
        boys_values<BoysOrder>(primal_value(values[op.right]), boys);
        boys_ready = true;
      }
      Scalar result[2];
      tape_boys(values[op.right], boys, op.left, result);
      values[i] = result[0];
      continue;
    }
    Scalar inputs[2] = {values[op.left], values[op.right]}, result[3];
    tape_rule(op.opcode, inputs, result);
    values[i] = result[0];
  }
  out[0] = values[root];
  if (!derivative) return;
  for (unsigned i = 0; i < inputs; ++i) out[i + 1] = 0;
  for (unsigned i = 0; i < nodes; ++i) adjoints[i] = 0;
  adjoints[root] = 1;
  for (unsigned i = nodes; i-- > 0;) {
    const auto& op = instructions[i];
    if (op.opcode == 0) continue;
    if (op.opcode == 1) {
      out[op.left + 1] += adjoints[i];
      continue;
    }
    if (op.opcode == 2) {
      Scalar result[2];
      tape_boys(values[op.right], boys, op.left, result);
      adjoints[op.right] += adjoints[i] * result[1];
      continue;
    }
    Scalar inputs[2] = {values[op.left], values[op.right]}, result[3];
    tape_rule(op.opcode, inputs, result);
    adjoints[op.left] += adjoints[i] * result[1];
    adjoints[op.right] += adjoints[i] * result[2];
  }
}
#undef GQC_AD_TAPE_FN
