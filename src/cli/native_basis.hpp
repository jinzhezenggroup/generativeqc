#ifndef GENERATIVEQC_CLI_NATIVE_BASIS_HPP
#define GENERATIVEQC_CLI_NATIVE_BASIS_HPP

#include <span>
#include <string_view>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::cli {

struct NativeBasisData {
  std::vector<generativeqc_primitive> primitives;
  std::vector<generativeqc_shell> shells;
  generativeqc_basis_representation representation{GENERATIVEQC_BASIS_CARTESIAN};
};

std::vector<std::string_view> bundled_basis_names();

bool bundled_basis_supports(std::string_view name, int atomic_number);

NativeBasisData expand_bundled_basis(std::string_view name,
                                     std::span<const generativeqc_atom> atoms,
                                     generativeqc_basis_representation representation);

}  // namespace generativeqc::cli

#endif
