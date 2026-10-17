#ifndef GENERATIVEQC_CLI_NATIVE_BASIS_HPP
#define GENERATIVEQC_CLI_NATIVE_BASIS_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "molecule/basis_catalog.hpp"

namespace generativeqc::cli {

using NativeBasisData = molecule::ExpandedBasis;
using molecule::bundled_basis_names;
using molecule::bundled_basis_supports;
using molecule::expand_bundled_basis;

// Canonical local records retain their declared representation and checksum.
// BSE format conversion remains explicit build-time work, never a runtime lookup.
bool is_local_basis(std::string_view name);
NativeBasisData expand_basis_input(std::string_view name, std::span<const generativeqc_atom> atoms,
                                   std::optional<generativeqc_basis_representation> representation);

}  // namespace generativeqc::cli

#endif
