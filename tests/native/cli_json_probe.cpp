#include <iostream>
#include <string>

#include "cli/json.hpp"

// A host-only protocol probe lets Python independently check canonical hashes.
int main() {
  std::string input;
  while (std::getline(std::cin, input)) {
    const auto value = generativeqc::cli::Json::parse(input);
    std::cout << generativeqc::cli::canonical_json(value) << '\n';
    std::cout << generativeqc::cli::canonical_hash(value) << '\n';
  }
}
