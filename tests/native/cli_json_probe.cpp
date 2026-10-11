#include <iostream>
#include <string>

#include "cli/json.hpp"
#include "dft/grid.hpp"

// A host-only protocol probe lets Python independently check canonical hashes.
int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--native-grid") {
    const generativeqc::dft::GridSpec grid;
    std::cout << generativeqc::cli::Json(
                     {grid.radial_points, grid.angular_polar, grid.angular_azimuth})
                     .dump()
              << '\n';
    return 0;
  }
  std::string input;
  while (std::getline(std::cin, input)) {
    const auto value = generativeqc::cli::Json::parse(input);
    std::cout << generativeqc::cli::canonical_json(value) << '\n';
    std::cout << generativeqc::cli::canonical_hash(value) << '\n';
  }
}
