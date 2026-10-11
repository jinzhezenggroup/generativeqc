#ifndef GENERATIVEQC_CLI_JSON_HPP
#define GENERATIVEQC_CLI_JSON_HPP

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace generativeqc::cli {
using Json = nlohmann::json;

// Match Python's sorted, compact, ensure_ascii JSON encoding used by bundles.
std::string canonical_json(const Json& value);
std::string canonical_hash(const Json& value);
std::string file_hash(const std::filesystem::path& path);
bool is_digest(std::string_view value);
// Embedded metadata follows the same duplicate-key and nesting rules as files.
Json parse_json(std::string_view text);
Json read_json(const std::filesystem::path& path);
// Publish via an exclusive sibling temporary file and atomic rename, not truncation.
void atomic_json(const std::filesystem::path& path, const Json& value);

}  // namespace generativeqc::cli
#endif
