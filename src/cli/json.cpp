#include "cli/json.hpp"

#include <openssl/evp.h>

#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>

#include <cerrno>
#endif

namespace generativeqc::cli {
namespace {

template <typename Input>
Json parse_metadata(Input&& input) {
  std::vector<std::set<std::string>> keys;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
    if (depth > 256) throw std::runtime_error("JSON nesting exceeds metadata limit");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key && !keys.back().insert(parsed.get<std::string>()).second)
      throw std::runtime_error("duplicate JSON key: " + parsed.get<std::string>());
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  return Json::parse(std::forward<Input>(input), callback);
}

class Digest {
 public:
  Digest() : context_(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
    if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("cannot initialize SHA256");
  }
  void update(const char* bytes, std::size_t size) {
    if (EVP_DigestUpdate(context_.get(), bytes, size) != 1)
      throw std::runtime_error("cannot update SHA256");
  }
  std::string finish() {
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned count = 0;
    if (EVP_DigestFinal_ex(context_.get(), bytes.data(), &count) != 1)
      throw std::runtime_error("cannot finalize SHA256");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned index = 0; index < count; ++index) {
      result += hex[bytes[index] >> 4];
      result += hex[bytes[index] & 15];
    }
    return result;
  }

 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};

std::string python_float(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("non-finite JSON number");
  if (value == 0) return std::signbit(value) ? "-0.0" : "0.0";
  std::array<char, 64> buffer{};
  const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                       std::chars_format::scientific);
  if (converted.ec != std::errc{}) throw std::runtime_error("cannot encode JSON number");
  std::string text(buffer.data(), converted.ptr);
  const auto separator = text.find('e');
  const int exponent = std::stoi(text.substr(separator + 1));
  // Python repr uses fixed notation for decimal exponents in [-4, 15].
  if (exponent < -4 || exponent >= 16) return text;
  const bool negative = text.front() == '-';
  std::string digits = text.substr(negative ? 1 : 0, separator - (negative ? 1 : 0));
  const auto point = digits.find('.');
  if (point != std::string::npos) digits.erase(point, 1);
  const int position = exponent + 1;
  if (position <= 0)
    digits = "0." + std::string(-position, '0') + digits;
  else if (position >= static_cast<int>(digits.size()))
    digits += std::string(position - digits.size(), '0') + ".0";
  else
    digits.insert(static_cast<std::size_t>(position), ".");
  return (negative ? "-" : "") + digits;
}

}  // namespace

std::string canonical_json(const Json& value) {
  if (value.is_number_float()) return python_float(value.get<double>());
  if (!value.is_structured()) return value.dump(-1, ' ', true);
  std::string result = value.is_array() ? "[" : "{";
  bool first = true;
  for (auto entry = value.begin(); entry != value.end(); ++entry) {
    if (!first) result += ',';
    first = false;
    if (value.is_object()) result += Json(entry.key()).dump(-1, ' ', true) + ':';
    result += canonical_json(entry.value());
  }
  return result + (value.is_array() ? "]" : "}");
}

std::string canonical_hash(const Json& value) {
  Digest digest;
  const auto encoded = canonical_json(value);
  digest.update(encoded.data(), encoded.size());
  return digest.finish();
}

std::string file_hash(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)))
    throw std::runtime_error("profile artifact is not a regular file: " + path.string());
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open artifact: " + path.string());
  Digest digest;
  std::array<char, 65536> buffer{};
  while (stream.read(buffer.data(), buffer.size()) || stream.gcount())
    digest.update(buffer.data(), static_cast<std::size_t>(stream.gcount()));
  if (!stream.eof()) throw std::runtime_error("cannot read artifact: " + path.string());
  return digest.finish();
}

bool is_digest(std::string_view value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == value.npos;
}

Json parse_json(std::string_view text) {
  if (text.size() > 64 * 1024 * 1024) throw std::runtime_error("JSON metadata exceeds size limit");
  return parse_metadata(std::string(text));
}

Json read_json(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)))
    throw std::runtime_error("JSON input is not a regular file: " + path.string());
  // Bound metadata independently of the potentially multi-GiB native library.
  if (std::filesystem::file_size(path) > 64 * 1024 * 1024)
    throw std::runtime_error("JSON metadata exceeds size limit");
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open JSON: " + path.string());
  return parse_metadata(stream);
}

void atomic_json(const std::filesystem::path& path, const Json& value) {
#if defined(__unix__) || defined(__APPLE__)
  std::string pattern = path.string() + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const int descriptor = ::mkstemp(name.data());
  if (descriptor < 0) throw std::runtime_error("cannot create temporary JSON");
  const std::filesystem::path temporary(name.data());
  try {
    const auto content = canonical_json(value) + '\n';
    std::size_t position = 0;
    while (position < content.size()) {
      const auto written =
          ::write(descriptor, content.data() + position, content.size() - position);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) throw std::runtime_error("cannot write temporary JSON");
      position += static_cast<std::size_t>(written);
    }
    if (::fsync(descriptor)) throw std::runtime_error("cannot sync temporary JSON");
  } catch (...) {
    ::close(descriptor);
    std::filesystem::remove(temporary);
    throw;
  }
  if (::close(descriptor)) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("cannot close temporary JSON");
  }
  try {
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::filesystem::remove(temporary);
    throw;
  }
#else
  (void)path;
  (void)value;
  throw std::runtime_error("atomic profile publication requires POSIX");
#endif
}

}  // namespace generativeqc::cli
