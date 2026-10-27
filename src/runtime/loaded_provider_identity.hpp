#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>

#include "runtime/loaded_provider_identity_types.hpp"

#if defined(__linux__) && defined(__ELF__)
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#endif

namespace generativeqc::runtime {

/** Parse bounded native-endian ELF notes, never filesystem content. A build-ID
 * identifies a linked build; it is not a checksum of the object's current bytes. */
inline LoadedBuildIdentity parse_loaded_build_id_notes(std::span<const std::byte> notes) {
  LoadedBuildIdentity result;
  if (notes.size() > 65536) return {LoadedIdentityStatus::malformed_notes, {}, {}};
  std::size_t offset = 0;
  while (offset != notes.size()) {
    std::uint32_t header[3]{};
    if (notes.size() - offset < sizeof(header))
      return {LoadedIdentityStatus::malformed_notes, {}, {}};
    std::memcpy(header, notes.data() + offset, sizeof(header));
    offset += sizeof(header);
    const auto padded = [](std::uint32_t value) -> std::uint64_t {
      return (std::uint64_t(value) + 3) & ~std::uint64_t(3);
    };
    const auto name_bytes = padded(header[0]), descriptor_bytes = padded(header[1]);
    if (name_bytes > notes.size() - offset || descriptor_bytes > notes.size() - offset - name_bytes)
      return {LoadedIdentityStatus::malformed_notes, {}, {}};
    const auto* name = notes.data() + offset;
    const auto* descriptor = name + name_bytes;
    if (header[2] == 3 && header[0] == 4 && std::memcmp(name, "GNU\0", 4) == 0) {
      if (result.status == LoadedIdentityStatus::available)
        return {LoadedIdentityStatus::ambiguous_build_id, {}, {}};
      if (header[1] == 0 || header[1] > 64) return {LoadedIdentityStatus::malformed_notes, {}, {}};
      constexpr char digits[] = "0123456789abcdef";
      for (std::size_t index = 0; index != header[1]; ++index) {
        const auto byte = std::to_integer<unsigned>(descriptor[index]);
        result.build_id += digits[byte >> 4];
        result.build_id += digits[byte & 15];
      }
      result.status = LoadedIdentityStatus::available;
    }
    offset += static_cast<std::size_t>(name_bytes + descriptor_bytes);
  }
  return result;
}

/** Pin an already loaded object and authenticate the symbol's actual owning
 * mapping before reading its bounded, mapped PT_NOTE. PLT/interposer/dependency
 * ownership, another loader namespace, absent notes and malformed bounds fail
 * closed. The pin follows the consuming native owner, never an observer scope. */
inline LoadedBuildIdentity loaded_provider_identity(const char* library, const char* symbol_name,
                                                    const void* execution_symbol = nullptr) {
#if defined(__linux__) && defined(__ELF__)
  if (!library || !symbol_name) return {LoadedIdentityStatus::missing_symbol, {}, {}};
  void* handle = dlopen(library, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (!handle) return {LoadedIdentityStatus::not_loaded, {}, {}};
  std::shared_ptr<void> pin(handle, [](void* value) { (void)dlclose(value); });
  void* symbol = dlsym(handle, symbol_name);
  if (!symbol) return {LoadedIdentityStatus::missing_symbol, {}, {}};
  if (execution_symbol && symbol != execution_symbol)
    return {LoadedIdentityStatus::wrong_symbol_owner, {}, {}};
  link_map* map = nullptr;
  Dl_info owner{};
  if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || !map || !dladdr(symbol, &owner) ||
      owner.dli_fbase != reinterpret_cast<void*>(map->l_addr))
    return {LoadedIdentityStatus::wrong_symbol_owner, {}, {}};
  struct Query {
    std::uintptr_t base, symbol;
    unsigned objects{}, matches{};
    LoadedBuildIdentity result;
    std::exception_ptr failure;
  } query{reinterpret_cast<std::uintptr_t>(owner.dli_fbase),
          reinterpret_cast<std::uintptr_t>(symbol),
          0,
          0,
          {}};
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t size, void* opaque) -> int {
        auto& state = *static_cast<Query*>(opaque);
        try {
          if (++state.objects > 512 ||
              size < offsetof(dl_phdr_info, dlpi_phnum) + sizeof(info->dlpi_phnum)) {
            state.result = {LoadedIdentityStatus::malformed_notes, {}, {}};
            return 1;
          }
          if (info->dlpi_addr != state.base) return 0;
          if (++state.matches != 1 || !info->dlpi_phdr || info->dlpi_phnum > 256) {
            state.result = {LoadedIdentityStatus::ambiguous_build_id, {}, {}};
            return 1;
          }
          const auto range = [&](const auto& program, std::uintptr_t& first, std::uintptr_t& last) {
            constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
            if (program.p_vaddr > maximum - state.base) return false;
            first = state.base + program.p_vaddr;
            if (program.p_memsz > maximum - first) return false;
            last = first + program.p_memsz;
            return true;
          };
          bool executable = false;
          for (unsigned index = 0; index != info->dlpi_phnum; ++index) {
            const auto& program = info->dlpi_phdr[index];
            std::uintptr_t first{}, last{};
            if (program.p_type == PT_LOAD && (program.p_flags & PF_X) &&
                range(program, first, last) && state.symbol >= first && state.symbol < last)
              executable = true;
          }
          if (!executable) {
            state.result = {LoadedIdentityStatus::wrong_symbol_owner, {}, {}};
            return 1;
          }
          std::size_t total_notes = 0;
          for (unsigned index = 0; index != info->dlpi_phnum; ++index) {
            const auto& note = info->dlpi_phdr[index];
            if (note.p_type != PT_NOTE) continue;
            std::uintptr_t first{}, last{};
            if (!range(note, first, last) || note.p_memsz > 65536 - total_notes) {
              state.result = {LoadedIdentityStatus::malformed_notes, {}, {}};
              return 1;
            }
            total_notes += note.p_memsz;
            bool readable = false;
            for (unsigned load = 0; load != info->dlpi_phnum; ++load) {
              const auto& program = info->dlpi_phdr[load];
              std::uintptr_t begin{}, end{};
              if (program.p_type == PT_LOAD && (program.p_flags & PF_R) &&
                  range(program, begin, end) && first >= begin && last <= end)
                readable = true;
            }
            if (!readable) {
              state.result = {LoadedIdentityStatus::malformed_notes, {}, {}};
              return 1;
            }
            auto identity = parse_loaded_build_id_notes({reinterpret_cast<const std::byte*>(first),
                                                         static_cast<std::size_t>(note.p_memsz)});
            if (identity.status == LoadedIdentityStatus::missing_build_id) continue;
            if (identity.status != LoadedIdentityStatus::available) {
              state.result = std::move(identity);
              return 1;
            }
            if (state.result.status == LoadedIdentityStatus::available) {
              state.result = {LoadedIdentityStatus::ambiguous_build_id, {}, {}};
              return 1;
            }
            state.result = std::move(identity);
          }
          return 0;
        } catch (...) {
          // Never unwind C++ allocation failures through the loader's C frame.
          state.failure = std::current_exception();
          return 1;
        }
      },
      &query);
  if (query.failure) std::rethrow_exception(query.failure);
  if (query.matches == 0 && query.result.status == LoadedIdentityStatus::missing_build_id)
    return {LoadedIdentityStatus::wrong_symbol_owner, {}, {}};
  if (query.result.status == LoadedIdentityStatus::available)
    query.result.library_pin = std::move(pin);
  return query.result;
#else
  (void)library;
  (void)symbol_name;
  (void)execution_symbol;
  return {LoadedIdentityStatus::unsupported_platform, {}, {}};
#endif
}

}  // namespace generativeqc::runtime
