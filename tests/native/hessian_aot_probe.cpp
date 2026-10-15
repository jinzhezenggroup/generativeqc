#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/hessian_aot.hpp"

namespace {
namespace aot = generativeqc::dft::hessian_aot;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};

template <class Value>
struct Device {
  Value* value{};
  std::size_t count;

  explicit Device(std::size_t size) : count(size) {
    if (size) check(cudaMalloc(reinterpret_cast<void**>(&value), size * sizeof(Value)));
  }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  void upload(const std::vector<Value>& input, cudaStream_t stream) {
    if (input.size() != count) throw std::runtime_error("probe upload shape mismatch");
    if (count)
      check(cudaMemcpyAsync(value, input.data(), count * sizeof(Value), cudaMemcpyHostToDevice,
                            stream));
  }
};

std::size_t size_argument(const char* text) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed);
  if (consumed != std::string(text).size() || !value || value > (1ULL << 20))
    throw std::runtime_error("probe size must be in [1,2^20]");
  return value;
}

std::vector<double> read_values(std::size_t count) {
  if (count > (1ULL << 20)) throw std::runtime_error("probe panel exceeds its explicit limit");
  std::vector<double> values(count);
  for (auto& value : values)
    if (!(std::cin >> value) || !std::isfinite(value))
      throw std::runtime_error("probe input must be complete and finite");
  return values;
}

void publish(const Device<double>& output, const Device<int>& status, cudaStream_t stream) {
  int error{};
  std::vector<double> values(output.count);
  check(cudaMemcpyAsync(&error, status.value, sizeof(error), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  if (error) throw std::runtime_error("native AOT leaf failed; no output published");
  check(cudaMemcpyAsync(values.data(), output.value, values.size() * sizeof(double),
                        cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("native AOT leaf failed; no output published");
  std::cout << std::setprecision(17) << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << "]\n";
}

void list_programs() {
  std::cout << '[';
  bool first = true;
  for (const auto& program : aot::programs()) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"name\":" << std::quoted(std::string(program.name))
              << ",\"identity\":" << std::quoted(std::string(program.identity))
              << ",\"kind\":" << static_cast<int>(program.kind)
              << ",\"outputs\":" << program.outputs << ",\"centers\":" << program.centers
              << ",\"stride\":" << program.record_stride << ",\"components\":[";
    for (std::size_t index = 0; index < program.components.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << program.components[index];
    }
    std::cout << "],\"inputs\":[";
    for (std::size_t index = 0; index < program.inputs.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << std::quoted(std::string(program.inputs[index]));
    }
    std::cout << "]}";
  }
  std::cout << "]\n";
}

void contract_checks() {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (aot::find("absent") ||
      aot::launch_array("absent", nullptr, 1, nullptr, nullptr, nullptr) != cudaErrorNotSupported ||
      aot::launch_integral("absent", nullptr, 1, nullptr, nullptr, nullptr, nullptr, nullptr) !=
          cudaErrorNotSupported)
    throw std::runtime_error("missing AOT leaf must fail without compilation/fallback");
  for (const auto& program : aot::programs()) {
    if (program.kind != aot::Kind::array) continue;
    if (aot::launch_array(program.name, nullptr, 0, nullptr, nullptr, nullptr) != cudaSuccess ||
        aot::launch_array(program.name, nullptr, maximum, nullptr, nullptr, nullptr) !=
            cudaErrorInvalidValue ||
        aot::launch_array(program.name, nullptr, 1, nullptr, nullptr, nullptr) !=
            cudaErrorInvalidValue)
      throw std::runtime_error("array bounds/null validation failed");
  }
  if (aot::find("ao") && (aot::launch_ao(nullptr, 1, 1, maximum, nullptr, 1, 20, nullptr, nullptr,
                                         nullptr) != cudaErrorInvalidValue ||
                          aot::launch_ao(nullptr, 1, 1, 1, nullptr, 0, 3, nullptr, nullptr,
                                         nullptr) != cudaErrorInvalidValue))
    throw std::runtime_error("AO overflow/jet validation failed");
  std::cout << "{\"contract\":\"passed\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "list") {
      list_programs();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "contract") {
      contract_checks();
      return 0;
    }
    if (argc < 4) throw std::runtime_error("expected array/integral/AO probe arguments");
    const std::string operation = argv[1];
    Stream stream;
    Device<int> status(1);
    check(cudaMemsetAsync(status.value, 0, sizeof(int), stream.value));
    if (operation == "array" && argc == 4) {
      const auto* program = aot::find(argv[2]);
      if (!program || program->kind != aot::Kind::array)
        throw std::runtime_error("array program is not in this native AOT package");
      const auto count = size_argument(argv[3]);
      const auto values = read_values(count * program->inputs.size());
      Device<double> inputs(values.size()), outputs(count * program->outputs);
      inputs.upload(values, stream.value);
      check(aot::launch_array(program->name, inputs.value, count, outputs.value, status.value,
                              stream.value));
      publish(outputs, status, stream.value);
    } else if (operation == "integral" && argc == 5) {
      const auto* program = aot::find(argv[2]);
      if (!program || program->kind != aot::Kind::integral)
        throw std::runtime_error("integral program is not in this native AOT package");
      const auto count = size_argument(argv[3]), atoms = size_argument(argv[4]);
      const auto values = read_values(count * program->record_stride);
      std::vector<long long> labels(program->centers);
      for (auto& label : labels)
        if (!(std::cin >> label) || label < 0 || static_cast<std::size_t>(label) >= atoms)
          throw std::runtime_error("invalid physical-atom map");
      const auto direction = read_values(3 * atoms);
      Device<double> records(values.size()), motion(direction.size());
      Device<double> outputs(program->outputs * program->components.size());
      Device<long long> physical(labels.size());
      records.upload(values, stream.value);
      motion.upload(direction, stream.value);
      physical.upload(labels, stream.value);
      check(aot::launch_integral(program->name, records.value, count, physical.value, motion.value,
                                 outputs.value, status.value, stream.value));
      publish(outputs, status, stream.value);
    } else if (operation == "ao" && argc == 7) {
      const auto atoms = size_argument(argv[2]), primitives = size_argument(argv[3]);
      const auto aos = size_argument(argv[4]), count = size_argument(argv[5]);
      const auto jets = size_argument(argv[6]);
      if (jets != 1 && jets != 4 && jets != 10 && jets != 20)
        throw std::runtime_error("invalid AO jet prefix");
      const auto basis = read_values(3 * atoms + 2 * primitives + 16 * aos);
      const auto points = read_values(3 * count);
      if (count * aos > (1ULL << 20) / jets)
        throw std::runtime_error("AO output exceeds explicit probe limit");
      for (std::size_t index = 0; index < aos; ++index) {
        const auto* row = basis.data() + 3 * atoms + 2 * primitives + 16 * index;
        if (row[0] < 0 || row[0] >= atoms || std::floor(row[0]) != row[0] || row[1] < 0 ||
            row[2] < 1 || row[1] + row[2] > primitives || row[3] < 1 || row[3] > 3 ||
            std::floor(row[1]) != row[1] || std::floor(row[2]) != row[2] ||
            std::floor(row[3]) != row[3])
          throw std::runtime_error("invalid packed AO metadata");
        for (std::size_t term = 0; term < static_cast<std::size_t>(row[3]); ++term)
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto power = row[4 + 4 * term + axis];
            if (power < 0 || power > 3 || std::floor(power) != power)
              throw std::runtime_error("AO powers must be s/p/d/f");
          }
      }
      Device<double> packed(basis.size()), positions(points.size()), outputs(jets * count * aos);
      packed.upload(basis, stream.value);
      positions.upload(points, stream.value);
      check(aot::launch_ao(packed.value, atoms, primitives, aos, positions.value, count, jets,
                           outputs.value, status.value, stream.value));
      publish(outputs, status, stream.value);
    } else {
      throw std::runtime_error("invalid native AOT probe command");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
