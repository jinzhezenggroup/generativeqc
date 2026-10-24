// Independent primitive qualification; no complete-endpoint performance claim.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/structured_matrix.hpp"

using generativeqc::scf::cuda_execution::MatrixLibraryOwner;
using generativeqc::scf::cuda_execution::MatrixLibraryResources;
using generativeqc::tensor::CudaMatrixProductBinding;
using generativeqc::tensor::CudaStructuredProductBinding;
using Operation = generativeqc::tensor::StructuredOperation;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void status_check(generativeqc_status status) { require(status == 0, "structured launch failed"); }

template <class Value>
struct Buffer {
  Value* data{};
  std::size_t size{};
  explicit Buffer(std::size_t count) : size(count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Value)));
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const std::vector<Value>& values, cudaStream_t stream) {
    require(values.size() == size, "upload extent changed");
    cuda_check(
        cudaMemcpyAsync(data, values.data(), size * sizeof(Value), cudaMemcpyHostToDevice, stream));
  }
  std::vector<Value> download(cudaStream_t stream) {
    std::vector<Value> values(size);
    cuda_check(
        cudaMemcpyAsync(values.data(), data, size * sizeof(Value), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return values;
  }
};

template <class Enqueue, class Verify, class Mutate>
void execute(cudaStream_t stream, bool graph, const Enqueue& enqueue, const Verify& verify,
             const Mutate& mutate) {
  cudaGraph_t captured{};
  cudaGraphExec_t executable{};
  if (graph) {
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    enqueue();
    cuda_check(cudaStreamEndCapture(stream, &captured));
    cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
  }
  for (int replay : {0, 1}) {
    if (replay) mutate();
    if (graph)
      cuda_check(cudaGraphLaunch(executable, stream));
    else
      enqueue();
    verify();
  }
  if (graph) {
    cuda_check(cudaGraphExecDestroy(executable));
    cuda_check(cudaGraphDestroy(captured));
  }
}

void compare(double actual, long double expected, long double magnitude, int dimension) {
  const auto limit =
      128 * dimension * std::numeric_limits<double>::epsilon() * std::max(1.L, magnitude);
  require(std::isfinite(actual) && std::fabs(static_cast<long double>(actual) - expected) <= limit,
          "independent long-double oracle mismatch");
}

void gram_case(cudaStream_t stream, cublasHandle_t handle, int dimension, int systems, int spins,
               int mode, bool graph, int invalid) {
  const auto states = systems * spins;
  const auto matrix = static_cast<std::size_t>(dimension) * dimension;
  const auto elements = matrix * states;
  Buffer<double> coefficients(elements), energies(static_cast<std::size_t>(states) * dimension),
      output(elements), scaled(elements), unweighted(elements), scratch(elements);
  Buffer<std::int32_t> occupations(states);
  Buffer<std::uint8_t> activity(systems), failed(systems);
  std::vector<double> host_coefficients(elements), host_energies(states * dimension);
  std::vector<std::int32_t> host_occupations(states);
  std::vector<std::uint8_t> host_active(systems, 1), host_failed(systems, 0);
  std::vector<double> initial(elements, -7.25);
  if (host_active.size() > 1) host_active.at(host_active.size() - 1) = 0;
  for (int state = 0; state < states; ++state) {
    host_occupations[state] = state % 3 == 0 ? 0 : state % 3 == 1 ? dimension / 2 : dimension;
    for (int orbital = 0; orbital < dimension; ++orbital) {
      host_energies[state * dimension + orbital] =
          mode == 2 ? 0.01 + 0.1 * (orbital + 1) : (orbital % 2 ? -0.7 : 0.4) * (orbital + 1);
      for (int row = 0; row < dimension; ++row) {
        const auto element = state * matrix + orbital * dimension + row;
        host_coefficients[element] = 0.1 * std::sin(double(element + 1));
        if (mode != 2 && orbital >= host_occupations[state])
          host_coefficients[element] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  if (invalid) {
    const double bad[] = {0, -1, 1.e-11, std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity()};
    host_energies[dimension / 2] = bad[invalid - 1];
  }
  coefficients.upload(host_coefficients, stream);
  energies.upload(host_energies, stream);
  occupations.upload(host_occupations, stream);
  activity.upload(host_active, stream);
  failed.upload(host_failed, stream);
  output.upload(initial, stream);
  CudaMatrixProductBinding square_binding(stream, handle, dimension, systems, spins);
  auto binding = CudaStructuredProductBinding::qualification(stream, handle, dimension, dimension,
                                                             systems, spins);
  const MatrixLibraryResources resources{stream,   handle,          scratch.data,
                                         elements, &square_binding, &binding};
  const auto enqueue = [&] {
    if (mode == 2)
      status_check(generativeqc::scf::cuda_execution::launch_structured_orthogonalizer(
          resources, systems, dimension, coefficients.data, energies.data, activity.data,
          output.data, failed.data, scaled.data, elements));
    else
      status_check(generativeqc::scf::cuda_execution::launch_structured_density(
          resources, systems, spins, dimension, occupations.data, coefficients.data,
          mode == 1 ? energies.data : nullptr, activity.data, output.data, scaled.data, elements,
          unweighted.data, elements));
  };
  const auto verify = [&] {
    const auto actual = output.download(stream);
    const auto failures = failed.download(stream);
    for (int state = 0; state < states; ++state) {
      const bool rejected = mode == 2 && invalid && state == 0 && host_active[state / spins];
      if (mode == 2)
        require(failures[state] == (rejected ? 1 : 0), "eigenvalue failure isolation changed");
      for (int column = 0; column < dimension; ++column) {
        for (int row = 0; row < dimension; ++row) {
          const auto element = state * matrix + column * dimension + row;
          if (!host_active[state / spins] || rejected) {
            require(std::memcmp(&actual[element], &initial[element], sizeof(double)) == 0,
                    "inactive/rejected publication changed bytes");
            continue;
          }
          long double expected = 0, magnitude = 0;
          const auto orbitals = mode == 2 ? dimension : host_occupations[state];
          for (int orbital = 0; orbital < orbitals; ++orbital) {
            auto value = static_cast<long double>(
                             host_coefficients[state * matrix + orbital * dimension + row]) *
                         host_coefficients[state * matrix + orbital * dimension + column];
            if (mode == 2)
              value /=
                  std::sqrt(static_cast<long double>(host_energies[state * dimension + orbital]));
            else {
              value *= spins == 1 ? 2 : 1;
              if (mode == 1) value *= host_energies[state * dimension + orbital];
            }
            expected += value;
            magnitude += std::fabs(value);
          }
          compare(actual[element], expected, magnitude, dimension);
        }
      }
    }
  };
  const auto mutate = [&] {
    if (host_active.size() > 1) {
      host_active.front() = 0;
      host_active.at(host_active.size() - 1) = 1;
    }
    for (int state = 0; state < states; ++state) {
      host_occupations[state] = state % 2 ? dimension : dimension / 2;
      for (int orbital = 0; orbital < dimension; ++orbital) {
        host_energies[state * dimension + orbital] *= 1.3;
        for (int row = 0; row < dimension; ++row) {
          const auto element = state * matrix + orbital * dimension + row;
          host_coefficients[element] = mode != 2 && orbital >= host_occupations[state]
                                           ? std::numeric_limits<double>::quiet_NaN()
                                           : 0.2 * std::cos(double(element + 7));
        }
      }
    }
    std::fill(initial.begin(), initial.end(), -8.25);
    coefficients.upload(host_coefficients, stream);
    energies.upload(host_energies, stream);
    occupations.upload(host_occupations, stream);
    activity.upload(host_active, stream);
    failed.upload(host_failed, stream);
    output.upload(initial, stream);
  };
  execute(stream, graph, enqueue, verify, mutate);
  if (handle) {
    require(generativeqc::scf::cuda_execution::launch_structured_density(
                resources, systems, spins, dimension, occupations.data, coefficients.data, nullptr,
                activity.data, output.data, coefficients.data,
                elements) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "alias admission changed");
    require(generativeqc::scf::cuda_execution::launch_structured_density(
                resources, systems, spins, dimension, occupations.data, coefficients.data, nullptr,
                activity.data, output.data, scaled.data,
                elements - 1) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "short scratch admitted");
  }
}

void basis_case(cudaStream_t stream, cublasHandle_t handle, int dimension, int direct, int systems,
                int spins, Operation operation, bool graph, bool spans) {
  const auto states = systems * spins;
  const auto square = static_cast<std::size_t>(dimension) * dimension;
  const auto rectangle = static_cast<std::size_t>(dimension) * direct;
  const auto direct_square = static_cast<std::size_t>(direct) * direct;
  const auto input_size = operation == Operation::DensityRight ? square
                          : operation == Operation::FockLeft   ? direct_square
                                                               : rectangle;
  const auto output_size = operation == Operation::DensityLeft ? direct_square
                           : operation == Operation::FockRight ? square
                                                               : rectangle;
  std::vector<double> transform(rectangle * systems), input(input_size * states),
      core(square * systems), initial(output_size * states, -9.5);
  std::vector<std::int32_t> host_spans(2 * systems * (dimension + direct));
  std::vector<std::uint8_t> active(systems, 1);
  if (systems > 1) active.back() = 0;
  for (int system = 0; system < systems; ++system) {
    for (int row = 0; row < dimension; ++row) {
      host_spans[2 * (system * dimension + row)] = spans ? row % direct : 0;
      host_spans[2 * (system * dimension + row) + 1] = spans ? row % direct + 1 : direct;
    }
    for (int column = 0; column < direct; ++column) {
      const auto span = 2 * (systems * dimension + system * direct + column);
      host_spans[span] = 0;
      host_spans[span + 1] = dimension;
      for (int row = 0; row < dimension; ++row)
        transform[system * rectangle + column * dimension + row] =
            spans && row % direct != column ? 0 : 0.1 * std::cos(double(row + 3 * column + 1));
    }
  }
  for (std::size_t element = 0; element < input.size(); ++element)
    input[element] = std::sin(double(element + 2));
  for (std::size_t element = 0; element < core.size(); ++element)
    core[element] = 0.3 * std::cos(double(element + 4));
  Buffer<double> device_transform(transform.size()), device_input(input.size()),
      device_core(core.size()), output(initial.size()),
      scratch(std::max(square, direct_square) * states);
  Buffer<std::uint8_t> device_active(active.size());
  Buffer<std::int32_t> device_spans(host_spans.size());
  device_transform.upload(transform, stream);
  device_input.upload(input, stream);
  device_core.upload(core, stream);
  output.upload(initial, stream);
  device_active.upload(active, stream);
  device_spans.upload(host_spans, stream);
  CudaMatrixProductBinding square_binding(stream, handle, dimension, systems, spins);
  auto binding = CudaStructuredProductBinding::qualification(stream, handle, dimension, direct,
                                                             systems, spins);
  const MatrixLibraryResources resources{stream, handle, nullptr, 0, &square_binding, &binding};
  const auto enqueue = [&] {
    status_check(generativeqc::scf::cuda_execution::launch_structured_basis(
        resources, operation, systems, spins, dimension, direct, device_transform.data,
        device_input.data, device_core.data, device_active.data, output.data, scratch.data,
        scratch.size, spans ? device_spans.data : nullptr));
  };
  const auto verify = [&] {
    const auto actual = output.download(stream);
    const int rows = operation == Operation::DensityLeft ? direct : dimension;
    const int columns = operation == Operation::FockRight ? dimension : direct;
    const int reduction = operation <= Operation::DensityLeft ? dimension : direct;
    for (int state = 0; state < states; ++state) {
      const auto system = state / spins;
      for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row) {
          const auto element = state * output_size + column * rows + row;
          if (!active[system]) {
            require(actual[element] == initial[element], "basis mask changed");
            continue;
          }
          long double expected = operation == Operation::FockRight
                                     ? core[system * square + column * dimension + row]
                                     : 0;
          long double magnitude = std::fabs(expected);
          for (int contracted = 0; contracted < reduction; ++contracted) {
            long double value{};
            if (operation == Operation::DensityRight)
              value = static_cast<long double>(
                          input[state * input_size + contracted * dimension + row]) *
                      transform[system * rectangle + column * dimension + contracted];
            else if (operation == Operation::DensityLeft)
              value = static_cast<long double>(
                          transform[system * rectangle + row * dimension + contracted]) *
                      input[state * input_size + column * dimension + contracted];
            else if (operation == Operation::FockLeft)
              value = static_cast<long double>(
                          transform[system * rectangle + contracted * dimension + row]) *
                      input[state * input_size + column * direct + contracted];
            else
              value = static_cast<long double>(
                          input[state * input_size + contracted * dimension + row]) *
                      transform[system * rectangle + contracted * dimension + column];
            expected += value;
            magnitude += std::fabs(value);
          }
          compare(actual[element], expected, magnitude, reduction);
        }
    }
  };
  const auto mutate = [&] {
    if (systems > 1) {
      active.front() = 0;
      active.back() = 1;
    }
    for (auto& value : transform) value *= -0.75;
    for (std::size_t element = 0; element < input.size(); ++element)
      input[element] = std::cos(double(element + 3));
    for (auto& value : core) value += 0.25;
    std::fill(initial.begin(), initial.end(), -11.5);
    device_transform.upload(transform, stream);
    device_input.upload(input, stream);
    device_core.upload(core, stream);
    device_active.upload(active, stream);
    output.upload(initial, stream);
  };
  execute(stream, graph, enqueue, verify, mutate);
  if (handle && !spans) {
    require(generativeqc::scf::cuda_execution::launch_structured_basis(
                resources, operation, systems, spins, dimension, direct, device_transform.data,
                device_input.data, device_core.data, device_active.data, output.data, output.data,
                scratch.size) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "basis alias admitted");
    require(generativeqc::scf::cuda_execution::launch_structured_basis(
                resources, operation, systems, spins, dimension, direct, device_transform.data,
                device_input.data, device_core.data, device_active.data, output.data, scratch.data,
                output_size * states - 1) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "short basis scratch admitted");
  }
}

int main() {
  try {
    cudaStream_t stream{};
    cuda_check(cudaStreamCreate(&stream));
    std::size_t cases = 0;
    {
      MatrixLibraryOwner owner;
      status_check(owner.prepare(stream, 17, 3, 2));
      require(owner.library_enabled(), "qualification provider unavailable");
      const auto* production = owner.view().structured_binding_;
      require(production != nullptr, "production binding absent");
      for (Operation operation :
           {Operation::Gram, Operation::SignedGram, Operation::InverseRoot, Operation::DensityRight,
            Operation::DensityLeft, Operation::FockLeft, Operation::FockRight}) {
        require(!production->library(operation), "unqualified default promoted");
        const auto offers = production->candidates(operation);
        require(offers.size() == 7 && !offers[3].rejection.empty(),
                "negative rank-2k evidence lost");
        require(!offers[1].rejection.empty() && offers[1].provider_bytes == 96ULL << 20,
                "qualification gate or provider charge missing");
        require(!offers[1].cost.measured && !offers[1].cost.kernel_ns,
                "primitive cost invented an endpoint crossover");
      }
      for (int dimension : {3, 17, 31, 33, 96})
        for (int systems : {1, 3})
          for (int spins : {1, 2})
            for (bool graph : {false, true})
              for (bool library : {false, true}) {
                const auto handle = library ? owner.view().blas_ : nullptr;
                for (int mode : {0, 1, 2}) {
                  if (mode == 2 && spins != 1) continue;
                  gram_case(stream, handle, dimension, systems, spins, mode, graph, 0);
                  ++cases;
                  if (mode == 2)
                    for (int invalid : {1, 2, 3, 4, 5}) {
                      gram_case(stream, handle, dimension, systems, spins, mode, graph, invalid);
                      ++cases;
                    }
                }
                for (Operation operation : {Operation::DensityRight, Operation::DensityLeft,
                                            Operation::FockLeft, Operation::FockRight})
                  for (bool spans : {false, true})
                    for (int direct : {dimension + 4, std::max(1, dimension - 2)}) {
                      basis_case(stream, handle, dimension, direct, systems, spins, operation,
                                 graph, spans);
                      ++cases;
                    }
              }
      cuda_check(cudaStreamSynchronize(stream));
    }
    cuda_check(cudaStreamDestroy(stream));
    std::cout << "{\"status\":\"PASS\",\"cases_passed\":" << cases
              << ",\"replays_per_case\":2,\"changed_graph_data_and_masks\":true"
              << ",\"complete_endpoint_qualified\":false}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
