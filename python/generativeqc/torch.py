"""PyTorch energies with qualified analytic first- and second-order derivatives."""

from __future__ import annotations

import typing

import torch

from .calculator import Calculator
from .elements import checked_integer

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from .batch import PreparedBatch

__all__ = ["batched_energy", "energy", "integral_energy"]


def _validated_atomic_numbers(
    coordinates: typing.Any, atomic_numbers: typing.Any
) -> typing.Any:
    """Preserve molecular identity before any transfer or native evaluation."""
    if not isinstance(coordinates, torch.Tensor) or not coordinates.is_floating_point():
        raise TypeError("coordinates must be a real floating-point tensor")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (natoms, 3)")
    if len(atomic_numbers) != coordinates.shape[0]:
        raise ValueError("atomic_numbers does not match its coordinates")
    return tuple(
        checked_integer(value, "atomic number", low=1, high=118)
        for value in atomic_numbers
    )


def _require_second_order(calculator: typing.Any) -> None:
    """Expose only a context's independently qualified native HVP consumer."""
    if "hvp" not in getattr(calculator, "second_order_capabilities", ()):
        raise RuntimeError(
            "higher-order derivatives are unsupported for this calculator context"
        )


def _check_model_signature(calculator: typing.Any, signature: typing.Any) -> None:
    """Keep a later response on the model that produced the saved energy/forces."""
    if signature is not None and calculator._model_signature() != signature:
        raise RuntimeError("calculator model changed after differentiable forward")


class _CoordinateGradient(torch.autograd.Function):
    """Bind an analytic gradient to coordinates with a native HVP backward."""

    @staticmethod
    def forward(
        ctx: typing.Any,
        coordinates: typing.Any,
        gradient: typing.Any,
        calculator: typing.Any,
        numbers: typing.Any,
        charge: int,
        multiplicity: int,
        model_signature: typing.Any,
    ) -> typing.Any:
        _check_model_signature(calculator, model_signature)
        ctx.save_for_backward(coordinates)
        ctx.calculator, ctx.numbers = calculator, numbers
        ctx.charge, ctx.multiplicity = charge, multiplicity
        ctx.model_signature = model_signature
        return gradient.clone()

    @staticmethod
    def backward(ctx: typing.Any, direction: typing.Any) -> typing.Any:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "third-order derivatives through native response are unsupported"
            )
        (coordinates,) = ctx.saved_tensors
        _check_model_signature(ctx.calculator, ctx.model_signature)
        xyz = coordinates.detach().to(device="cpu", dtype=torch.float64).numpy()
        vector = direction.detach().to(device="cpu", dtype=torch.float64).numpy()
        result = ctx.calculator.hessian_vector_product(
            list(zip(ctx.numbers, xyz, strict=True)),
            vector,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
        )
        product = torch.as_tensor(
            result.value.copy(), dtype=coordinates.dtype, device=coordinates.device
        )
        return product, None, None, None, None, None, None


class _EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: typing.Any,
        coordinates: typing.Any,
        atomic_numbers: typing.Any,
        calculator: typing.Any,
        charge: typing.Any,
        multiplicity: typing.Any,
    ) -> typing.Any:
        # Capture before the primal call, rather than when first backward later
        # builds its graph: callers may mutate the calculator between the two.
        ctx.model_signature = (
            calculator._model_signature()
            if hasattr(calculator, "_model_signature")
            else None
        )
        if coordinates.device.type != "cpu":
            # The C ABI can accept device buffers in a later version; the MVP
            # makes this transfer explicit instead of hiding it in native code.
            coordinate_values = coordinates.detach().cpu()
        else:
            coordinate_values = coordinates.detach()
        atoms = [
            (atomic_numbers[index], coordinate_values[index].tolist())
            for index in range(coordinates.shape[0])
        ]
        result = calculator.singlepoint(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
        force_tensor = torch.as_tensor(
            result.forces, dtype=coordinates.dtype, device=coordinates.device
        )
        ctx.save_for_backward(coordinates, force_tensor)
        ctx.calculator, ctx.numbers = calculator, atomic_numbers
        ctx.charge, ctx.multiplicity = charge, multiplicity
        return coordinates.new_tensor(result.energy)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: typing.Any) -> typing.Any:  # type: ignore[override]
        coordinates, forces = ctx.saved_tensors
        gradient = -forces
        if torch.is_grad_enabled():
            _require_second_order(ctx.calculator)
            gradient = _CoordinateGradient.apply(
                coordinates,
                gradient,
                ctx.calculator,
                ctx.numbers,
                ctx.charge,
                ctx.multiplicity,
                ctx.model_signature,
            )
        return gradient * grad_output, None, None, None, None


def energy(
    coordinates: torch.Tensor,
    atomic_numbers: Sequence[int],
    calculator: Calculator | None = None,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> torch.Tensor:
    """Return energy with analytic coordinate backward and qualified double backward.

    Coordinates must be a real floating-point tensor; energy and coordinate
    gradients retain its dtype and device. Atomic numbers, charge and
    multiplicity must be exact integers (not bools).
    ``create_graph=True`` and double backward require the calculator's qualified
    HVP capability. An unqualified context rejects before detached forces could
    hide missing Hessian terms. Third derivatives remain unsupported.

    Coordinates have shape ``(natoms, 3)`` in Bohr; the result is a scalar
    in Hartree and backward returns ``dE/dR`` in Hartree/Bohr. Forward
    explicitly stages non-CPU coordinates on the host; the input tensor's
    device does not select the calculator backend. Validation and native
    execution failures propagate without an energy tensor. Saved detached
    forces survive for first-order backward. See :ref:`python-torch-values`,
    :ref:`python-torch-errors`, :ref:`python-torch-ownership` and
    :ref:`python-torch-backends`."""

    atomic_numbers = _validated_atomic_numbers(coordinates, atomic_numbers)
    charge = checked_integer(charge, "ionic charge", low=-(2**31))
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1)
    if calculator is None:
        calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    return _EnergyFunction.apply(
        coordinates,
        atomic_numbers,
        calculator,
        charge,
        multiplicity,
    )


class _BatchedEnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: typing.Any,
        calculator: typing.Any,
        prepared_batch: typing.Any,
        atomic_numbers: typing.Any,
        charges: typing.Any,
        multiplicities: typing.Any,
        *coordinates: typing.Any,
    ) -> typing.Any:
        if not coordinates:
            raise ValueError("batched energy requires at least one system")
        reference = coordinates[0]
        for item in coordinates:
            if item.device != reference.device or item.dtype != reference.dtype:
                raise ValueError(
                    "all ragged coordinate tensors must share device and dtype"
                )

        ctx.calculator = getattr(prepared_batch, "_calculator", calculator)
        ctx.model_signature = (
            ctx.calculator._model_signature()
            if hasattr(ctx.calculator, "_model_signature")
            else None
        )
        # Native coordinates are FP64. Convert before NumPy so real Torch
        # dtypes without a NumPy representation (notably bfloat16) work too.
        cpu_coordinates = [
            item.detach().to(device="cpu", dtype=torch.float64).numpy()
            for item in coordinates
        ]
        if prepared_batch is not None:
            if prepared_batch.atomic_numbers != atomic_numbers:
                raise ValueError(
                    "prepared batch topology does not match atomic numbers"
                )
            if prepared_batch.charges != charges:
                raise ValueError("prepared batch charges do not match")
            if prepared_batch.multiplicities != multiplicities:
                raise ValueError("prepared batch multiplicities do not match")
            result = prepared_batch.execute(
                cpu_coordinates, strict=True, properties=("energy", "forces")
            )
        else:
            systems = [
                [
                    (atomic_numbers[system][atom], cpu_coordinates[system][atom])
                    for atom in range(len(atomic_numbers[system]))
                ]
                for system in range(len(coordinates))
            ]
            result = calculator.batch_singlepoint(
                systems,
                charges=charges,
                multiplicities=multiplicities,
                strict=True,
                properties=("energy", "forces"),
            )

        force_tensors = tuple(
            torch.as_tensor(item.forces, dtype=reference.dtype, device=reference.device)
            for item in result.items
        )
        ctx.save_for_backward(*coordinates, *force_tensors)
        ctx.count = len(coordinates)
        ctx.numbers, ctx.charges, ctx.multiplicities = (
            atomic_numbers,
            charges,
            multiplicities,
        )
        return torch.as_tensor(
            result.energies, dtype=reference.dtype, device=reference.device
        )

    @staticmethod
    def backward(ctx: typing.Any, grad_output: typing.Any) -> typing.Any:  # type: ignore[override]
        coordinates = ctx.saved_tensors[: ctx.count]
        forces = ctx.saved_tensors[ctx.count :]
        if torch.is_grad_enabled():
            _require_second_order(ctx.calculator)
            gradients = tuple(
                _CoordinateGradient.apply(
                    r,
                    -force,
                    ctx.calculator,
                    ctx.numbers[i],
                    ctx.charges[i],
                    ctx.multiplicities[i],
                    ctx.model_signature,
                )
                for i, (r, force) in enumerate(zip(coordinates, forces, strict=True))
            )
        else:
            gradients = tuple(-force for force in forces)
        coordinate_gradients = tuple(
            gradient * grad_output[i] for i, gradient in enumerate(gradients)
        )
        return None, None, None, None, None, *coordinate_gradients


def batched_energy(
    coordinates: Sequence[torch.Tensor],
    atomic_numbers: Sequence[Sequence[int]],
    calculator: Calculator | None = None,
    *,
    charges: Sequence[int] | None = None,
    multiplicities: Sequence[int] | None = None,
    prepared_batch: PreparedBatch | None = None,
) -> torch.Tensor:
    """Evaluate a ragged native HF batch with analytic coordinate backward.

    Systems remain separate tensors, so the interface never pads all molecules
    to the largest atom count. Passing a `PreparedBatch` enables native
    topology-aware warm starts across repeated forward calls.

    Coordinate tensors must be real floating point; energies and gradients
    retain their shared dtype and device. Each atomic-number list must match
    its coordinates. Atomic numbers, charges
    and multiplicities must be exact integers (not bools). Double backward uses
    the executing calculator's qualified HVP capability for each item, including
    caller-owned prepared batches. Third derivatives remain unsupported.

    Each coordinate tensor has shape ``(natoms_i, 3)`` in Bohr; the result
    has shape ``(nsystems,)`` in Hartree. Forward stages all coordinates as
    CPU float64 and copies energies/saved forces to the shared input
    dtype/device. Native execution uses strict mode: any failed item raises
    rather than publishing a partial energy tensor. A caller-owned prepared
    batch must remain open and must not be used concurrently. Its default
    model must support forces, which this wrapper explicitly requests even
    when the native method has an energy-only default. See
    :ref:`python-torch-errors`, :ref:`python-torch-ownership` and
    :ref:`python-torch-backends`."""

    if len(coordinates) != len(atomic_numbers) or not coordinates:
        raise ValueError(
            "coordinates and atomic_numbers must have equal nonzero length"
        )
    normalized_atomic_numbers = tuple(
        _validated_atomic_numbers(item, numbers)
        for item, numbers in zip(coordinates, atomic_numbers, strict=True)
    )
    normalized_charges = (
        tuple(0 for _ in coordinates)
        if charges is None
        else tuple(
            checked_integer(value, "ionic charge", low=-(2**31)) for value in charges
        )
    )
    normalized_multiplicities = (
        tuple(1 for _ in coordinates)
        if multiplicities is None
        else tuple(
            checked_integer(value, "multiplicity", low=1) for value in multiplicities
        )
    )
    if len(normalized_charges) != len(coordinates) or len(
        normalized_multiplicities
    ) != len(coordinates):
        raise ValueError("charges and multiplicities must match the ragged batch size")
    if calculator is None:
        calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    return _BatchedEnergyFunction.apply(
        calculator,
        prepared_batch,
        normalized_atomic_numbers,
        normalized_charges,
        normalized_multiplicities,
        *tuple(coordinates),
    )


class _IntegralGradient(torch.autograd.Function):
    """Joint parameter gradient whose adjoint is the relaxed integral HVP."""

    @staticmethod
    def forward(
        ctx: typing.Any,
        coordinates: typing.Any,
        exponents: typing.Any,
        coefficients: typing.Any,
        state: typing.Any,
        *gradients: typing.Any,
    ) -> typing.Any:
        ctx.save_for_backward(coordinates, exponents, coefficients)
        ctx.state = state
        ctx.set_materialize_grads(False)
        return tuple(value.clone() for value in gradients)

    @staticmethod
    def backward(ctx: typing.Any, *directions: typing.Any) -> typing.Any:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "third-order derivatives through integral response are unsupported"
            )
        from .experimental.integral_autodiff import IntegralGradient

        inputs = ctx.saved_tensors
        arrays = tuple(
            (torch.zeros_like(value) if direction is None else direction)
            .detach()
            .cpu()
            .numpy()
            for value, direction in zip(inputs, directions, strict=True)
        )
        product = ctx.state.hvp(IntegralGradient(*arrays))
        gradients = tuple(
            torch.as_tensor(array.copy(), dtype=value.dtype, device=value.device)
            for array, value in zip(
                (product.coordinates, product.exponents, product.coefficients),
                inputs,
                strict=True,
            )
        )
        return (*gradients, None, None, None, None)


class _IntegralEnergy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: typing.Any,
        coordinates: typing.Any,
        exponents: typing.Any,
        coefficients: typing.Any,
        owner: typing.Any,
    ) -> typing.Any:
        state = owner.prepare(
            coordinates.detach().cpu().numpy(),
            exponents=exponents.detach().cpu().numpy(),
            coefficients=coefficients.detach().cpu().numpy(),
        )
        try:
            derivative = state.gradient()
            gradients = tuple(
                torch.as_tensor(
                    array.copy(), dtype=coordinates.dtype, device=coordinates.device
                )
                for array in (
                    derivative.coordinates,
                    derivative.exponents,
                    derivative.coefficients,
                )
            )
        except BaseException:
            state.close()
            raise
        object.__setattr__(owner, "_diagnostics", state.diagnostics())
        ctx.save_for_backward(coordinates, exponents, coefficients, *gradients)
        ctx.state, ctx.second_order = state, owner.derivative_order == 2
        return coordinates.new_tensor(state.energy)

    @staticmethod
    def backward(ctx: typing.Any, cotangent: typing.Any) -> typing.Any:
        inputs, gradients = ctx.saved_tensors[:3], ctx.saved_tensors[3:]
        if torch.is_grad_enabled():
            if not ctx.second_order:
                raise RuntimeError(
                    "integral double backward requires derivative_order=2"
                )
            gradients = _IntegralGradient.apply(*inputs, ctx.state, *gradients)
        return (*(gradient * cotangent for gradient in gradients), None)


def integral_energy(
    coordinates: torch.Tensor,
    exponents: torch.Tensor,
    coefficients: torch.Tensor,
    owner: typing.Any,
) -> torch.Tensor:
    """Differentiate RHF/UHF energy jointly through coordinates and raw basis inputs.

    ``owner`` is an experimental IntegralHF instance. Inputs are finite FP64
    tensors on one device, using that owner's units, shapes and fixed topology.
    CPU staging is explicit; the owner selects scientific CPU/CUDA execution.
    ``derivative_order=2`` admits double backward through the retained converged
    state, including mixed basis/geometry terms. Third derivatives reject. The
    graph owns its state independently of ``owner.close()`` and input mutation
    is checked by Torch's saved-tensor versioning.
    """
    from .experimental.integral_autodiff import IntegralHF

    if not isinstance(owner, IntegralHF):
        raise TypeError("owner must be an IntegralHF instance")
    inputs = (coordinates, exponents, coefficients)
    if any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float64
        for value in inputs
    ):
        raise TypeError("integral AD inputs must be float64 Torch tensors")
    if any(value.device != coordinates.device for value in inputs):
        raise ValueError("integral AD inputs must share one device")
    return _IntegralEnergy.apply(*inputs, owner)
