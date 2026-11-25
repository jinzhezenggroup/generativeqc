"""Shared provider-carrier metadata for compiled and explicit KS graphs."""

from generativeqc_compiler.xc.spec import functional

from .dispersion import D4Spec, DispersionCorrectionPrimitive
from .spec import MethodIR, SemilocalXCPrimitive


def is_pbe_d4_composition(method_ir: object) -> bool:
    """Match the complete restricted PBE plus EEQ-D4 graph, not its label."""
    if not isinstance(method_ir, MethodIR) or len(method_ir.primitives) != 2:
        return False
    semilocal, correction = method_ir.primitives
    pbe = functional("PBE", spin="unpolarized")
    return (
        type(semilocal) is SemilocalXCPrimitive
        and type(correction) is DispersionCorrectionPrimitive
        and isinstance(correction.specification, D4Spec)
        and correction.specification.charge_model == "eeq2019"
        and correction.specification.reference_model == "eeq"
        and semilocal.functional.spin == "unpolarized"
        and semilocal.functional.ingredients == pbe.ingredients
        and SemilocalXCPrimitive(semilocal.functional).semantic_payload()
        == SemilocalXCPrimitive(pbe).semantic_payload()
    )


def electronic_carrier(ingredients: tuple[str, ...], spin: str) -> str:
    """Name the registry carrier after a lowerer validates the electronic graph."""
    family = {
        ("rho",): "lda",
        ("rho", "sigma"): "pbe",
        ("rho", "sigma", "tau"): "r2scan",
    }.get(ingredients)
    if family is None:
        raise NotImplementedError(
            "native KS has no provider carrier for these XC ingredients"
        )
    suffix = "uks" if spin == "polarized" else "rks"
    return f"{family}-{suffix}"
