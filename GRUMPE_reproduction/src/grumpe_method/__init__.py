"""Core algorithms for the Grumpe lunar DEM reproduction."""

from .integration import IntegrationResult, forward_gradient, integrate_gradients
from .extended_sfs import ExtendedSFSResult, estimate_extended_sfs
from .photoclinometry import PHCLResult, estimate_phcl
from .registration import RegistrationResult, register_by_single_image_phcl
from .reflectance import (
    HapkeParameters,
    hapke_amsa,
    hapke_imsa,
    hapke_reflectance,
    warell_moon_2004,
)
from .relaxation import RelaxationResult, solve_paper_relaxation

__all__ = [
    "HapkeParameters",
    "IntegrationResult",
    "ExtendedSFSResult",
    "PHCLResult",
    "RelaxationResult",
    "RegistrationResult",
    "estimate_extended_sfs",
    "estimate_phcl",
    "forward_gradient",
    "hapke_amsa",
    "hapke_imsa",
    "hapke_reflectance",
    "warell_moon_2004",
    "integrate_gradients",
    "register_by_single_image_phcl",
    "solve_paper_relaxation",
]
