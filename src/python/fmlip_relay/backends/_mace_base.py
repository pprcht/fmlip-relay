"""
fmlip_relay.backends._mace_base
------------------------------
Internal mixin that provides the shared ASE Atoms → energy/forces/stress
evaluation logic used by all MACE backends.

Not part of the public API.
"""

from __future__ import annotations

import numpy as np
from .base import BackendBase


class _MACEComputeMixin(BackendBase):
    """
    Mixin that implements ``compute()`` on top of a stored ASE calculator
    ``self._calc``.  Subclasses only need to construct ``self._calc`` in
    ``__init__``.
    """

    _calc = None   # must be set by subclass __init__

    def compute(self,
                atomic_numbers: np.ndarray,
                positions:      np.ndarray,
                cell:           np.ndarray,
                pbc:            np.ndarray,
                compute_stress: bool,
                ) -> tuple[float, np.ndarray, np.ndarray]:
        from ase import Atoms

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        atoms.calc = self._calc

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().astype(np.float64)
        stress = (
            atoms.get_stress(voigt=False).astype(np.float64)
            if compute_stress
            else np.zeros((3, 3), dtype=np.float64)
        )
        return energy, forces, stress
