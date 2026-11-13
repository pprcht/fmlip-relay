"""
fmlip_relay.backends._ase_base
------------------------------
Generic mixin that implements ``compute()`` on top of any ASE calculator
stored in ``self._calc``.  Subclasses only need to construct ``self._calc``
in ``__init__``; everything else is handled here.

Not part of the public API.
"""

from __future__ import annotations

import numpy as np
from .base import BackendBase


class _ASEComputeMixin(BackendBase):
    """
    Mixin that implements ``compute()`` on top of a stored ASE calculator
    ``self._calc``.  Subclasses only need to construct ``self._calc`` in
    ``__init__``.

    ``_e_to_ev`` and ``_l_to_ang`` convert a model that works in other
    units: positions and cell are passed to the calculator in model length
    units, and energy, forces and stress are converted back to eV and Å.
    """

    _calc = None     # must be set by subclass __init__
    _e_to_ev = 1.0   # eV per model energy unit
    _l_to_ang = 1.0  # Å per model length unit

    def compute(self,
                atomic_numbers: np.ndarray,
                positions:      np.ndarray,
                cell:           np.ndarray,
                pbc:            np.ndarray,
                compute_stress: bool,
                charge:         int,
                spin:           int,
                ) -> tuple[float, np.ndarray, np.ndarray]:
        from ase import Atoms

        sE = float(self._e_to_ev)
        sL = float(self._l_to_ang)

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=np.asarray(positions, dtype=np.float64) / sL,
            cell=np.asarray(cell, dtype=np.float64) / sL,
            pbc=pbc,
        )
        # Spin is the multiplicity (2S+1), which is >= 1; a stray 0 (e.g. an
        # uninitialised caller default) collapses to the ground-state singlet.
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin if spin >= 1 else 1
        # Backends that model a field response (mace_polar) set this.
        external_field = getattr(self, "_external_field", None)
        if external_field is not None:
            atoms.info["external_field"] = list(external_field)
        atoms.calc = self._calc

        energy = float(atoms.get_potential_energy()) * sE
        forces = atoms.get_forces().astype(np.float64) * (sE / sL)
        stress = (
            atoms.get_stress(voigt=False).astype(np.float64) * (sE / sL**3)
            if compute_stress
            else np.zeros((3, 3), dtype=np.float64)
        )
        return energy, forces, stress
