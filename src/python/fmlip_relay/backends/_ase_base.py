"""
fmlip_relay.backends._ase_base
------------------------------
Generic mixin that implements ``compute()`` on top of any ASE calculator
stored in ``self._calc``.  Subclasses only need to construct ``self._calc``
in ``__init__``; everything else is handled here.

Not part of the public API.
"""

from __future__ import annotations

import logging

import numpy as np
from .base import BackendBase, partition_energy

log = logging.getLogger(__name__)


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
        return self._evaluate(atomic_numbers, positions, cell, pbc,
                              compute_stress, charge, spin)[:3]

    def compute_atomic(self,
                       atomic_numbers: np.ndarray,
                       positions:      np.ndarray,
                       cell:           np.ndarray,
                       pbc:            np.ndarray,
                       compute_stress: bool,
                       charge:         int,
                       spin:           int,
                       ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Per-atom energies come from the calculator's ``energies`` result
        if it has one (MACE, EMT, ...), else the total is split equally."""
        return self._evaluate(atomic_numbers, positions, cell, pbc,
                              compute_stress, charge, spin, atomic=True)

    def _make_atoms(self, atomic_numbers, positions, cell, pbc, charge, spin):
        """Build the ASE Atoms object (in model length units) for one request."""
        from ase import Atoms

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
        return atoms

    def _evaluate(self, atomic_numbers, positions, cell, pbc,
                  compute_stress, charge, spin, atomic=False):
        sE = float(self._e_to_ev)
        sL = float(self._l_to_ang)
        atoms = self._make_atoms(atomic_numbers, positions, cell, pbc, charge, spin)

        energy = float(atoms.get_potential_energy()) * sE
        forces = atoms.get_forces().astype(np.float64) * (sE / sL)
        stress = (
            atoms.get_stress(voigt=False).astype(np.float64) * (sE / sL**3)
            if compute_stress
            else np.zeros((3, 3), dtype=np.float64)
        )
        if not atomic:
            return energy, forces, stress, None

        # "energies" is ASE's name for per-atom energies. Calculators that
        # do not provide it raise or leave it unset.
        raw = (getattr(self._calc, "results", None) or {}).get("energies")
        if raw is None:
            try:
                raw = atoms.get_potential_energies()
            except Exception:
                raw = None
        if raw is not None:
            raw = np.asarray(raw, dtype=np.float64) * sE
        return energy, forces, stress, partition_energy(energy, len(atoms), raw)

    def compute_hessian(self,
                        atomic_numbers: np.ndarray,
                        positions:      np.ndarray,
                        cell:           np.ndarray,
                        pbc:            np.ndarray,
                        charge:         int,
                        spin:           int,
                        ) -> np.ndarray:
        """Use the calculator's own Hessian (autograd in MACE, a ``hessian``
        property elsewhere) if it has one, else finite differences of forces."""
        n3 = 3 * len(atomic_numbers)
        raw = None
        try:
            atoms = self._make_atoms(atomic_numbers, positions, cell, pbc, charge, spin)
            if callable(getattr(self._calc, "get_hessian", None)):
                raw = self._calc.get_hessian(atoms)
            elif "hessian" in (getattr(self._calc, "implemented_properties", None) or ()):
                raw = self._calc.get_property("hessian", atoms)
        except Exception as exc:
            log.info("No analytical Hessian from %s (%s).", self.name, exc)
            raw = None

        if raw is not None:
            raw = np.asarray(raw, dtype=np.float64)
            if raw.size == n3 * n3 and np.all(np.isfinite(raw)):
                hessian = raw.reshape(n3, n3) * (
                    float(self._e_to_ev) / float(self._l_to_ang) ** 2)
                return 0.5 * (hessian + hessian.T)
            log.info("Analytical Hessian of %s has an unexpected shape %s.",
                     self.name, raw.shape)

        log.info("Using a finite-difference Hessian for %s.", self.name)
        return self._numerical_hessian(atomic_numbers, positions, cell, pbc,
                                       charge, spin)
