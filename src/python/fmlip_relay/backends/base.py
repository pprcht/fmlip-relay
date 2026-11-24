"""
fmlip_relay.backends.base
------------------------
Abstract base class that every backend must implement.

A backend is responsible for one thing only: given an atomic configuration,
return energy, forces, and (optionally) stress.  All I/O and process
management is handled by the server layer.
"""

from abc import ABC, abstractmethod
import numpy as np


def partition_energy(energy: float, natoms: int,
                     atomic_energies=None) -> np.ndarray:
    """
    Return ``(natoms,)`` per-atom energies that sum to *energy*.

    Without *atomic_energies* (or if their shape does not fit) the total is
    split equally. Otherwise the model's values are used, and whatever they
    leave out of the total (e.g. a dispersion correction added by a wrapper
    calculator) is spread equally over the atoms.
    """
    if atomic_energies is not None:
        e_at = np.array(atomic_energies, dtype=np.float64).reshape(-1)
        if e_at.shape == (natoms,) and np.all(np.isfinite(e_at)):
            return e_at + (energy - e_at.sum()) / natoms
    return np.full(natoms, energy / natoms, dtype=np.float64)


class BackendBase(ABC):
    """
    Minimal interface every fmlip_relay backend must satisfy.

    Subclasses should load their model in ``__init__`` so that loading
    happens exactly once at server startup.
    """

    @abstractmethod
    def compute(self,
                atomic_numbers: np.ndarray,
                positions:      np.ndarray,
                cell:           np.ndarray,
                pbc:            np.ndarray,
                compute_stress: bool,
                charge: int,
                spin: int,
                ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the potential.

        Parameters
        ----------
        atomic_numbers : (N,)   int32  – atomic numbers (Z)
        positions      : (N, 3) float64 – Cartesian coordinates, Angstrom
        cell           : (3, 3) float64 – lattice row vectors, Angstrom
        pbc            : (3,)   bool    – periodic boundary flags
        compute_stress : bool           – whether to compute the stress tensor
        charge         : int32          - total charge (neutral = 0)
        spin           : int32          - spin multiplicity 2S+1
                                          (ground-state singlet = 1)

        Returns
        -------
        energy : float         – potential energy, eV
        forces : (N, 3) float64 – forces,  eV / Angstrom
        stress : (3, 3) float64 – stress tensor, eV / Angstrom^3
                                  (zero matrix when compute_stress is False)
        """

    def compute_atomic(self,
                       atomic_numbers: np.ndarray,
                       positions:      np.ndarray,
                       cell:           np.ndarray,
                       pbc:            np.ndarray,
                       compute_stress: bool,
                       charge: int,
                       spin: int,
                       ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the potential and also return per-atom energies.

        Takes the arguments of :meth:`compute` and returns
        ``(energy, forces, stress, atomic_energies)``, where
        ``atomic_energies`` is an ``(N,)`` float64 array in eV that sums to
        ``energy``.

        The default splits the total energy equally over the atoms.
        Backends whose model has a real per-atom decomposition override this.
        """
        energy, forces, stress = self.compute(
            atomic_numbers, positions, cell, pbc, compute_stress, charge, spin)
        return energy, forces, stress, partition_energy(energy, len(atomic_numbers))

    #: displacement in Å for the finite-difference Hessian
    hessian_step = 0.005

    def compute_hessian(self,
                        atomic_numbers: np.ndarray,
                        positions:      np.ndarray,
                        cell:           np.ndarray,
                        pbc:            np.ndarray,
                        charge: int,
                        spin: int,
                        ) -> np.ndarray:
        """
        Return the Hessian, the second derivatives of the energy with respect
        to the atomic positions, as a symmetric ``(3N, 3N)`` float64 array in
        eV/Å². Row and column ``3*i + a`` belong to Cartesian component ``a``
        of atom ``i``. The cell is held fixed.

        The default takes central finite differences of the forces, which
        costs 6N calls of :meth:`compute`. Backends that can differentiate
        their model analytically override this.
        """
        return self._numerical_hessian(atomic_numbers, positions, cell, pbc,
                                       charge, spin)

    def _numerical_hessian(self, atomic_numbers, positions, cell, pbc,
                           charge, spin) -> np.ndarray:
        h = float(self.hessian_step)
        x0 = np.array(positions, dtype=np.float64)
        n3 = x0.size
        hessian = np.empty((n3, n3), dtype=np.float64)
        for k in range(n3):
            forces = []
            for sign in (+1.0, -1.0):
                x = x0.copy()
                x.flat[k] += sign * h
                forces.append(np.asarray(self.compute(
                    atomic_numbers, x, cell, pbc, False, charge, spin)[1],
                    dtype=np.float64).reshape(n3))
            hessian[:, k] = -(forces[0] - forces[1]) / (2.0 * h)
        return 0.5 * (hessian + hessian.T)

    @property
    def name(self) -> str:
        """Human-readable backend name for log messages."""
        return type(self).__name__

    @property
    def device(self) -> str | None:
        """Compute device (e.g. ``"cpu"``, ``"cuda:0"``), or None if N/A."""
        return getattr(self, "_device", None)

    @property
    def precision(self) -> str | None:
        """Floating-point precision (e.g. ``"float32"``), or None if N/A."""
        return getattr(self, "_dtype", None)

    def close(self) -> None:
        """Release resources (temp dirs, GPU contexts) on server shutdown.

        Called once when the server exits, also after SIGTERM/SIGHUP/SIGINT.
        The default does nothing.
        """
