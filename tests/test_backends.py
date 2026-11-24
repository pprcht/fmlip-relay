"""
tests/test_backends.py
----------------------
Tests for:
  - BackendBase contract (cannot instantiate abstract class)
  - DummyBackend output shapes, dtypes, and reproducibility
  - LJBackend constructor validation, physics correctness
"""

import numpy as np
import pytest

from fmlip_relay.backends.base  import BackendBase
from fmlip_relay.backends.dummy import DummyBackend
from fmlip_relay.backends.lj    import LJBackend, _image_vectors


# ── BackendBase ───────────────────────────────────────────────────────────────

class TestBackendBase:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BackendBase()

    def test_name_default(self):
        """Concrete subclass without name override uses class name."""
        class Minimal(BackendBase):
            def compute(self, *a, **kw):
                return 0.0, np.zeros((1,3)), np.zeros((3,3))
        assert Minimal().name == "Minimal"


# ── DummyBackend ──────────────────────────────────────────────────────────────

class TestDummyBackend:
    @pytest.fixture
    def backend(self):
        return DummyBackend(seed=0)

    def _call(self, backend, natoms=4, compute_stress=False):
        Z   = np.ones(natoms, dtype=np.int32)
        pos = np.zeros((natoms, 3))
        cell = np.eye(3) * 10.0
        pbc  = np.array([False, False, False])
        return backend.compute(Z, pos, cell, pbc, compute_stress)

    def test_output_shapes(self, backend):
        N = 7
        energy, forces, stress = self._call(backend, natoms=N)
        assert isinstance(energy, float)
        assert forces.shape == (N, 3)
        assert stress.shape == (3, 3)

    def test_forces_dtype(self, backend):
        _, forces, stress = self._call(backend, natoms=3)
        assert forces.dtype == np.float64
        assert stress.dtype == np.float64

    def test_stress_zeros(self, backend):
        _, _, stress = self._call(backend, compute_stress=False)
        np.testing.assert_array_equal(stress, np.zeros((3, 3)))

    def test_reproducibility_with_same_seed(self):
        b1 = DummyBackend(seed=99)
        b2 = DummyBackend(seed=99)
        e1, f1, _ = self._call(b1)
        e2, f2, _ = self._call(b2)
        assert e1 == e2
        np.testing.assert_array_equal(f1, f2)

    def test_different_seeds_give_different_results(self):
        b1 = DummyBackend(seed=1)
        b2 = DummyBackend(seed=2)
        e1, _, _ = self._call(b1)
        e2, _, _ = self._call(b2)
        assert e1 != e2

    def test_name(self, backend):
        assert backend.name == "dummy"


# ── LJBackend – constructor ───────────────────────────────────────────────────

class TestLJBackendConstructor:
    def test_defaults_are_argon(self):
        b = LJBackend()
        assert b._eps  == pytest.approx(0.0104)
        assert b._sig  == pytest.approx(3.40)
        assert b._rcut == pytest.approx(8.50)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            LJBackend(epsilon=-1.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            LJBackend(sigma=0.0)

    def test_negative_cutoff_raises(self):
        with pytest.raises(ValueError):
            LJBackend(r_cut=-1.0)

    def test_name_contains_params(self):
        b = LJBackend(epsilon=0.01, sigma=3.0, r_cut=7.5)
        assert "lj(" in b.name
        assert "eV"  in b.name
        assert "Å"   in b.name


# ── LJBackend – output contract ───────────────────────────────────────────────

class TestLJBackendOutputContract:
    @pytest.fixture
    def lj(self):
        return LJBackend()

    def test_output_shapes(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        N = len(pos)
        energy, forces, stress = lj.compute(Z, pos, cell, pbc, True)
        assert isinstance(energy, float)
        assert forces.shape == (N, 3)
        assert stress.shape == (3, 3)

    def test_forces_dtype_float64(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        _, forces, stress = lj.compute(Z, pos, cell, pbc, False)
        assert forces.dtype == np.float64
        assert stress.dtype == np.float64

    def test_stress_zero_when_not_requested(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        _, _, stress = lj.compute(Z, pos, cell, pbc, compute_stress=False)
        np.testing.assert_array_equal(stress, np.zeros((3, 3)))


# ── LJBackend – physics ───────────────────────────────────────────────────────

class TestLJBackendPhysics:
    """
    Physical correctness tests.  We use a small epsilon / sigma so that
    the numbers are easy to reason about.
    """

    @pytest.fixture
    def lj(self):
        return LJBackend(epsilon=1.0, sigma=1.0, r_cut=4.0)

    # ── Two-atom analytical comparison ───────────────────────────────────────

    def test_two_atoms_energy_analytic(self, lj):
        """
        For two atoms at separation r, the force-shifted energy is:
            E = 4ε[(σ/r)^12 - (σ/r)^6] - E(rc) - (r-rc)·dE/dr|rc
        Compare against manually computed value at r = 1.5σ.
        """
        eps, sig, rc = 1.0, 1.0, 4.0
        r = 1.5

        # analytic unshifted
        inv_r  = sig / r
        e_raw  = 4 * eps * (inv_r**12 - inv_r**6)

        # shift terms
        inv_rc  = sig / rc
        e_shift = 4 * eps * (inv_rc**12 - inv_rc**6)
        f_shift = 4 * eps * (-12*inv_rc**12 + 6*inv_rc**6) / rc
        e_analytic = e_raw - e_shift - f_shift * (r - rc)

        pos = np.array([[0.0, 0.0, 0.0],
                        [r,   0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)

        energy, _, _ = lj.compute(Z, pos, cell, pbc, False)
        assert energy == pytest.approx(e_analytic, rel=1e-10)

    def test_energy_at_cutoff_is_zero(self, lj):
        """Force-shifted potential must give E≈0 at r = r_cut."""
        rc = 4.0
        pos = np.array([[0.0, 0.0, 0.0],
                        [rc,  0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)
        energy, _, _ = lj.compute(Z, pos, cell, pbc, False)
        assert energy == pytest.approx(0.0, abs=1e-12)

    def test_forces_at_cutoff_are_zero(self, lj):
        """Force-shifted potential must give F≈0 at r = r_cut."""
        rc = 4.0
        pos = np.array([[0.0, 0.0, 0.0],
                        [rc,  0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        np.testing.assert_allclose(forces, np.zeros((2, 3)), atol=1e-12)

    # ── Newton's third law ────────────────────────────────────────────────────

    def test_newton_third_law_open(self, lj, two_atoms_far):
        """Sum of forces must vanish for any configuration."""
        Z, pos, cell, pbc = two_atoms_far
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-14)

    def test_newton_third_law_periodic(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-12)

    def test_newton_third_law_cluster(self, lj, argon_cluster):
        Z, pos, cell, pbc = argon_cluster
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-12)

    # ── Force–energy consistency (finite differences) ─────────────────────────

    def test_forces_match_finite_difference(self, lj):
        """F = -∇E verified by central finite differences on each Cartesian DOF.
        Uses a geometry well within r_cut=4.0 for the custom lj fixture (sigma=1.0).
        """
        # Two atoms at r=1.5σ, comfortably inside r_cut=4.0
        pos  = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)
        dx   = 1e-5

        _, forces, _ = lj.compute(Z, pos.copy(), cell, pbc, False)

        for atom in range(2):
            for coord in range(3):
                p_fwd = pos.copy(); p_fwd[atom, coord] += dx
                p_bwd = pos.copy(); p_bwd[atom, coord] -= dx
                e_fwd, _, _ = lj.compute(Z, p_fwd, cell, pbc, False)
                e_bwd, _, _ = lj.compute(Z, p_bwd, cell, pbc, False)
                fd = -(e_fwd - e_bwd) / (2 * dx)
                assert forces[atom, coord] == pytest.approx(fd, rel=1e-5, abs=1e-10)

    def test_forces_match_finite_difference_periodic(self, argon_fcc):
        """Same finite-difference check for a periodic system with Argon params."""
        lj = LJBackend()   # default Argon: sigma=3.40, r_cut=8.50
        Z, pos, cell, pbc = argon_fcc
        dx = 1e-5

        _, forces, _ = lj.compute(Z, pos.copy(), cell, pbc, False)
        for atom in [0, 5, 10]:
            for coord in range(3):
                p_fwd = pos.copy(); p_fwd[atom, coord] += dx
                p_bwd = pos.copy(); p_bwd[atom, coord] -= dx
                e_fwd, _, _ = lj.compute(Z, p_fwd, cell, pbc, False)
                e_bwd, _, _ = lj.compute(Z, p_bwd, cell, pbc, False)
                fd = -(e_fwd - e_bwd) / (2 * dx)
                assert forces[atom, coord] == pytest.approx(fd, rel=1e-5, abs=1e-10)

    # ── PBC vs open boundary ──────────────────────────────────────────────────

    def test_pbc_differs_from_open_boundary(self, lj, argon_fcc):
        """A periodic system should give a different energy than the same
        cluster with pbc=False (images add interactions)."""
        Z, pos, cell, _ = argon_fcc
        pbc_on  = np.array([True,  True,  True])
        pbc_off = np.array([False, False, False])
        e_pbc,  _, _ = lj.compute(Z, pos, cell, pbc_on,  False)
        e_open, _, _ = lj.compute(Z, pos, cell, pbc_off, False)
        assert e_pbc != pytest.approx(e_open, rel=1e-6)

    # ── Stress tensor properties ──────────────────────────────────────────────

    def test_stress_is_symmetric(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        _, _, stress = lj.compute(Z, pos, cell, pbc, compute_stress=True)
        np.testing.assert_allclose(stress, stress.T, atol=1e-12)

    def test_stress_nonzero_for_periodic(self, lj, argon_fcc):
        Z, pos, cell, pbc = argon_fcc
        _, _, stress = lj.compute(Z, pos, cell, pbc, compute_stress=True)
        assert not np.allclose(stress, np.zeros((3, 3)))

    def test_stress_zero_for_single_pair_open(self, lj, two_atoms_far):
        """Stress is not requested → must be exactly zero."""
        Z, pos, cell, pbc = two_atoms_far
        _, _, stress = lj.compute(Z, pos, cell, pbc, compute_stress=False)
        np.testing.assert_array_equal(stress, np.zeros((3, 3)))

    # ── Energy monotonicity near equilibrium ─────────────────────────────────

    def test_energy_minimum_near_r_min(self, lj):
        """LJ minimum is at r = 2^(1/6) σ.  Check E(r_min) < E(r_min ± δ)."""
        r_min = 2**(1/6)   # in units of σ=1
        delta = 0.05
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)

        def E(r):
            pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            e, _, _ = lj.compute(Z, pos, cell, pbc, False)
            return e

        assert E(r_min) < E(r_min - delta)
        assert E(r_min) < E(r_min + delta)

    # ── Two-atom force direction ──────────────────────────────────────────────

    def test_force_direction_repulsive(self, lj):
        """At r < r_min = 2^(1/6)σ ≈ 1.122 atoms repel:
        force on atom 0 (at origin) points away from atom 1 (at +x), i.e. F[0,0] < 0."""
        r = 0.9   # < r_min for sigma=1.0
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        assert forces[0, 0] < 0.0   # atom 0 pushed in -x (away from atom 1)

    def test_force_direction_attractive(self, lj):
        """At r > r_min = 2^(1/6)σ ≈ 1.122 atoms attract:
        force on atom 0 (at origin) points toward atom 1 (at +x), i.e. F[0,0] > 0."""
        r = 1.5   # > r_min for sigma=1.0
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
        cell = np.eye(3) * 100.0
        pbc  = np.array([False, False, False])
        Z    = np.array([1, 1], dtype=np.int32)
        _, forces, _ = lj.compute(Z, pos, cell, pbc, False)
        assert forces[0, 0] > 0.0   # atom 0 pulled in +x (toward atom 1)


# ── _image_vectors ────────────────────────────────────────────────────────────

class TestImageVectors:
    def test_all_pbc_gives_27_images(self):
        cell = np.eye(3) * 5.0
        pbc  = np.array([True, True, True])
        imgs = _image_vectors(cell, pbc)
        assert imgs.shape == (27, 3)

    def test_no_pbc_gives_1_image(self):
        cell = np.eye(3) * 5.0
        pbc  = np.array([False, False, False])
        imgs = _image_vectors(cell, pbc)
        assert imgs.shape == (1, 3)
        np.testing.assert_array_equal(imgs[0], [0.0, 0.0, 0.0])

    def test_one_periodic_axis_gives_3_images(self):
        cell = np.eye(3) * 5.0
        pbc  = np.array([True, False, False])
        imgs = _image_vectors(cell, pbc)
        assert imgs.shape == (3, 3)

    def test_two_periodic_axes_gives_9_images(self):
        cell = np.eye(3) * 5.0
        pbc  = np.array([True, True, False])
        imgs = _image_vectors(cell, pbc)
        assert imgs.shape == (9, 3)

    def test_image_vectors_scale_with_cell(self):
        """The ±1 image should be exactly one cell vector away."""
        L = 7.3
        cell = np.eye(3) * L
        pbc  = np.array([True, False, False])
        imgs = _image_vectors(cell, pbc)
        xs = sorted(set(imgs[:, 0]))
        assert xs == pytest.approx([-L, 0.0, L])


# ── _ASEComputeMixin: unit conversion and unset charge/spin ───────────────────

class TestASEMixin:
    ase = pytest.importorskip("ase")

    def _backend(self, e_to_ev=1.0, l_to_ang=1.0):
        from ase.calculators.lj import LennardJones
        from fmlip_relay.backends._ase_base import _ASEComputeMixin

        class _LJ(_ASEComputeMixin):
            def __init__(self):
                self._calc = LennardJones(epsilon=1.0, sigma=1.0, rc=3.0)

        b = _LJ()
        b._e_to_ev = e_to_ev
        b._l_to_ang = l_to_ang
        return b

    def _args(self, r):
        Z   = np.array([18, 18], dtype=np.int32)
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        return Z, pos, np.eye(3) * 50.0, np.array([False] * 3)

    def test_unit_scaling(self):
        """A model in (2 eV, 0.5 Å) units at r = 0.6 Å sees r = 1.2 model units."""
        ref_e, ref_f, _ = self._backend().compute(*self._args(1.2), False, 0, 1)
        e, f, _ = self._backend(e_to_ev=2.0, l_to_ang=0.5).compute(
            *self._args(0.6), False, 0, 1)
        assert e == pytest.approx(2.0 * ref_e)
        np.testing.assert_allclose(f, ref_f * 2.0 / 0.5)

    def test_charge_spin_reach_atoms_info(self):
        """Charge and spin are always set; a stray spin 0 becomes a singlet."""
        seen = {}
        b = self._backend()
        orig = b._calc.calculate

        def spy(atoms=None, *a, **kw):
            seen.update(atoms.info)
            return orig(atoms, *a, **kw)

        b._calc.calculate = spy
        b.compute(*self._args(1.2), False, -1, 0)
        assert seen["charge"] == -1 and seen["spin"] == 1

    def test_atomic_energies_from_calculator(self):
        """A calculator with an ``energies`` result (EMT) is used as is."""
        from ase import Atoms
        from ase.calculators.emt import EMT
        from fmlip_relay.backends._ase_base import _ASEComputeMixin

        class _EMT(_ASEComputeMixin):
            def __init__(self):
                self._calc = EMT()

        Z   = np.array([29, 29, 13], dtype=np.int32)
        pos = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [1.2, 2.2, 0.3]])
        cell, pbc = np.eye(3) * 50.0, np.array([False] * 3)
        b = _EMT()
        b._e_to_ev = 2.0
        e, f, s, e_at = b.compute_atomic(Z, pos, cell, pbc, False, 0, 1)

        ref = Atoms(numbers=Z, positions=pos, cell=cell, pbc=pbc)
        ref.calc = EMT()
        np.testing.assert_allclose(e_at, 2.0 * ref.get_potential_energies())
        assert e_at.sum() == pytest.approx(e)
        e3, f3, _ = b.compute(Z, pos, cell, pbc, False, 0, 1)
        assert e3 == pytest.approx(e)
        np.testing.assert_allclose(f3, f)

    def test_native_hessian_is_used_and_scaled(self):
        """A calculator's get_hessian() wins over finite differences, in any
        of the shapes with 9N² entries, and is converted to eV/Å²."""
        b = self._backend(e_to_ev=2.0, l_to_ang=0.5)
        native = np.arange(36, dtype=np.float64).reshape(6, 2, 3)
        b._calc.get_hessian = lambda atoms: native
        H = b.compute_hessian(*self._args(0.6), 0, 1)
        ref = native.reshape(6, 6) * 2.0 / 0.5**2
        np.testing.assert_allclose(H, 0.5 * (ref + ref.T))

    def test_hessian_falls_back_to_finite_differences(self):
        """No get_hessian, or one that raises: forces are differentiated."""
        b = self._backend()
        ref = b._numerical_hessian(*self._args(1.2), 0, 1)
        np.testing.assert_allclose(b.compute_hessian(*self._args(1.2), 0, 1), ref)

        def broken(atoms):
            raise NotImplementedError
        b._calc.get_hessian = broken
        np.testing.assert_allclose(b.compute_hessian(*self._args(1.2), 0, 1), ref)

    def test_atomic_energies_fallback(self):
        """ASE's LennardJones here stands for any calculator; whatever it
        gives, the result has one value per atom and sums to the total."""
        e, _, _, e_at = self._backend().compute_atomic(*self._args(1.2), False, 0, 1)
        assert e_at.shape == (2,)
        assert e_at.sum() == pytest.approx(e)


# ── custom backend: ASE calculator from a user script ─────────────────────────

class TestCustomBackend:
    ase = pytest.importorskip("ase")

    def test_loads_script_and_computes(self, tmp_path, two_atoms_far):
        script = tmp_path / "ase_custom.py"
        script.write_text(
            "from ase.calculators.lj import LennardJones\n"
            "def create_calculator():\n"
            "    return LennardJones(epsilon=0.0104, sigma=3.40, rc=8.5)\n"
        )
        from fmlip_relay.backends.custom import CustomBackend
        b = CustomBackend(str(script))
        Z, pos, cell, pbc = two_atoms_far
        e, f, s = b.compute(Z, pos, cell, pbc, False, 0, 1)
        assert np.isfinite(e)
        np.testing.assert_allclose(f[0], -f[1])

    def test_missing_factory_rejected(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("x = 1\n")
        from fmlip_relay.backends.custom import CustomBackend
        with pytest.raises(TypeError):
            CustomBackend(str(script))


# ── nequip backend: temp dir is removed when loading fails ────────────────────

def test_nequip_cleans_tmpdir_on_failure(tmp_path, monkeypatch):
    import os, tempfile
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    from fmlip_relay.backends.nequip import NequIPBackend
    with pytest.raises(Exception):
        NequIPBackend("/nonexistent/model.nequip.pt2")
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("fmlip-relay-")]
    monkeypatch.setattr(tempfile, "tempdir", None)


# ── openmm backend: results arrive in relay units (eV, Å) ─────────────────────

class TestOpenMMBackend:
    """An argon cluster from a GROMACS topology must match the LJ backend."""

    TOP = """[ defaults ]
1 2 no 1.0 1.0

[ atomtypes ]
AR 18 39.948 0.0 A 0.340 {eps_kjmol}

[ moleculetype ]
AR 1

[ atoms ]
1 AR 1 AR AR 1 0.0 39.948

[ system ]
Ar4

[ molecules ]
AR 4
"""

    def test_units_match_lj_backend(self, tmp_path):
        pytest.importorskip("openmm")
        from fmlip_relay.backends.openmm import OpenMMBackend

        eps_ev, sigma = 0.0104, 3.40
        pos = np.array([[0.0, 0.0, 0.0], [3.3, 0.0, 0.0],
                        [1.65, 2.86, 0.0], [1.65, 0.953, 2.694]])
        gro = "Ar4\n    4\n"
        for i, (x, y, z) in enumerate(pos / 10.0):
            gro += f"{i+1:5d}{'AR':<5s}{'AR':>5s}{i+1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
        gro += "  10.00000  10.00000  10.00000\n"
        (tmp_path / "ar4.gro").write_text(gro)
        (tmp_path / "ar4.top").write_text(
            self.TOP.format(eps_kjmol=eps_ev * 96.48533212331002))

        args = (np.full(4, 18, dtype=np.int32), pos, np.eye(3) * 100.0,
                np.zeros(3, dtype=bool), False, 0, 1)
        omm = OpenMMBackend("GROMACS", str(tmp_path / "ar4.gro"),
                            str(tmp_path / "ar4.top"), platform_name="reference")
        e_omm, f_omm, _ = omm.compute(*args)
        e_lj, f_lj, _ = LJBackend(epsilon=eps_ev, sigma=sigma, r_cut=500.0).compute(*args)

        assert e_omm == pytest.approx(e_lj, rel=1e-4)
        np.testing.assert_allclose(f_omm, f_lj, rtol=1e-4, atol=1e-7)


# ── per-atom energies ─────────────────────────────────────────────────────────

class TestAtomicEnergies:
    def test_partition_equal_split(self):
        from fmlip_relay.backends.base import partition_energy
        np.testing.assert_allclose(partition_energy(-6.0, 3), [-2.0, -2.0, -2.0])

    def test_partition_spreads_missing_part(self):
        """A total larger than the model's sum (e.g. added dispersion) is
        spread equally, so the result still sums to the total."""
        from fmlip_relay.backends.base import partition_energy
        e_at = partition_energy(-7.0, 3, [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(e_at, [-1.0 - 1/3, -2.0 - 1/3, -3.0 - 1/3])

    def test_partition_ignores_wrong_shape(self):
        from fmlip_relay.backends.base import partition_energy
        np.testing.assert_allclose(partition_energy(-6.0, 3, [-1.0, -2.0]), [-2.0] * 3)

    def test_base_default(self):
        Z, pos = np.ones(4, dtype=np.int32), np.zeros((4, 3))
        e, f, s, e_at = DummyBackend(seed=1).compute_atomic(
            Z, pos, np.eye(3), np.array([False] * 3), False, 0, 1)
        np.testing.assert_allclose(e_at, np.full(4, e / 4))

    def test_lj_decomposition(self):
        """Trimer with one atom far away: it carries no energy, the bonded
        pair shares its energy equally."""
        lj  = LJBackend(epsilon=1.0, sigma=1.0, r_cut=3.0)
        Z   = np.full(3, 18, dtype=np.int32)
        pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [20.0, 0.0, 0.0]])
        args = (Z, pos, np.eye(3) * 100.0, np.array([False] * 3), False)
        e, f, s, e_at = lj.compute_atomic(*args)
        e_ref, f_ref, _ = lj.compute(*args)
        assert e == pytest.approx(e_ref)
        np.testing.assert_allclose(f, f_ref)
        np.testing.assert_allclose(e_at, [e / 2, e / 2, 0.0], atol=1e-14)


# ── Hessian ───────────────────────────────────────────────────────────────────

class TestHessian:
    def test_lj_dimer_matches_analytical_curvature(self):
        """Along the bond the Hessian is +-U''(r), across it +-U'(r)/r."""
        eps, sig, rc, r = 1.0, 1.0, 6.0, 1.3
        lj  = LJBackend(epsilon=eps, sigma=sig, r_cut=rc)
        Z   = np.full(2, 18, dtype=np.int32)
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        H = lj.compute_hessian(Z, pos, np.eye(3) * 100.0, np.array([False] * 3), 0, 1)

        dU  = lambda x: 4 * eps * (-12 * sig**12 / x**13 + 6 * sig**6 / x**7)
        d2U = 4 * eps * (156 * sig**12 / r**14 - 42 * sig**6 / r**8)
        k_par, k_perp = d2U, (dU(r) - dU(rc)) / r      # force-shifted cutoff
        assert H.shape == (6, 6)
        np.testing.assert_allclose(H, H.T)
        assert H[0, 0] == pytest.approx(k_par, rel=1e-3)
        assert H[0, 3] == pytest.approx(-k_par, rel=1e-3)
        assert H[1, 1] == pytest.approx(k_perp, rel=1e-3)
        assert H[1, 4] == pytest.approx(-k_perp, rel=1e-3)
        np.testing.assert_allclose(H.sum(axis=1), 0.0, atol=1e-8)   # translation

    def test_step_is_configurable(self):
        lj = LJBackend(epsilon=1.0, sigma=1.0, r_cut=6.0)
        args = (np.full(2, 18, dtype=np.int32),
                np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
                np.eye(3) * 100.0, np.array([False] * 3), 0, 1)
        coarse = lj.compute_hessian(*args)
        lj.hessian_step = 1e-4
        fine = lj.compute_hessian(*args)
        assert not np.array_equal(coarse, fine)
        np.testing.assert_allclose(coarse, fine, rtol=1e-2, atol=1e-6)


# ── stress sign convention: σ = (1/V) dE/dε, as in ASE ────────────────────────

class TestStressConvention:
    def test_lj_stress_is_strain_derivative_of_energy(self):
        lj   = LJBackend(epsilon=1.0, sigma=1.0, r_cut=2.5)
        rng  = np.random.default_rng(3)
        a    = 1.5                                   # compressed: positive pressure
        frac = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        cell = np.eye(3) * a * 2
        pos  = np.concatenate([(frac + s) * a for s in np.ndindex(2, 2, 2)])
        pos  = pos + rng.normal(scale=0.02, size=pos.shape)
        Z    = np.full(len(pos), 18, dtype=np.int32)
        pbc  = np.array([True] * 3)
        _, _, stress = lj.compute(Z, pos, cell, pbc, True)

        vol, h = abs(np.linalg.det(cell)), 1e-5
        for i, j in [(0, 0), (1, 1), (0, 1)]:
            def energy(eps):
                strain = np.eye(3)
                strain[i, j] += eps / 2
                strain[j, i] += eps / 2
                return lj.compute(Z, pos @ strain, cell @ strain, pbc, False)[0]
            numerical = (energy(h) - energy(-h)) / (2 * h) / vol
            assert stress[i, j] == pytest.approx(numerical, rel=1e-5, abs=1e-8)
        assert np.trace(stress) < 0                  # compressed solid
