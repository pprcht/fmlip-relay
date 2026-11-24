"""OpenMM backend for fmlip_relay.

OpenMM works in kJ/mol and bar. Results are converted to the relay units
(eV, eV/Å, eV/Å³) before they are returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .base import BackendBase
from .opencg import build_system as build_opencg_system
from .opencg import platform_properties as opencg_platform_properties
from .opencg import validate_model as validate_opencg_model
from .opencg import resolve_device_indices


# CODATA 2018 (exact): eV = 1.602176634e-19 J, N_A = 6.02214076e23 / mol
KJMOL_TO_EV = 1.0 / 96.48533212331002
BAR_TO_EV_PER_ANG3 = 1.0e-25 / 1.602176634e-19


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML files.") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("OpenMM config must use .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError("OpenMM config must contain a mapping/object")

    return data


class OpenMMBackend(BackendBase):
    def __init__(
        self,
        source: str,
        coord_path: str,
        topology_path: str,
        config_path: str = "NOCONFIG",
        parameter_files: list[str] | None = None,
        include_dir: str | None = None,
        platform_name: str | None = None,
    ):
        self._config_dir = Path.cwd()
        self._platform_properties: dict[str, str] = {}

        source = str(source).lower()
        if source not in {"amber", "gromacs", "charmm", "opencg"}:
            raise ValueError(
                "OpenMM source must be AMBER, GROMACS, CHARMM, or OPENCG"
            )

        config_text = str(config_path).strip()

        if source == "opencg":
            if config_text.upper() == "NOCONFIG":
                raise ValueError("OPENCG requires a model file through --omm-config")

            model_path = Path(self._resolve_path(config_text))
            model = load_config(model_path)
            validate_opencg_model(model)

            self._config = {
                "type": "opencg",
                "coord": coord_path,
                "reference": topology_path,
                "model": model,
            }
            self._platform_properties = opencg_platform_properties(model)

        else:
            if source == "amber":
                self._config = {"type": "amber", "coord": coord_path, "prmtop": topology_path}
            elif source == "gromacs":
                self._config = {"type": "gromacs", "coord": coord_path, "top": topology_path}
                if include_dir:
                    self._config["include_dir"] = include_dir
            elif source == "charmm":
                self._config = {
                    "type": "charmm", "coord": coord_path,
                    "psf": topology_path, "parameter_files": parameter_files or []}

            if config_text.upper() != "NOCONFIG":
                path = Path(self._resolve_path(config_text))
                config = load_config(path)

                allowed = {
                    "nonbondedMethod", "nonbondedCutoff", "ewaldErrorTolerance", "switchDistance",
                    "constraints", "rigidWater",
                    "implicitSolvent", "implicitSolventSaltConc", "implicitSolventKappa", "temperature",
                    "soluteDielectric", "solventDielectric", "sasaMethod", "platformProperties",
                }

                unknown = set(config) - allowed
                if unknown:
                    raise ValueError(f"OpenMM config has unknown keys: {sorted(unknown)}")

                platform_properties = config.pop("platformProperties", {})
                if not isinstance(platform_properties, dict):
                    raise ValueError("platformProperties must be a mapping")

                self._platform_properties = {
                    str(key): (
                        str(value).lower()
                        if isinstance(value, bool)
                        else str(value)
                    )
                    for key, value in platform_properties.items()
                }

                self._config["createSystem"] = config

        try:
            import openmm as mm
            from openmm import app, unit
        except ImportError as exc:
            raise ImportError(
                "OpenMM is not installed in the relay Python environment."
            ) from exc

        self._mm = mm
        self._app = app
        self._unit = unit

        self._system = None
        self._context = None
        self._stress_barostat = None
        self._natoms = None
        self._platform_name = platform_name

        self._reference_positions = None
        self._real_indices = None

    @property
    def name(self) -> str:
        return f"openmm({self._config['type']}:{Path(self._config['coord']).name})"

    def compute(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
        compute_stress: bool,
        charge: int,
        spin: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        pbc = np.asarray(pbc, dtype=bool)
        if np.any(pbc) and not np.all(pbc):
            raise ValueError("OpenMM backend supports only full 3D PBC or no PBC")

        request_pbc = bool(np.all(pbc))
        self._ensure_context(request_pbc, cell)

        if len(atomic_numbers) != self._natoms:
            raise ValueError(
                f"OpenMM System has {self._natoms} particles, "
                f"but request contains {len(atomic_numbers)} atoms"
            )

        if request_pbc:
            self._context.setPeriodicBoxVectors(*self._cell_vectors(cell))

        positions = np.asarray(positions, dtype=np.float64)

        if self._config["type"] == "opencg":
            full_positions = self._reference_positions.copy()
            full_positions[self._real_indices] = positions
            self._context.setPositions(full_positions * self._unit.angstrom)
            self._context.computeVirtualSites()
        else:
            self._context.setPositions(positions * self._unit.angstrom)

        state = self._context.getState(getEnergy=True, getForces=True)

        energy = state.getPotentialEnergy().value_in_unit(self._unit.kilojoule_per_mole)
        forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                self._unit.kilojoule_per_mole / self._unit.angstrom
            ),
            dtype=np.float64,
        )

        energy *= KJMOL_TO_EV
        forces *= KJMOL_TO_EV

        if self._config["type"] == "opencg":
            forces = forces[self._real_indices]

        stress = np.zeros((3, 3), dtype=np.float64)
        if compute_stress:
            if not request_pbc:
                raise ValueError("OpenMM stress tensor requires a periodic system")
            stress = self._stress_tensor_bar() * BAR_TO_EV_PER_ANG3

        return energy, forces, stress


    def _ensure_context(self, request_pbc: bool, cell: np.ndarray) -> None:
        if self._context is not None:
            return

        self._system = self._build_system(request_pbc, cell)
        if self._config["type"] == "opencg":
            self._natoms = len(self._real_indices)
        else:
            self._natoms = int(self._system.getNumParticles())
        self._stress_barostat = self._add_stress_barostat(request_pbc)
        self._integrator = self._mm.VerletIntegrator(1.0 * self._unit.femtoseconds)

        device = str(self._platform_name).strip().upper()
        if device == "CUDA":
            platform = self._mm.Platform.getPlatformByName("CUDA")
        elif device == "OPENCL":
            platform = self._mm.Platform.getPlatformByName("OpenCL")
        elif device == "CPU":
            platform = self._mm.Platform.getPlatformByName("CPU")
        elif device == "REFERENCE":
            platform = self._mm.Platform.getPlatformByName("Reference")
        else:
            raise ValueError("OpenMM device must be CPU, CUDA, or REFERENCE")

        properties = self._create_platform_properties(platform)

        self._context = self._mm.Context(
            self._system,
            self._integrator,
            platform,
            properties,
        )

    def _create_platform_properties(self, platform) -> dict[str, str]:
        platform_name = platform.getName().upper()

        if platform_name in {"CUDA", "OPENCL"}:
            properties = {"Precision": "double"}
        else:
            properties = {}

        properties.update(self._platform_properties)
        if self._config["type"] == "opencg":
            properties = resolve_device_indices(platform, properties)

        supported = set(platform.getPropertyNames())
        unsupported = set(properties) - supported
        if unsupported:
            raise ValueError(
                f"OpenMM platform {platform.getName()} does not support "
                f"properties: {sorted(unsupported)}"
            )

        return properties


    def _build_system(self, request_pbc: bool, cell: np.ndarray):
        source = str(self._config.get("type", "")).lower()

        if source == "opencg":
            return self._build_opencg_system(request_pbc)

        kwargs = self._create_system_kwargs(request_pbc)

        if source == "amber":
            return self._build_amber_system(kwargs, request_pbc, cell)
        if source == "gromacs":
            return self._build_gromacs_system(kwargs, request_pbc, cell)
        if source == "charmm":
            return self._build_charmm_system(kwargs, request_pbc, cell)

        raise ValueError("OpenMM config type must be amber, gromacs, charmm, or opencg")

    def _build_amber_system(self, kwargs: dict[str, Any], pbc: bool, cell: np.ndarray):
        prmtop_kwargs = {}
        if pbc:
            prmtop_kwargs["periodicBoxVectors"] = self._cell_vectors(cell)

        prmtop = self._app.AmberPrmtopFile(
            self._resolve_path(self._config["prmtop"]),
            **prmtop_kwargs,
        )
        return prmtop.createSystem(**kwargs)

    def _build_gromacs_system(self, kwargs: dict[str, Any], pbc: bool, cell: np.ndarray):
        top_kwargs: dict[str, Any] = {}
        if pbc:
            top_kwargs["periodicBoxVectors"] = self._cell_vectors(cell)

        include_dir = self._config.get("include_dir")
        if include_dir:
            top_kwargs["includeDir"] = self._resolve_path(include_dir)

        top = self._app.GromacsTopFile(
            self._resolve_path(self._config["top"]),
            **top_kwargs,
        )
        return top.createSystem(**kwargs)

    def _build_charmm_system(self, kwargs: dict[str, Any], pbc: bool, cell: np.ndarray):
        psf_kwargs = {}
        if pbc:
            psf_kwargs["periodicBoxVectors"] = self._cell_vectors(cell)

        psf = self._app.CharmmPsfFile(
            self._resolve_path(self._config["psf"]),
            **psf_kwargs,
        )

        parameter_files = self._as_list(self._config.get("parameter_files"))
        if not parameter_files:
            raise ValueError("CHARMM config requires parameter_files")

        params = self._app.CharmmParameterSet(
            *[self._resolve_path(path) for path in parameter_files]
        )
        return psf.createSystem(params, **kwargs)

    def _build_opencg_system(self, request_pbc: bool):
        reference_path = Path(self._resolve_path(self._config["reference"]))
        if not reference_path.is_file():
            raise FileNotFoundError(f"OpenCG reference PDB does not exist: {reference_path}.")

        system = build_opencg_system(
            self._config["model"],
            str(reference_path),
            request_pbc,
        )

        pdb = self._app.PDBFile(str(reference_path))
        self._reference_positions = np.asarray(
            pdb.getPositions(asNumpy=True).value_in_unit(self._unit.angstrom),
            dtype=np.float64,
        )
        self._real_indices = np.array(
            [
                index
                for index in range(system.getNumParticles())
                if not system.isVirtualSite(index)
            ],
            dtype=np.int64,
        )

        return system

    def _create_system_kwargs(self, request_pbc: bool) -> dict[str, Any]:
        opts = dict(self._config.get("createSystem", {}))
        source = str(self._config.get("type", "")).lower()

        method_name = str(
            opts.get("nonbondedMethod", "PME" if request_pbc else "NoCutoff",)).strip()

        periodic_methods = {"PME", "Ewald", "CutoffPeriodic", "LJPME",}
        nonperiodic_methods = {"NoCutoff", "CutoffNonPeriodic",}

        if request_pbc and method_name not in periodic_methods:
            raise ValueError(
                "Periodic request requires a periodic " f"nonbondedMethod, got {method_name}"
            )

        if not request_pbc and method_name not in nonperiodic_methods:
            raise ValueError(
                "Non-periodic request requires a non-periodic " f"nonbondedMethod, got {method_name}"
            )

        implicit_name = opts.get("implicitSolvent")
        if isinstance(implicit_name, str):
            implicit_name = implicit_name.strip()
            if implicit_name.lower() in {"none", "null"}:
                implicit_name = None

        if implicit_name is not None:
            allowed_implicit = {"HCT", "OBC1", "OBC2", "GBn", "GBn2"}
            if implicit_name not in allowed_implicit:
                raise ValueError(f"Unknown implicitSolvent: {implicit_name}")

            if source == "gromacs":
                raise ValueError("GROMACS source does not support implicitSolvent")

            allowed_gb_methods = {"NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic"}
            if method_name not in allowed_gb_methods:
                raise ValueError(
                    "Implicit solvent supports only NoCutoff, CutoffNonPeriodic, or CutoffPeriodic"
                )

        opts["nonbondedMethod"] = getattr(self._app, method_name)
        opts["nonbondedCutoff"] = (float(opts.get("nonbondedCutoff", 1.0)) * self._unit.nanometer)
        opts["ewaldErrorTolerance"] = float(opts.get("ewaldErrorTolerance", 0.0005))
        switch_distance = opts.get("switchDistance")
        if switch_distance is None:
            opts.pop("switchDistance", None)
        else:
            opts["switchDistance"] = float(switch_distance) * self._unit.nanometer

        opts.setdefault("constraints", None)
        opts.setdefault("rigidWater", False)
        opts["removeCMMotion"] = False

        constraints = opts["constraints"]
        if constraints is None or str(constraints).lower() == "none":
            opts["constraints"] = None
        else:
            allowed_constraints = {"HBonds", "AllBonds", "HAngles"}
            if constraints not in allowed_constraints:
                raise ValueError(f"Unknown constraints setting: {constraints}")
            opts["constraints"] = getattr(self._app, constraints)

        implicit_keys = {
            "implicitSolvent", "implicitSolventSaltConc", "implicitSolventKappa", "temperature",
            "soluteDielectric", "solventDielectric", "sasaMethod",
        }

        if implicit_name is None:
            for key in implicit_keys:
                opts.pop(key, None)
        else:
            opts["implicitSolvent"] = getattr(self._app, implicit_name)
            opts["implicitSolventSaltConc"] = (
                float(opts.get("implicitSolventSaltConc", 0.0)) * self._unit.moles / self._unit.liter
            )

            kappa = opts.get("implicitSolventKappa")
            opts["implicitSolventKappa"] = (
                None
                if kappa is None
                else float(kappa) / self._unit.nanometer
            )

            opts["temperature"] = (float(opts.get("temperature", 298.15)) * self._unit.kelvin)
            opts["soluteDielectric"] = float(opts.get("soluteDielectric", 1.0))
            opts["solventDielectric"] = float(opts.get("solventDielectric", 78.5))

            sasa_method = opts.get("sasaMethod", "ACE")
            if sasa_method not in {"ACE", "LCPO", None}:
                raise ValueError(f"Unknown sasaMethod: {sasa_method}")
            opts["sasaMethod"] = sasa_method

        return opts


    def _add_stress_barostat(self, request_pbc: bool):
        if not request_pbc:
            return None

        barostat = self._mm.MonteCarloFlexibleBarostat(
            1.0 * self._unit.bar,
            300.0 * self._unit.kelvin,
            0,
            False,
        )
        self._system.addForce(barostat)
        return barostat


    def _cell_vectors(self, cell: np.ndarray):
        cell = np.asarray(cell, dtype=np.float64)
        return tuple(
            self._mm.Vec3(float(cell[i, 0]), float(cell[i, 1]), float(cell[i, 2]))
            * self._unit.angstrom
            for i in range(3)
        )


    def _stress_tensor_bar(self) -> np.ndarray:
        values = self._quantity_array(
            self._stress_barostat.computeStressTensor(self._context, False),
            self._unit.bar,
        ).reshape(-1)

        xx, yy, zz, xy, xz, yz = values
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)


    def _resolve_path(self, path: str) -> str:
        path_obj = Path(str(path)).expanduser()
        if path_obj.is_absolute():
            return str(path_obj)
        return str((self._config_dir / path_obj).resolve())

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _quantity_array(value: Any, unit_value: Any) -> np.ndarray:
        if hasattr(value, "value_in_unit"):
            return np.asarray(value.value_in_unit(unit_value), dtype=np.float64)
        if isinstance(value, (list, tuple)):
            converted = [
                item.value_in_unit(unit_value) if hasattr(item, "value_in_unit") else item
                for item in value
            ]
            return np.asarray(converted, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)
