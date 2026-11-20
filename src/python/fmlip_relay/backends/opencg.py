"""OpenCGChromatin helpers used by the OpenMM backend."""

from pathlib import Path
from typing import Any
import numpy as np

_REQUIRED_KEYS = {"nucleosome_sequence", "dna_sequence", "dyad_positions"}
_REGULAR_KEYS = {
    "N_nucleosomes",
    "nucleosome_id",
    "linker_length",
    "linker_pattern",
    "positioning_sequence",
}

_ALLOWED_KEYS = _REQUIRED_KEYS | _REGULAR_KEYS | {
    "cg_template",
    "from_CG",
    "initialization",
    "platformProperties",
    "system_options",
    "acetylation",
    "disable_core_enm",
    "ah_cutoff_factor",
}


def validate_model(model: dict[str, Any]) -> None:
    unknown = set(model) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"OpenCG model has unknown keys: {sorted(unknown)}")

    explicit = bool(set(model) & _REQUIRED_KEYS)
    regular = bool(set(model) & _REGULAR_KEYS)

    if explicit and regular:
        raise ValueError("Do not mix explicit and regular-array OpenCG inputs")
    if not explicit and not regular:
        raise ValueError("OpenCG model definition is missing")

    required = _REQUIRED_KEYS if explicit else _REGULAR_KEYS
    missing = required - set(model)
    if missing:
        raise ValueError(f"OpenCG model is missing keys: {sorted(missing)}")


def _build_regular_array(model):
    n = model["N_nucleosomes"]
    nucleosome_id = model["nucleosome_id"]
    linker_length = model["linker_length"]
    pattern = model["linker_pattern"].upper()
    positioning_sequence = model["positioning_sequence"].upper()

    def make_linker(length):
        return (pattern * (length // len(pattern) + 1))[:length]

    half_linker = make_linker(linker_length // 2)
    full_linker = make_linker(linker_length)

    dna_sequence = half_linker
    dna_sequence += (positioning_sequence + full_linker) * (n - 1)
    dna_sequence += positioning_sequence + half_linker

    first_dyad = linker_length // 2 + len(positioning_sequence) // 2
    dyad_positions = [
        first_dyad + i * (len(positioning_sequence) + linker_length)
        for i in range(n)
    ]

    return [nucleosome_id] * n, dna_sequence, dyad_positions


def _apply_acetylation(nucleosome_array, mapping):
    if not isinstance(mapping, dict):
        raise ValueError("OpenCG acetylation must be a mapping")
    if not mapping:
        return
    if not callable(getattr(nucleosome_array, "acetylate_pattern", None)):
        raise RuntimeError("This OpenCGChromatin version lacks acetylate_pattern()")

    normalized = {}
    for raw_index, residues in mapping.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, str)):
            raise ValueError("Acetylation nucleosome indices must be integers")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("Acetylation nucleosome indices must be integers") from exc

        if not 0 <= index < len(nucleosome_array.nucleosome_sequences):
            raise ValueError(f"Acetylation nucleosome index out of range: {index}")
        if index in normalized:
            raise ValueError(f"Duplicate acetylation nucleosome index: {index}")
        if not isinstance(residues, list):
            raise ValueError(f"Acetylation residues for nucleosome {index} must be a list")

        sequence = nucleosome_array.nucleosome_sequences[index]
        for residue in residues:
            if type(residue) is not int or not 0 <= residue < len(sequence):
                raise ValueError(f"Invalid acetylation residue index: {residue!r}")
            if sequence[residue] != "K":
                raise ValueError(
                    f"Nucleosome {index}, residue {residue} is not lysine (K)"
                )
        normalized[index] = residues

    nucleosome_array.acetylate_pattern(normalized)


def resolve_device_indices(platform, properties):
    if properties.get("DeviceIndex") == "all":
        del properties["DeviceIndex"]
        devices = list(platform.getDevices(properties))
        if platform.getName() == "OpenCL":
            index = devices[0]["OpenCLPlatformIndex"]
            properties["OpenCLPlatformIndex"] = index
            devices = [d for d in devices if d["OpenCLPlatformIndex"] == index]
        properties["DeviceIndex"] = ",".join(d["DeviceIndex"] for d in devices)
    return properties


def build_array(model: dict[str, Any]):
    validate_model(model)

    if "N_nucleosomes" in model:
        nucleosome_sequence, dna_sequence, dyad_positions = _build_regular_array(model)
    else:
        nucleosome_sequence = model["nucleosome_sequence"]
        dna_sequence = model["dna_sequence"]
        dyad_positions = model["dyad_positions"]

    try:
        from OpenCGChromatin import NucleosomeArray
    except ImportError as exc:
        raise ImportError(
            "OpenCGChromatin is not installed in this Python environment."
        ) from exc

    NucleosomeArray._array_counter = 0
    nucleosome_array = NucleosomeArray(
        nucleosome_sequence,
        dna_sequence,
        dyad_positions,
        cg_template=model.get("cg_template", "closed"),
        from_CG=model.get("from_CG", True),
    )
    _apply_acetylation(nucleosome_array, model.get("acetylation", {}))
    return nucleosome_array

def platform_properties(model: dict[str, Any]) -> dict[str, str]:
    values = model.get("platformProperties", {})
    if not isinstance(values, dict):
        raise ValueError("platformProperties must be a mapping")

    return {
        str(key): str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in values.items()
    }

def topology_signature(topology):
    return [
        (atom.residue.chain.index, atom.residue.name, atom.name)
        for atom in topology.atoms()
    ]

def prepare_pdb(
    model: dict[str, Any],
    reference_path: str,
    coordinate_path: str,
    platform_name: str,
    periodic: bool,
) -> Path:
    validate_model(model)

    reference_is_none = str(reference_path).strip().upper() == "NONE"
    coordinate_is_none = str(coordinate_path).strip().upper() == "NONE"

    if reference_is_none:
        if not coordinate_is_none:
            raise ValueError("OpenCG coordinates require an explicit reference PDB")

        reference_path = Path("reference.pdb").resolve()
        coordinate_path = Path("initial.pdb").resolve()
    else:
        if coordinate_is_none:
            raise ValueError("OpenCG coordinates must be explicit when a reference PDB is supplied")

        reference_path = Path(reference_path).expanduser().resolve()
        coordinate_path = Path(coordinate_path).expanduser().resolve()

    if reference_is_none:
        import openmm as mm
        from openmm import app, unit

        initialization = model.get("initialization", {})
        if not isinstance(initialization, dict):
            raise ValueError("initialization must be a mapping")

        device = str(platform_name).strip().upper()
        names = {
            "CUDA": "CUDA",
            "OPENCL": "OpenCL",
            "CPU": "CPU",
            "REFERENCE": "Reference",
        }
        if device not in names:
            raise ValueError("OpenMM device must be CPU, CUDA, OPENCL, or REFERENCE")

        platform = mm.Platform.getPlatformByName(names[device])
        properties = (
            {"Precision": "double"}
            if device in {"CUDA", "OPENCL"}
            else {}
        )
        properties.update(platform_properties(model))
        properties = resolve_device_indices(platform, properties)

        unsupported = set(properties) - set(platform.getPropertyNames())
        if unsupported:
            raise ValueError(
                f"OpenMM platform {platform.getName()} does not support "
                f"properties: {sorted(unsupported)}"
            )

        nucleosome_array = build_array(model)
        nucleosome_array.relax(
            steps=int(initialization.get("steps", 100000)),
            periodic=periodic,
            CoMM_remover=initialization.get("CoMM_remover", True),
            platform=platform,
            properties=properties,
        )

        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with reference_path.open("w", encoding="utf-8") as pdbfile:
            app.PDBFile.writeFile(
                nucleosome_array.topology,
                nucleosome_array.relaxed_coords * unit.nanometer,
                pdbfile,
            )

        system = build_system(
            model,
            str(reference_path),
            periodic=periodic,
        )

        pdb = app.PDBFile(str(reference_path))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        virtual_atoms = [
            atom
            for atom in modeller.topology.atoms()
            if system.isVirtualSite(atom.index)
        ]
        modeller.delete(virtual_atoms)

        with coordinate_path.open("w", encoding="utf-8") as pdbfile:
            app.PDBFile.writeFile(
                modeller.topology,
                modeller.positions,
                pdbfile,
            )

    return coordinate_path


def _remove_core_enm(system, topology):
    import openmm as mm
    from itertools import combinations

    cores = [
        sorted(a.index for a in chain.atoms() if a.element.symbol == "Pt")
        for chain in topology.chains()
    ]
    cores = [group for group in cores if group]
    body_of = {
        a: body
        for body, group in enumerate(cores)
        for a in group
    }

    # Remove harmonic bonds within each rigid core.
    for index in reversed(range(system.getNumForces())):
        old = system.getForce(index)
        if not isinstance(old, mm.HarmonicBondForce):
            continue

        new = mm.HarmonicBondForce()
        new.setName(old.getName())
        new.setForceGroup(old.getForceGroup())
        new.setUsesPeriodicBoundaryConditions(
            old.usesPeriodicBoundaryConditions()
        )

        for j in range(old.getNumBonds()):
            a, b, length, k = old.getBondParameters(j)
            if a in body_of and body_of[a] == body_of.get(b):
                continue
            new.addBond(a, b, length, k)

        system.removeForce(index)
        system.addForce(new)

    # Exclude nonbonded pairs within each rigid core.
    for force in system.getForces():
        if not isinstance(force, mm.CustomNonbondedForce):
            continue

        excluded = {
            tuple(sorted(force.getExclusionParticles(j)))
            for j in range(force.getNumExclusions())
        }
        for group in cores:
            for a, b in combinations(group, 2):
                if (a, b) not in excluded:
                    force.addExclusion(a, b)


def build_system(
    model: dict[str, Any],
    reference_path: str,
    periodic: bool,
):
    from openmm import app, unit
    import openmm as mm

    try:
        from OpenCGChromatin import get_system
    except ImportError as exc:
        raise ImportError(
            "OpenCGChromatin is not installed in this Python environment."
        ) from exc

    nucleosome_array = build_array(model)
    pdb = app.PDBFile(reference_path)
    positions = np.asarray(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )

    expected = topology_signature(nucleosome_array.topology)
    actual = topology_signature(pdb.topology)
    if actual != expected:
        diff = next(
            ((i, e, a) for i, (e, a) in enumerate(zip(expected, actual))
             if e != a),
            None,
        )
        raise ValueError(
            f"reference={reference_path}; "
            f"acetylation={model.get('acetylation', {})}; "
            f"atoms(model/PDB)={len(expected)}/{len(actual)}; "
            f"first_difference={diff}"
        )

    system_options = model.get("system_options", {})
    if not isinstance(system_options, dict):
        raise ValueError("system_options must be a mapping")
    if "periodic" in system_options:
        raise ValueError("Do not set periodic in system_options; use PBC/NOPBC")

    system = get_system(
        positions,
        nucleosome_array.topology,
        {nucleosome_array.chain_id: nucleosome_array.globular_indices},
        [nucleosome_array.dyad_positions],
        periodic=periodic,
        **system_options,
    )

    if model.get("disable_core_enm", False):
        _remove_core_enm(system, nucleosome_array.topology)

    factor = model.get("ah_cutoff_factor")
    if factor is not None:
        factor = float(factor)
        for force in system.getForces():
            if not isinstance(force, mm.CustomNonbondedForce):
                continue

            expression = force.getEnergyFunction()
            if "KH_table" not in expression:
                continue

            old_rc = "rc = 3*sigma;"
            if old_rc not in expression:
                raise ValueError("Unexpected OpenCG AH cutoff expression")

            force.setEnergyFunction(
                expression.replace(old_rc, f"rc = {factor}*sigma;")
            )

            nx, ny, _, values = force.getTabulatedFunction(0).getFunctionParameters()
            sigma_max = max(values[nx*ny:2*nx*ny])
            force.setCutoffDistance(factor * sigma_max * unit.nanometer)

    return system
