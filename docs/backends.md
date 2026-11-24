# Backends

A backend turns a structure into energy, forces, and stress.
It is selected with `--backend NAME` and loads its model once, when the server starts.

| Backend | `--backend` | Needs | Model selection |
|---------|-------------|-------|-----------------|
| Dummy | `dummy` | | none; returns random numbers |
| Lennard-Jones | `lj` | | `--lj-*` parameters |
| MACE | `mace` | `mace-torch`, `ase` | `--model` file |
| MACE-MP | `mace_mp` | `mace-torch`, `ase` | `--mace-model` variant |
| MACE-OFF | `mace_off` | `mace-torch`, `ase` | `--mace-model` variant |
| MACE-POLAR | `mace_polar` | `mace-torch>=0.3.16`, `ase` | `--model` key, path, or URL |
| UMA | `uma` | `fairchem-core`, `ase` | `--uma-model`, `--uma-task` |
| NequIP | `nequip` | `nequip`, `ase` | `--model` file |
| Allegro | `allegro` | `nequip`, `nequip-allegro`, `ase` | `--model` file |
| DeepMD | `deepmd` | `deepmd-kit`, `ase` | `--model` file |
| SchNet | `schnet` | `schnetpack`, `ase` | `--model` file |
| OpenMM | `openmm` | `openmm` | `--omm-*` inputs |
| Custom | `custom` | `ase` | `--model` script |

Run `fmlip-relay-check` to see which of these are usable in the current environment.

## Shared options

`--device`
: PyTorch device string (`cpu`, `cuda`, `cuda:0`), default `cpu`.
  For `openmm` it names the platform instead (`cpu`, `cuda`, `opencl`, `reference`).

`--dtype float32|float64`
: Floating-point precision for `mace`, `mace_mp`, `mace_off`, `mace_polar`, `deepmd`, and `schnet`.
  The default is `float64` for model files and `float32` for the MACE foundation models.
  UMA precision is fixed by fairchem.

`--e-to-ev`, `--l-to-ang`
: Unit conversion for models trained in other units, see [Server](server.md#models-in-other-units).
  They apply to the ASE-based backends, i.e. all except `dummy`, `lj`, and `openmm`.

## Test backends

### `dummy`

Returns random energies, forces, and stresses of the right shape.
It has no dependencies and exists to validate the plumbing between Fortran and Python.

### `lj`

A Lennard-Jones potential in pure NumPy with a force-shifted cutoff, so energy and force both go to zero continuously at the cutoff.
Periodic systems are handled via the minimum-image convention.
All atoms interact through the same parameters, regardless of element.

| Flag | Default | Description |
|------|---------|-------------|
| `--lj-epsilon` | `0.0104` eV | Well depth ε (argon) |
| `--lj-sigma` | `3.40` Å | Zero-crossing distance σ |
| `--lj-cutoff` | `8.50` Å | Cutoff radius, about 2.5 σ |

## MACE family

All four backends come with the `mace` extra.

### `mace`

A custom-trained model loaded from a `.model` file:

```bash
fmlip-relay-server --port 54321 --backend mace --model /path/to/model.model \
    --device cuda --dtype float64
```

### `mace_mp`

The MACE-MP foundation models for materials (89 elements).
Checkpoints are downloaded on first use and cached in `~/.cache/mace`.

| Flag | Default | Description |
|------|---------|-------------|
| `--mace-model` | `medium-mpa-0` | `small`, `medium`, `large`, `small-0b`, `medium-0b`, `small-0b2`, `medium-0b2`, `medium-0b3`, `medium-mpa-0`, `large-0b2`, `medium-omat-0` |
| `--dispersion` | off | Add a D3 dispersion correction (needs `torch-dftd`) |

### `mace_off`

The MACE-OFF foundation force field for organic molecules (10 elements).

| Flag | Default | Description |
|------|---------|-------------|
| `--mace-model` | `medium` | `small`, `medium`, `large` |

`--dispersion` is not supported here.

### `mace_polar`

The electrostatic MACE-POLAR-1 models.
`--model` takes a model key such as `polar-1-s`, a local path, or a URL.
The model reads total charge and spin multiplicity, so the client should pass both to `mlip_compute`.
A zero external field is applied in every call.

The MACE-POLAR-1 foundation weights are distributed under the Academic Software Licence.

## UMA

Meta FAIR's universal model from the `fairchem` package, installed with the `uma` extra.
One checkpoint covers several chemical domains, and the task head selects which one is active.

| Flag | Default | Description |
|------|---------|-------------|
| `--uma-model` | `uma-s-1p2` | `uma-s-1`, `uma-s-1p1`, `uma-s-1p2`, `uma-m-1` |
| `--uma-task` | `omol` | Domain head, see below |
| `--device` | `cpu` | UMA runs best on `cuda` |

| Task | Domain |
|------|--------|
| `omol` | Molecules; uses charge and spin |
| `omat` | Inorganic bulk materials |
| `omc` | Molecular crystals |
| `oc20` | Heterogeneous catalysis |
| `odac` | Metal-organic frameworks, direct air capture |

With `omol`, the total charge and spin multiplicity passed to `mlip_compute` travel through the protocol into the calculator.

The checkpoints are gated on the Hugging Face Hub: request access on the model page, then run `huggingface-cli login` or set `HF_TOKEN`.
Downloads are cached in `~/.cache/huggingface`.

## Other ML models

### `nequip` and `allegro`

Models compiled with `nequip-compile`, passed via `--model`.
Both backends use the same loader.
PyTorch's inductor cache is redirected to a private temporary directory under `$TMPDIR`, which is removed when the server shuts down.

### `deepmd`

A DeePMD-kit model file.
`--device` accepts `cpu` or `cuda` only, and `--dtype` is mapped to `DP_INTERFACE_PREC` (`float64` to `high`, `float32` to `low`).

### `schnet`

A SchNetPack model file.
The neighbour-list cutoff is fixed at 5 Å.

## OpenMM

Classical force fields evaluated with OpenMM, from AMBER, GROMACS, CHARMM, or OpenCG inputs.

```bash
fmlip-relay-server --port 54321 --backend openmm --device cuda \
    --omm-source AMBER --omm-coord system.inpcrd --omm-top system.prmtop
```

| Flag | Description |
|------|-------------|
| `--omm-source` | Input format: `AMBER`, `GROMACS`, `CHARMM`, `OPENCG` (required) |
| `--omm-coord` | Coordinate file (required); for OpenCG the start or restart PDB |
| `--omm-top` | Topology file: prmtop, top, or psf (required); for OpenCG the fixed reference PDB |
| `--omm-param` | CHARMM parameter, topology, or stream file; repeatable |
| `--omm-include-dir` | GROMACS include directory |
| `--omm-config` | JSON or YAML file with `createSystem` options, or the OpenCG model definition |
| `--omm-prepare-only` | Write an OpenCG start PDB to standard output and exit; needs `--omm-pbc PBC\|NOPBC` |

YAML configuration files need `pyyaml`, and the OpenCG source needs the `OpenCGChromatin` package.

:::{warning}
This backend returns OpenMM's native units, **kJ/mol** for the energy and **kJ/mol/Å** for the forces, not eV.
`--e-to-ev` and `--l-to-ang` have no effect on it, so the conversion is up to the calling code.
:::

## Custom ASE calculators

The `custom` backend wraps any ASE calculator without touching the fmlip-relay sources.
Write a script that defines `create_calculator()`:

```python
# ase_custom.py
from ase.calculators.emt import EMT

def create_calculator():
    return EMT()
```

and point the server at it:

```bash
fmlip-relay-server --port 54321 --backend custom --model /path/to/ase_custom.py
```

Without `--model` the script is looked up as `./ase_custom.py` in the server's working directory.
The function must return an ASE `Calculator` instance; it is called once at startup.

For a backend with its own command-line options, see [Adding a backend](development.md#adding-a-backend).
