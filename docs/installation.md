# Installation

fmlip-relay has two halves that are installed separately: the Python server (a regular pip package) and the Fortran/C client library that your code links against.

## Python package

```bash
pip install .            # core: dummy and Lennard-Jones backends, numpy only
pip install -e ".[dev]"  # editable install with pytest
```

Each ML backend is an optional extra, so the core install stays free of PyTorch:

| Extra | Installs | Backends enabled |
|-------|----------|------------------|
| `mace` | `mace-torch`, `ase` | `mace`, `mace_mp`, `mace_off`, `mace_polar` |
| `uma` | `fairchem-core`, `ase` | `uma` |
| `nequip` | `nequip`, `ase` | `nequip` |
| `allegro` | `nequip`, `nequip-allegro`, `ase` | `allegro` |
| `deepmd` | `deepmd-kit`, `ase` | `deepmd` |
| `schnet` | `schnetpack`, `ase` | `schnet` |
| `openmm` | `openmm`, `pyyaml` | `openmm` |
| `dev` | `pytest`, `pytest-cov` | |
| `docs` | `sphinx`, `shibuya`, `myst-parser`, `sphinx-argparse` | |

```bash
pip install ".[mace]"
pip install ".[uma]"
```

The `custom` backend needs only `ase`, which every ML extra already pulls in.

:::{note}
The extras install the upstream packages with their default settings, which may not match your platform (CUDA version, PyTorch build).
For a working GPU setup, install PyTorch and the model package first by following the upstream instructions ([MACE](https://github.com/ACEsuit/mace), [fairchem/UMA](https://github.com/facebookresearch/fairchem)), then install fmlip-relay on top.
:::

UMA checkpoints are gated on the Hugging Face Hub.
Request access on the model page and authenticate once with `huggingface-cli login`.

### Checking the environment

`fmlip-relay-check` looks up the packages each backend needs and reports which backends are usable:

```console
$ fmlip-relay-check
fmlip-relay environment check
====================================================

Core dependencies
  ✓ ok  numpy          numpy 2.1.0

ML dependencies
  ✓ ok  torch            torch 2.8.0
                          CUDA available: True
  ✓ ok  ase              ase 3.26.0
  ✓ ok  mace-torch       mace-torch 0.3.16
  ✗ unavailable  nequip           No module named 'nequip'
  ...

Available backends
  ✓ ok  --backend dummy       Random numbers, no deps
  ✓ ok  --backend lj          Lennard-Jones, no deps
  ✓ ok  --backend mace        Custom MACE model from file
  ✗ unavailable  --backend nequip      Compiled NequIP model
             missing: nequip
  ...
```

Most backends import their packages only when the server constructs them.
Starting a server with a backend whose packages are missing therefore fails at startup with an `ImportError` naming the package.

## Fortran/C library

The client consists of one Fortran module (`src/fortran/mlip_client.f90`) and one C file with POSIX socket and process helpers (`src/c/socket_utils.c`).
It needs a Fortran and a C compiler and a POSIX system (Linux, macOS).
Three build routes are available and interchangeable.

### CMake

```bash
cmake -S . -B build -DWITH_OpenMP=ON -DBUILD_EXAMPLE=ON
cmake --build build
cmake --install build --prefix ~/.local
```

| Option | Default | Meaning |
|--------|---------|---------|
| `WITH_OpenMP` | `ON` | Enable OpenMP, needed for parallel instances in the example |
| `BUILD_EXAMPLE` | `OFF` | Build the `run_example` executable |

After installation, another CMake project picks the library up via `find_package`:

```cmake
find_package(fmlip_relay REQUIRED)
target_link_libraries(my_code PRIVATE fmlip_relay::fmlip_relay)
```

### Meson

```bash
meson setup _build -Dbuild_example=true
meson compile -C _build
```

The example is built as `_build/example/fmlip-relay-example` here, whereas CMake and the plain Makefile call it `run_example`.

| Option | Default | Meaning |
|--------|---------|---------|
| `openmp` | `true` | Enable OpenMP |
| `static` | `false` | Attempt a fully static binary |
| `build_example` | `false` | Build the example executable |

### Plain Makefile

`example/Makefile` compiles the two source files and the example with `gfortran` and `gcc`, with no build system in between:

```bash
cd example && make
./build/run_example dummy
```

This is also the shortest recipe for vendoring the client: copy `mlip_client.f90` and `socket_utils.c` into your source tree and compile them with the rest of your code.

## Running the example

`run_example` starts one server, runs a ping and a compute call, then starts several servers and evaluates them from an OpenMP parallel region:

```bash
./build/example/run_example dummy
./build/example/run_example lj
./build/example/run_example lj 0.0104 3.40 8.50      # epsilon / eV, sigma / Å, cutoff / Å
./build/example/run_example mace_mp medium-mpa-0
./build/example/run_example mace_off small
./build/example/run_example mace /path/to/model.model
```

The example spawns `fmlip-relay-server` by name, so the Python package must be installed in the active environment.
