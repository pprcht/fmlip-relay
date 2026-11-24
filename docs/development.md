# Development

## Layout

The repository spans three languages:

| Path | Content |
|------|---------|
| `src/python/fmlip_relay/` | Server, protocol codec, command-line interface |
| `src/python/fmlip_relay/backends/` | One module per backend, plus the registry |
| `src/fortran/mlip_client.f90` | The client module `fmlip_relay_client` |
| `src/c/socket_utils.c` | POSIX socket and process helpers called from Fortran |
| `example/` | Example driver with single and parallel instances |
| `tests/` | pytest suite for the Python side |
| `docs/` | This documentation |

The server (`server.py`) contains no ML code.
It binds the socket, prints `READY <port>`, and dispatches `COMPUTE`, `PING`, and `QUIT` to whichever backend object it was given.

## Adding a backend

For a one-off ASE calculator, the [`custom` backend](backends.md#custom-ase-calculators) needs no changes to the package.
A built-in backend takes three steps.

**1. Implement the class** in `src/python/fmlip_relay/backends/mybackend.py`.
Subclass `BackendBase` and load the model in `__init__`, which runs once at server startup:

```python
import numpy as np
from .base import BackendBase

class MyBackend(BackendBase):
    def __init__(self, model_path, device="cpu"):
        ...  # load the model here

    def compute(self, atomic_numbers, positions, cell, pbc,
                compute_stress, charge, spin):
        # atomic_numbers : (N,)   int32
        # positions      : (N, 3) float64, Å
        # cell           : (3, 3) float64, Å, lattice vectors as rows
        # pbc            : (3,)   bool
        # compute_stress : bool
        # charge, spin   : int (total charge, multiplicity 2S+1)
        ...
        return energy, forces, stress   # float, (N, 3), (3, 3)
```

If the model already has an ASE calculator, subclass `_ASEComputeMixin` instead and only set `self._calc` in `__init__`; `compute` and the unit conversion are inherited.
`backends/_template_ase_backend.py` is a starting point to copy.

An optional `close()` method is called at shutdown and can release temporary directories or GPU contexts.

**2. Register it** in `src/python/fmlip_relay/backends/__init__.py`, inside `_register_optional()`.
The `try`/`except ImportError` keeps a missing dependency from breaking the other backends:

```python
try:
    from .mybackend import MyBackend
    _REGISTRY["mybackend"] = MyBackend
except ImportError:
    pass
```

The existing backends import their heavy dependencies inside `__init__` of the class, so the module itself always imports and the missing package is reported when the server starts.

**3. Wire up the command line** in `src/python/fmlip_relay/__main__.py`: add an argument group in `_build_parser()` if the backend has its own options, and a branch in `_build_backend()` that maps the parsed arguments to constructor keywords.

Then add a row to `docs/backends.md` and an entry to `_BACKENDS` in `check.py`, listing the packages the backend needs so that `fmlip-relay-check` can report on it.

## Tests

```bash
pip install -e ".[dev]"
pytest
pytest tests/test_server.py::TestPing     # a single class or test
```

The suite needs no ML packages.

| Module | Covers |
|--------|--------|
| `test_protocol.py` | Constants, `recv_exactly` and `send_all` edge cases, request round trips, response layout |
| `test_backends.py` | `DummyBackend` output contract; Lennard-Jones constructor validation, analytical energies, force-shift continuity, finite-difference forces, stress symmetry, periodic images; unit conversion in the ASE mixin; the `custom` backend |
| `test_server.py` | TCP integration: `PING`, sequential `COMPUTE` calls, `QUIT`, backend exceptions, unknown message types, client disconnects, `--exit-on-disconnect`, `--port 0` |

Tests that need `ase` are skipped when it is not installed.

The Fortran client has no unit tests of its own; `run_example dummy` exercises the full path from Fortran through the socket and back.

## Building the documentation

```bash
pip install ".[docs]"
cd docs && make html
```

The result is in `docs/_build/html`.
The package itself must be installed, because the option reference and the API page are generated from the imported code.
On every push to `main`, the `docs.yml` workflow builds the site and deploys it to GitHub Pages.
