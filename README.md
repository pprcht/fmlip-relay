<div align="center">

<h1>fmlip-relay</h1>
<h3><em>Persistent ML potential server for Fortran clients</em></h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-pprcht.github.io%2Ffmlip--relay-blue.svg)](https://pprcht.github.io/fmlip-relay/)
[![CI](https://github.com/pprcht/fmlip-relay/actions/workflows/ci.yml/badge.svg)](https://github.com/pprcht/fmlip-relay/actions/workflows/ci.yml)

</div>

---

**fmlip-relay** lets a Fortran code (MD, MC, geometry optimiser) evaluate machine-learned interatomic potentials that only exist as Python packages.
The Fortran client spawns a Python server per instance and talks to it over a loopback TCP socket.
The model is loaded once at server startup, so every later evaluation costs a socket round trip (about 1 ms) instead of a model reload.

* A small Fortran module (`fmlip_relay_client`) with a handful of routines: init, compute, Hessian, ping, finalize
* Backends for MACE (custom, MP, OFF, POLAR), UMA, NequIP, Allegro, DeepMD, SchNet, OpenMM, and any ASE calculator
* Up to 16 independent server instances, callable from separate OpenMP threads or MPI ranks
* A plain binary wire protocol in Å and eV, with no dependencies on the Fortran side beyond a C compiler

```
┌──────────────────────────────────────────────────────────────┐
│  Fortran driver (MD, MC, geometry optimiser, ...)            │
│  use fmlip_relay_client                                      │
│    mlip_init      → spawn fmlip-relay-server, wait for READY │
│    mlip_compute   → send coords, recv energy/forces/stress   │
│    mlip_ping      → liveness check                           │
│    mlip_finalize  → send QUIT, stop the process              │
└────────────────────────────┬─────────────────────────────────┘
                             │  127.0.0.1:PORT  (binary TCP)
                  ┌──────────┴──────────┐
                  │  fmlip-relay-server │
                  │  server.py          │  ← protocol + event loop
                  │  backends/*.py      │  ← model loaded once at startup
                  └─────────────────────┘
```

## Quick start

Install the Python server and build the Fortran/C client library:

```bash
pip install .                 # core: dummy and Lennard-Jones backends
pip install ".[mace]"         # or another backend extra, e.g. uma, nequip, deepmd

cmake -S . -B build -DBUILD_EXAMPLE=ON
cmake --build build
./build/example/run_example lj        # spawns servers, runs a few evaluations, shuts down
```

Check a backend without any Fortran code:

```bash
fmlip-relay-check                                      # which backends work in this environment
fmlip-relay-server --backend mace_mp --test mol.xyz    # one single-point evaluation, then exit
```

From Fortran, the whole life cycle is three calls:

```fortran
use fmlip_relay_client

integer  :: ierr, pbc(3), atomic_numbers(64)
real(8)  :: pos(3,64), cell(3,3), energy, forces(3,64), stress(3,3)

!>-- spawn the server, blocks until the model is loaded
call mlip_init(1, 54321, &
    "fmlip-relay-server --port 54321 --backend mace --model /path/model.model", &
    60, ierr)

!>-- evaluate as often as needed (Å in, eV, eV/Å, eV/Å³ out)
pbc = [1, 1, 1]
call mlip_compute(1, 64, atomic_numbers, pos, cell, pbc, 1, &
                  energy=energy, forces=forces, stress=stress, ierr=ierr)

!>-- shut the server down
call mlip_finalize(1, ierr)
```

## Documentation

The full documentation lives at **[pprcht.github.io/fmlip-relay](https://pprcht.github.io/fmlip-relay/)**:

| Page | Content |
|------|---------|
| [Installation](https://pprcht.github.io/fmlip-relay/installation.html) | Python package and extras, CMake, Meson, linking into your own code |
| [Fortran client](https://pprcht.github.io/fmlip-relay/fortran.html) | The `fmlip_relay_client` API, parallel instances with OpenMP and MPI |
| [Server](https://pprcht.github.io/fmlip-relay/server.html) | `fmlip-relay-server` modes and the complete option reference |
| [Backends](https://pprcht.github.io/fmlip-relay/backends.html) | Supported models, their options and dependencies |
| [Wire protocol](https://pprcht.github.io/fmlip-relay/protocol.html) | Byte layout of requests and responses |
| [Development](https://pprcht.github.io/fmlip-relay/development.html) | Adding a backend, running the tests, building the docs |
| [Troubleshooting](https://pprcht.github.io/fmlip-relay/troubleshooting.html) | Common failure modes |

The same pages are readable as Markdown in [`docs/`](docs/).

## License

[MIT](LICENSE) © Philipp Pracht
