# Troubleshooting

## `mlip_init` times out waiting for READY

The server never got as far as printing `READY`.

- Check that `fmlip-relay-server` is on the `PATH` of the environment the Fortran program runs in (`which fmlip-relay-server`).
  Under a batch system this is often a different environment than the login shell.
- Run the exact server command by hand, or replace `--port N` with `--test some.xyz` to see whether the model loads at all.
- Add `--loglevel DEBUG --log /tmp/relay.log` to the server command to capture startup errors from Python.
- A large model on a cold filesystem, or a first-time checkpoint download, can exceed a short timeout.
  Increase `timeout_sec`.

## Server exits with an `ImportError`

```
ImportError: MACE is not installed. Run: pip install mace-torch
```

The packages of the selected backend are missing from the environment the server runs in.
From Fortran this shows up as a `READY` timeout.
`fmlip-relay-check` lists the missing packages per backend.

## Failed to connect after READY

After `READY`, the client tries to connect ten times with 100 ms pauses.
A failure at this stage usually means that the port announced or passed to `mlip_init` differs from the one in the server command, or that the server died right after startup.
Using `--port 0` removes the first cause.

## Address already in use

Another process, often a server left over from a crashed run, holds the port.
Use `--port 0` so that the operating system picks a free one, and `--exit-on-disconnect` so that orphaned servers shut themselves down.

## Wrong energies or garbled forces

- `positions` and `forces` must be dimensioned `(3, natoms)` in Fortran, not `(natoms, 3)`.
- The lattice vectors are the rows of `cell`.
- The `openmm` backend returns kJ/mol and kJ/mol/Å instead of eV and eV/Å.
- Both sides assume little-endian byte order, which holds on all x86 and ARM hardware.

## Type mismatch when compiling a `mlip_compute` call

`charge` and `spin` are optional but sit before the output arguments.
A positional call that leaves them out binds `energy` to `charge`.
Either pass both, or use keywords for the outputs (`energy=..., forces=..., stress=..., ierr=...`).

## `mlip_compute` returns `MLIP_ERR`

The backend raised an exception, e.g. for an element the model does not cover.
The server logs the Python traceback to standard error (and to the `--log` file) and keeps running.

## Stress is meaningless for a cluster

Pass `compute_stress = 0` for non-periodic systems.
The stress is undefined without a cell volume, and some calculators raise an error when asked for it.

## Several servers are slower than one

CPU servers each start as many threads as there are cores.
Limit them with `--max-threads` or `--cpu-affinity`, see [Server](server.md#threads-and-cpu-pinning).
