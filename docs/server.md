# Server

`fmlip-relay-server` is the Python side of the relay.
The Fortran client normally launches it via the command string given to `mlip_init`, but it is an ordinary command-line program and can be started by hand for testing.
`python -m fmlip_relay` is equivalent.

## Server mode

```bash
fmlip-relay-server --port 54321 --backend dummy
fmlip-relay-server --port 54321 --backend lj --lj-epsilon 0.0104 --lj-sigma 3.40 --lj-cutoff 8.50
fmlip-relay-server --port 54321 --backend mace --model /path/to/model.model --device cuda
fmlip-relay-server --port 54321 --backend uma --uma-task omol --device cuda
```

The server loads the backend, binds the port on `127.0.0.1`, and prints `READY <port>` to standard output.
It then serves one client at a time until it receives `QUIT`.
Log messages go to standard error; the client scans standard output for the `READY` line only.

The server only listens on the loopback interface and performs no authentication.
Client and server must run on the same machine.

### Automatic port selection

With `--port 0` the operating system picks a free port.
The actual number is part of the `READY <port>` line, which `mlip_init` parses, so nothing else has to be coordinated.

### Client loss

By default a server whose client disconnects goes back to waiting for a new connection.
With `--exit-on-disconnect` it shuts down instead, which keeps crashed Fortran jobs from leaving servers (and their GPU memory) behind.

### Logging

`--loglevel DEBUG|INFO|WARNING|ERROR` sets the verbosity of fmlip-relay's own messages; third-party libraries stay at `WARNING`.
`--log FILE` writes the log to a file in addition to standard error, which is the easiest way to see why a server spawned from Fortran failed to start.

## Single-point test mode

`--test GEOMETRY` loads a structure, runs one evaluation through the selected backend, prints a summary, and exits.
No port and no Fortran code are involved, which makes it the quickest check that a model loads and gives sensible numbers.

```console
$ fmlip-relay-server --backend lj --test ar3.xyz
────────────────────────────────────────────────────────
  fmlip-relay  single-point test
────────────────────────────────────────────────────────
  File         : ar3.xyz
  Backend      : lj(eps=0.0104 eV, sig=3.400 Å, rc=8.500 Å)
  Atoms        : 3
  Charge       : +0  (default)
  Multiplicity : 1  (2S+1, default)
  PBC          : [False, False, False]
────────────────────────────────────────────────────────
  Energy       : -0.021279  eV        (-0.007093 eV/atom)
  |F| max      : 0.001906  eV/Å
  |F| mean     : 0.001742  eV/Å
  |F| rms      : 0.001015  eV/Å
────────────────────────────────────────────────────────
  Wall time    : 1.8 ms
────────────────────────────────────────────────────────

     #  sym            Fx            Fy            Fz         |F|          E_atom
     1   Ar     -0.001000     -0.001000     +0.000000    0.001414       -0.009663
     2   Ar     -0.000752     +0.001752     +0.000000    0.001906       -0.005808
     3   Ar     +0.001752     -0.000752     +0.000000    0.001906       -0.005808
```

The last column lists the per-atom energies in eV.

Geometries are read with ASE when it is installed (XYZ, extended XYZ, CIF, POSCAR, and so on).
Without ASE a built-in parser handles plain XYZ files, so the `dummy` and `lj` backends can be tested on a core install.

Adding `--hessian` also computes the Hessian and prints its lowest and highest eigenvalues, a quick check that the six (or five) translational and rotational modes are near zero at a stationary point.

`--charge Q` and `--spin M` set the total charge and the multiplicity 2S+1 for the evaluation.
If omitted, the values are taken from the geometry file when it carries them, and default to 0 and 1 otherwise.

## Threads and CPU pinning

By default a CPU server uses every core it can see, which is the wrong behaviour when several instances share a node.

`--max-threads N`
: Caps the thread pools of PyTorch, OpenMP, MKL, and OpenBLAS at `N`.

`--cpu-affinity CPUS`
: Pins the server to a set of cores, given as a list such as `0-3`, `0,2,4`, or `0-3,8` (Linux only).
  Unless `--max-threads` is also given, the thread cap defaults to the number of pinned cores.

Four servers on a 16-core node would get `--cpu-affinity 0-3`, `4-7`, `8-11`, and `12-15`.

## Models in other units

The wire protocol is fixed to Å and eV.
For a model trained in other units, ASE-based backends accept two conversion factors:

`--e-to-ev F`
: eV per model energy unit.

`--l-to-ang F`
: Å per model length unit.

Coordinates and cell are converted into model units before the call, and energy, forces, and stress are converted back afterwards.
A model trained in Hartree and Bohr would use `--e-to-ev 27.211386 --l-to-ang 0.529177`.

## Option reference

Generated from the argument parser.

```{eval-rst}
.. argparse::
   :module: fmlip_relay.__main__
   :func: _build_parser
   :prog: fmlip-relay-server
   :nodescription:
```
