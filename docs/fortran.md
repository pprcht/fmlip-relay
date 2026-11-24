# Fortran client

The module `fmlip_relay_client` manages the server processes and the socket connections.
Each *instance* is one Python process plus one TCP connection, addressed by an integer id between 1 and `MLIP_MAX_INSTANCES` (16).

```fortran
use fmlip_relay_client
```

## Minimal usage

```fortran
use fmlip_relay_client

integer  :: ierr, charge, spin
integer  :: pbc(3), atomic_numbers(64)
real(8)  :: pos(3,64), cell(3,3), energy, forces(3,64), stress(3,3)

!>-- spawn the server, blocks until Python prints "READY"
call mlip_init(1, 54321, &
    "fmlip-relay-server --port 54321 --backend mace --model /path/model.model", &
    60, ierr)
if (ierr /= MLIP_OK) stop "mlip_init failed"

!>-- evaluate
pbc = [1, 1, 1]
atomic_numbers = 13   ! aluminium
charge = 0
spin = 1
call mlip_compute(1, 64, atomic_numbers, pos, cell, pbc, 1, charge, spin, &
                  energy, forces, stress, ierr)

!>-- teardown
call mlip_finalize(1, ierr)
```

For molecules and clusters, pass `pbc = [0, 0, 0]` and `compute_stress = 0`.
The cell is unused in that case but should still be a non-degenerate matrix (e.g. a large cubic box), since some backends invert it regardless.

## Conventions

Units
: Positions and cell in Å, energy in eV, forces in eV/Å, stress in eV/Å³.

Array layout
: `positions` and `forces` are dimensioned `(3, natoms)`, i.e. the first index runs over x, y, z.
  The rows of `cell` are the lattice vectors.

Integers
: All integer arguments are default 32-bit `INTEGER`.

Error handling
: Every routine returns `ierr`, which is `MLIP_OK` (0) on success and `MLIP_ERR` (1) otherwise.
  Diagnostics are written to standard output.

## API reference

### `mlip_init`

```fortran
call mlip_init(iid, port, server_cmd, timeout_sec, ierr)
```

Spawns the server process, waits for its `READY` line, and connects.
On failure the half-started instance is cleaned up.

| Argument | Type | Intent | Description |
|----------|------|--------|-------------|
| `iid` | `integer` | in | Instance id, `1..MLIP_MAX_INSTANCES` |
| `port` | `integer` | in | TCP port to connect to; ignored if the server announces its own (see below) |
| `server_cmd` | `character(len=*)` | in | Full shell command that launches the server, at most 1023 characters |
| `timeout_sec` | `integer` | in | Seconds to wait for `READY` |
| `ierr` | `integer` | out | Status |

The command is free-form, so any [server option](server.md) can be passed through it.
Loading a large model on a cold filesystem can take a minute or more; set `timeout_sec` accordingly.

If the command contains `--port 0`, the operating system assigns a free port, the server announces it as `READY <port>`, and `mlip_init` connects to the announced port.
The `port` argument is then ignored.
This avoids port collisions between independent jobs on the same node.

### `mlip_compute`

```fortran
call mlip_compute(iid, natoms, atomic_numbers, positions, cell, pbc, compute_stress, &
                  charge, spin, energy, forces, stress, ierr [, atomic_energies])
```

Sends one structure and receives energy, forces, and stress.

| Argument | Type | Intent | Description |
|----------|------|--------|-------------|
| `iid` | `integer` | in | Instance id |
| `natoms` | `integer` | in | Number of atoms |
| `atomic_numbers` | `integer(natoms)` | in | Atomic number Z of each atom |
| `positions` | `real(8)(3,natoms)` | in | Cartesian coordinates in Å |
| `cell` | `real(8)(3,3)` | in | Lattice vectors as rows, in Å |
| `pbc` | `integer(3)` | in | 1 for a periodic axis, 0 otherwise |
| `compute_stress` | `integer` | in | 1 to request the stress tensor |
| `charge` | `integer`, optional | in | Total charge, default 0 |
| `spin` | `integer`, optional | in | Spin multiplicity 2S+1, default 1 |
| `energy` | `real(8)` | out | Potential energy in eV |
| `forces` | `real(8)(3,natoms)` | out | Forces in eV/Å |
| `stress` | `real(8)(3,3)` | out | Stress tensor in eV/Å³, zero if not requested |
| `ierr` | `integer` | out | Status |
| `atomic_energies` | `real(8)(natoms)`, optional | out | Per-atom energies in eV |

`charge` and `spin` sit in the middle of the argument list.
To omit them, pass the output arguments by keyword:

```fortran
call mlip_compute(1, natoms, atomic_numbers, pos, cell, pbc, 1, &
                  energy=energy, forces=forces, stress=stress, ierr=ierr)
```

Charge and spin only affect backends that model them (UMA with the `omol` task, MACE-POLAR); all others ignore the values.

#### Per-atom energies

Passing the optional `atomic_energies` array requests a decomposition of the energy over the atoms:

```fortran
real(8) :: e_atom(natoms)

call mlip_compute(1, natoms, atomic_numbers, pos, cell, pbc, 0, &
                  energy=energy, forces=forces, stress=stress, ierr=ierr, &
                  atomic_energies=e_atom)
```

The values always sum to `energy`.
Models with a per-atom decomposition (the MACE family, `lj`, and `custom` calculators that provide ASE's `energies`) return it; all other backends return `energy/natoms` for every atom, see [Backends](backends.md#per-atom-energies).
The request uses a separate message type, so calls without the argument are unchanged, and a server too old to know it makes the call return `MLIP_ERR`.

#### Errors

A backend exception on the Python side does not kill the server.
The call returns `ierr = MLIP_ERR` with zeroed outputs, and the instance stays usable for the next structure.

### `mlip_hessian`

```fortran
call mlip_hessian(iid, natoms, atomic_numbers, positions, cell, pbc, hessian, ierr [, charge, spin])
```

Returns the second derivatives of the energy with respect to the atomic positions.

| Argument | Type | Intent | Description |
|----------|------|--------|-------------|
| `iid` | `integer` | in | Instance id |
| `natoms` | `integer` | in | Number of atoms |
| `atomic_numbers` | `integer(natoms)` | in | Atomic number Z of each atom |
| `positions` | `real(8)(3,natoms)` | in | Cartesian coordinates in Å |
| `cell` | `real(8)(3,3)` | in | Lattice vectors as rows, in Å |
| `pbc` | `integer(3)` | in | 1 for a periodic axis, 0 otherwise |
| `hessian` | `real(8)(3*natoms,3*natoms)` | out | Hessian in eV/Å², symmetric |
| `ierr` | `integer` | out | Status |
| `charge` | `integer`, optional | in | Total charge, default 0 |
| `spin` | `integer`, optional | in | Spin multiplicity 2S+1, default 1 |

Row and column `3*(i-1)+a` of `hessian` belong to Cartesian component `a` of atom `i`.
For periodic systems the cell is held fixed.

```fortran
real(8) :: hess(3*natoms,3*natoms)

call mlip_hessian(1, natoms, atomic_numbers, pos, cell, pbc, hess, ierr)
```

Backends that can differentiate their model return an analytical Hessian.
All others compute it on the server from central finite differences of the forces, which takes 6·`natoms` force evaluations, so a call can run for a long time; see [Backends](backends.md#hessians).
On failure `hessian` is zero and `ierr = MLIP_ERR`, and the instance stays usable.
A server too old to know the request also gives `MLIP_ERR`.

### `mlip_ping`

```fortran
call mlip_ping(iid, ierr)
```

Round trip without a computation, to check that the server is alive.

### `mlip_finalize` and `mlip_finalize_all`

```fortran
call mlip_finalize(iid, ierr)
call mlip_finalize_all(ierr)
```

`mlip_finalize` sends `QUIT`, closes the socket, and terminates the server's process group.
Calling it on an inactive instance is a no-op that returns `MLIP_OK`.
`mlip_finalize_all` does the same for every active instance.

### Constants and helpers

| Symbol | Value | Meaning |
|--------|-------|---------|
| `MLIP_MAX_INSTANCES` | 16 | Largest valid instance id |
| `MLIP_OK` | 0 | Success |
| `MLIP_ERR` | 1 | Failure |
| `get_pid()` | | PID of the calling process, handy for deriving unique port numbers |

## Parallel instances

Each instance owns its socket and only instance-local state, so different instances can be called from different OpenMP threads without locking.
A single instance must not be shared between threads.

```fortran
!>-- start all servers serially, before the parallel region
do i = 1, N
  write(cmd, '("fmlip-relay-server --port ",I0," --backend mace --model ",A)') &
        BASE_PORT+i, trim(model_path)
  call mlip_init(i, BASE_PORT+i, trim(cmd), 120, ierr)
end do

!$omp parallel do num_threads(N) private(iid, E, F, S, ierr) schedule(static,1)
do i = 1, N
  iid = omp_get_thread_num() + 1
  call mlip_compute(iid, natoms, atomic_numbers, pos(:,:,i), cell, pbc, 0, &
                    energy=E, forces=F, stress=S, ierr=ierr)
end do
!$omp end parallel do

call mlip_finalize_all(ierr)
```

Starting the servers serially matters: several processes loading the same checkpoint from a shared filesystem at once can race on download and cache files.

### Choosing ports

With fixed ports, every instance on a node needs its own number.
Under MPI, a scheme such as `port = base_port + mpi_rank * 100 + local_instance` keeps the ranks apart.
The simpler route is `--port 0` in every server command, which leaves the choice to the operating system (see [`mlip_init`](#mlip_init)).

### Sharing CPU cores

Several CPU servers on one node will each try to use all cores.
Give each server a thread cap or a disjoint core set through its command line:

```fortran
write(cmd, '(A,I0,A,I0)') "fmlip-relay-server --port 0 --backend mace_mp" // &
      " --cpu-affinity ", 4*(i-1), "-", 4*i-1
```

`--max-threads` and `--cpu-affinity` are described under [Server](server.md#threads-and-cpu-pinning).

### Orphaned servers

If the Fortran program crashes, `mlip_finalize` never runs and the servers would keep waiting for a new connection.
Add `--exit-on-disconnect` to the server command to make a server exit as soon as its client goes away without sending `QUIT`.
