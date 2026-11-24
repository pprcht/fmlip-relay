# Wire protocol

Client and server exchange fixed-layout binary messages over a TCP connection on `127.0.0.1`.
All values are little-endian; integers are 32-bit and reals are 64-bit IEEE doubles.
*N* denotes the number of atoms.

## Startup handshake

Before any message is exchanged, the server prints one line to its standard output:

```
READY <port>
```

The client reads the server's standard output until this line appears and then connects to the announced port.

## Message types

Every request starts with an `int32` message type.

| Value | Name | Request payload | Response |
|-------|------|-----------------|----------|
| 1 | `COMPUTE` | Structure, see below | Status, energy, forces, stress |
| 2 | `QUIT` | none | none; the server shuts down |
| 3 | `PING` | none | A single `int32` status (0) |

An unknown message type makes the server drop the connection.

## COMPUTE request (client → server)

```
Offset      Type           Field
────────────────────────────────────────────────────────────
 0          int32          msg_type        (1 = COMPUTE)
 4          int32          natoms          (N > 0)
 8          int32  × N     atomic_numbers  (Z per atom)
 8+4N       float64 × 3N   positions       [atom-major, Å]
 8+28N      float64 × 9    cell            [row-major, Å]
 8+28N+72   int32  × 3     pbc             (1 = periodic, per axis)
 8+28N+84   int32          compute_stress  (0/1)
 8+28N+88   int32          charge          (total charge)
 8+28N+92   int32          spin            (multiplicity 2S+1)
```

The request is 96 + 28*N* bytes long.

## COMPUTE response (server → client)

```
Offset   Type           Field
─────────────────────────────────────────────
 0       int32          status    (0 = OK, 1 = ERROR)
 4       float64        energy    [eV]
12       float64 × 3N   forces    [eV/Å, atom-major]
12+24N   float64 × 9    stress    [eV/Å³, row-major]
```

The response always has the full length of 84 + 24*N* bytes.
On `ERROR` (the backend raised an exception) energy, forces, and stress are zero, and the connection stays open for further requests.

## Array ordering

*Atom-major* means x, y, z of atom 1, then x, y, z of atom 2, and so on.
This is the memory order of a Fortran array `positions(3, natoms)` and of a C-ordered NumPy array of shape `(N, 3)`, so neither side has to transpose.
A Fortran array declared `(natoms, 3)` would send scrambled coordinates.

The cell is sent as three lattice vectors one after the other, which a NumPy reshape to `(3, 3)` turns into a matrix with the vectors as rows.

## Changing the protocol

The layout is hard-coded in three places that must change together: `src/python/fmlip_relay/protocol.py`, `src/fortran/mlip_client.f90`, and this page.
