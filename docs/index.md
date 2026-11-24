# fmlip-relay

*Persistent ML potential server for Fortran clients*

```{include} ../README.md
:start-after: </div>
:end-before: "## Documentation"
```

## Where to go next

- [Installation](installation.md) covers the Python package, the backend extras, and the Fortran/C library.
- [Fortran client](fortran.md) documents the routines of `fmlip_relay_client` and how to run several instances in parallel.
- [Server](server.md) describes `fmlip-relay-server`, its single-point test mode, and every command-line option.
- [Backends](backends.md) lists the supported models and what each one needs.
- [Wire protocol](protocol.md) gives the byte layout, in case you want to write a client in another language.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guide

installation.md
fortran.md
server.md
backends.md
troubleshooting.md
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

protocol.md
development.md
api.md
```
