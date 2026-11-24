# Python API

The Python package is mainly used through the `fmlip-relay-server` command, but the pieces below are importable, e.g. for writing a backend or a test client in Python.
The reference is generated from the docstrings.

## Backends

```{eval-rst}
.. autoclass:: fmlip_relay.backends.base.BackendBase
   :members:

.. autofunction:: fmlip_relay.backends.get_backend_class

.. autoclass:: fmlip_relay.backends._ase_base._ASEComputeMixin
```

## Server

```{eval-rst}
.. autofunction:: fmlip_relay.server.run
```

## Protocol

```{eval-rst}
.. autodata:: fmlip_relay.protocol.MSG_COMPUTE
.. autodata:: fmlip_relay.protocol.MSG_QUIT
.. autodata:: fmlip_relay.protocol.MSG_PING
.. autodata:: fmlip_relay.protocol.STATUS_OK
.. autodata:: fmlip_relay.protocol.STATUS_ERROR

.. autoclass:: fmlip_relay.protocol.ComputeRequest

.. autofunction:: fmlip_relay.protocol.read_compute_request
.. autofunction:: fmlip_relay.protocol.write_ok_response
.. autofunction:: fmlip_relay.protocol.write_error_response
.. autofunction:: fmlip_relay.protocol.recv_exactly
.. autofunction:: fmlip_relay.protocol.send_all
```

## Geometry input

```{eval-rst}
.. autofunction:: fmlip_relay.test_mode.read_geometry
```
