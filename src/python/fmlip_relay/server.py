"""
fmlip_relay.server
-----------------
Backend-agnostic TCP server loop.

The server:
  1. Binds to 127.0.0.1:<port> and listens for one connection at a time.
     With port 0 the OS picks a free port.
  2. Prints "READY <port>" to stdout once listening – Fortran waits for this
     signal and connects to the announced port.
  3. Dispatches MSG_COMPUTE, MSG_PING, MSG_QUIT in a tight loop.
  4. Delegates all computation to the injected ``BackendBase`` instance.

This module intentionally contains no ML-specific code.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
import traceback

from .backends.base import BackendBase
from .protocol import (
    MSG_COMPUTE, MSG_QUIT, MSG_PING,
    STATUS_OK,
    recv_exactly, send_all,
    read_compute_request,
    write_ok_response, write_error_response,
)

log = logging.getLogger(__name__)


def _handle_compute(conn: socket.socket, backend: BackendBase) -> None:
    req = read_compute_request(conn)

    t0 = time.perf_counter()
    try:
        energy, forces, stress = backend.compute(
            req.atomic_numbers,
            req.positions,
            req.cell,
            req.pbc,
            req.compute_stress,
            req.charge,
            req.spin,
        )
        dt_ms = (time.perf_counter() - t0) * 1e3
        log.debug("COMPUTE natoms=%d  E=%.6f eV  dt=%.2f ms",
                  req.natoms, energy, dt_ms)
        write_ok_response(conn, energy, forces, stress)

    except Exception:
        log.error("Backend raised an exception:\n%s", traceback.format_exc())
        write_error_response(conn, req.natoms)


def _handle_client(conn: socket.socket, backend: BackendBase,
                   exit_on_disconnect: bool = False) -> bool:
    """
    Process messages from one connected client.
    Returns True to keep the server running, False to shut it down.
    With *exit_on_disconnect*, losing the client without a QUIT (disconnect,
    unknown message, unexpected error) also shuts the server down, so a
    server whose Fortran parent has died does not linger.
    """
    keep_alive = not exit_on_disconnect
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            msg_type, = struct.unpack('<i', recv_exactly(conn, 4))

            if msg_type == MSG_COMPUTE:
                _handle_compute(conn, backend)

            elif msg_type == MSG_PING:
                send_all(conn, struct.pack('<i', STATUS_OK))

            elif msg_type == MSG_QUIT:
                log.info("Received QUIT – shutting down.")
                return False

            else:
                log.error("Unknown message type %d – dropping client.", msg_type)
                return keep_alive   # drop this connection

    except ConnectionError as exc:
        log.info("Client disconnected: %s", exc)
        return keep_alive
    except Exception:
        log.error("Unexpected error:\n%s", traceback.format_exc())
        return keep_alive
    finally:
        conn.close()


def run(port: int, backend: BackendBase,
        exit_on_disconnect: bool = False) -> None:
    """
    Start the server, block until a QUIT message is received.

    Parameters
    ----------
    port    : TCP port to bind (loopback only); 0 lets the OS pick one.
    backend : Initialised backend instance (model already loaded).
    exit_on_disconnect : also stop when the client goes away without QUIT.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    actual_port = srv.getsockname()[1]

    log.info("fmlip_relay server listening on 127.0.0.1:%d  backend=%s  PID=%d",
             actual_port, backend.name, os.getpid())

    # Signal Fortran that we are ready, and on which port.
    print(f"READY {actual_port}", flush=True)

    keep_running = True
    while keep_running:
        conn, addr = srv.accept()
        log.debug("Connection from %s", addr)
        keep_running = _handle_client(conn, backend, exit_on_disconnect)

    srv.close()
    log.info("Server stopped.")
