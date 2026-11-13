"""
fmlip_relay.__main__
-------------------
CLI entry point.  Installed as the ``fmlip-relay-server`` console script.
Also invokable as ``python -m fmlip_relay``.

Usage
-----
    # Custom-trained MACE model
    fmlip-relay-server --port 54321 --backend mace --model /path/to/model.model

    # MACE-MP foundation model (Materials Project, 89 elements)
    fmlip-relay-server --port 54321 --backend mace_mp
    fmlip-relay-server --port 54321 --backend mace_mp --mace-model large-0b2

    # MACE-OFF foundation model (organic, 10 elements)
    fmlip-relay-server --port 54321 --backend mace_off
    fmlip-relay-server --port 54321 --backend mace_off --mace-model small

    # UMA universal model (Meta FAIR / fairchem) — pick a task head
    fmlip-relay-server --port 54321 --backend uma --uma-task omol
    fmlip-relay-server --port 54321 --backend uma --uma-model uma-m-1 --uma-task omat

    # Lennard-Jones
    fmlip-relay-server --port 54321 --backend lj --lj-epsilon 0.0104

    # Dummy (no model required)
    fmlip-relay-server --port 54321 --backend dummy

    # Single-point test — loads geometry, runs one evaluation, prints summary, exits
    fmlip-relay-server --backend mace_mp --test molecule.xyz
    fmlip-relay-server --backend lj      --test crystal.xyz
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from .backends import get_backend_class
from .backends.mace_mp  import _KNOWN_MODELS as _MACE_MP_MODELS,  _DEFAULT_MODEL as _MACE_MP_DEFAULT
from .backends.mace_off import _KNOWN_MODELS as _MACE_OFF_MODELS, _DEFAULT_MODEL as _MACE_OFF_DEFAULT
from .backends.uma      import (_KNOWN_MODELS as _UMA_MODELS, _DEFAULT_MODEL as _UMA_DEFAULT,
                                _KNOWN_TASKS  as _UMA_TASKS,  _DEFAULT_TASK  as _UMA_TASK_DEFAULT)
from . import server
from . import test_mode


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fmlip-relay-server",
        description="Persistent ML potential server for Fortran clients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument("--port",    type=int, default=None,
                   help="TCP port to listen on (loopback only). 0 lets the OS "
                        "pick a free port, announced as 'READY <port>'. "
                        "Required unless --test is used.")
    p.add_argument("--backend", type=str, required=True,
                   help="Backend name: mace | mace_mp | mace_off | uma | lj | dummy")
    p.add_argument("--exit-on-disconnect", action="store_true", default=False,
                   help="Shut the server down when the client disconnects "
                        "without QUIT (e.g. the Fortran parent crashed), "
                        "instead of waiting for a new connection.")

    # ── test mode ─────────────────────────────────────────────────────────────
    grp_test = p.add_argument_group("Single-point test options  (--test)")
    grp_test.add_argument("--test", type=str, default=None, metavar="GEOMETRY",
                          help="Run a single-point calculation on the given geometry "
                               "file (XYZ / extended-XYZ), print a summary, and exit. "
                               "--port is not required in this mode.")
    grp_test.add_argument("--charge", type=int, default=None, metavar="Q",
                          help="Total charge for the --test evaluation. "
                               "Default: value from the geometry file if present, "
                               "else 0 (neutral). Used by charge-aware backends "
                               "(e.g. UMA omol).")
    grp_test.add_argument("--spin", type=int, default=None, metavar="M",
                          help="Spin multiplicity 2S+1 for the --test evaluation. "
                               "Default: value from the geometry file if present, "
                               "else 1 (ground-state singlet). Used by spin-aware "
                               "backends (e.g. UMA omol).")

    # ── logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log",      type=str, default=None,
                   help="Optional log file path (stderr is always used)")
    p.add_argument("--loglevel", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Log verbosity")

    # ── backend: mace (custom model file) ────────────────────────────────────
    grp_mace = p.add_argument_group(
        "Custom MACE model options  (--backend mace)"
    )
    grp_mace.add_argument("--model", type=str, default=None,
                          help="Path to .model file (required for mace backend)")

    # ── shared neural-network options ─────────────────────────────────────────
    grp_nn = p.add_argument_group(
        "Shared neural-network options  (mace | mace_mp | mace_off | uma)"
    )
    grp_nn.add_argument("--device", type=str.lower, default="cpu",
                        help="PyTorch device string, e.g. cpu | cuda | cuda:0")
    grp_nn.add_argument("--e-to-ev", type=float, default=1.0, metavar="F",
                        help="eV per model energy unit, for models trained in "
                             "other units (ASE-based backends)")
    grp_nn.add_argument("--l-to-ang", type=float, default=1.0, metavar="F",
                        help="Angstrom per model length unit, for models "
                             "trained in other units (ASE-based backends)")
    grp_nn.add_argument("--max-threads", type=int, default=None, metavar="N",
                        help="Cap CPU threads this server may use for inference "
                             "(sets OMP/MKL/BLAS thread env + torch.set_num_threads). "
                             "Default: unset — inherit OMP_NUM_THREADS / all cores. "
                             "Only meaningful on CPU.")
    grp_nn.add_argument("--cpu-affinity", type=str, default=None, metavar="CPUS",
                        help="Pin this server to a fixed set of CPU cores, e.g. "
                             "'0-3' or '0,2,4' or '0-3,8'. Linux only. If "
                             "--max-threads is not also given, the thread cap "
                             "defaults to the number of pinned cores. Use this to "
                             "run several servers on disjoint cores instead of "
                             "letting them fight over all of them.")

    # ── shared MACE precision option ──────────────────────────────────────────
    grp_mace_shared = p.add_argument_group(
        "Shared MACE options  (mace | mace_mp | mace_off)"
    )
    grp_mace_shared.add_argument("--dtype",  type=str, default=None,
                                 choices=["float32", "float64"],
                                 help="Floating-point precision "
                                      "(default: float64 for mace, float32 for foundation models). "
                                      "UMA precision is fixed by fairchem (float32).")

    # ── backend: mace_mp ─────────────────────────────────────────────────────
    grp_mp = p.add_argument_group(
        "MACE-MP options  (--backend mace_mp)  — Materials Project, 89 elements"
    )
    grp_mp.add_argument(
        "--mace-model", type=str, default=None,
        metavar="MODEL",
        help=(f"Foundation model variant. "
              f"mace_mp default: '{_MACE_MP_DEFAULT}', options: "
              f"{', '.join(_MACE_MP_MODELS)}. "
              f"mace_off default: '{_MACE_OFF_DEFAULT}', options: "
              f"{', '.join(_MACE_OFF_MODELS)}.")
    )
    grp_mp.add_argument("--dispersion", action="store_true", default=False,
                        help="Add D3 dispersion correction (mace_mp only; "
                             "requires torch-dftd)")

    # ── backend: uma ─────────────────────────────────────────────────────────
    grp_uma = p.add_argument_group(
        "UMA options  (--backend uma)  — Meta FAIR universal model (fairchem)"
    )
    grp_uma.add_argument(
        "--uma-model", type=str, default=None, metavar="MODEL",
        help=(f"UMA checkpoint. Default: '{_UMA_DEFAULT}', options: "
              f"{', '.join(_UMA_MODELS)}.")
    )
    grp_uma.add_argument(
        "--uma-task", type=str, default=None, metavar="TASK",
        choices=list(_UMA_TASKS),
        help=(f"Task / domain head. Default: '{_UMA_TASK_DEFAULT}', options: "
              f"{', '.join(_UMA_TASKS)}.")
    )

    # ── backend: lj ──────────────────────────────────────────────────────────
    grp_lj = p.add_argument_group("Lennard-Jones options  (--backend lj)")
    grp_lj.add_argument("--lj-epsilon", type=float, default=0.0104,
                        help="Well depth ε in eV")
    grp_lj.add_argument("--lj-sigma",   type=float, default=3.40,
                        help="Zero-crossing σ in Å")
    grp_lj.add_argument("--lj-cutoff",  type=float, default=8.50,
                        help="Cutoff radius in Å")

    return p


def _configure_logging(loglevel: str, logfile: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    # Keep the root logger at WARNING so chatty third-party libraries (httpx,
    # fairchem, urllib3, huggingface_hub, …) don't spam at INFO during model
    # download/load.  Only our own package logs at the requested level; its
    # records still reach the root handlers via propagation.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("fmlip_relay").setLevel(getattr(logging, loglevel))


_terminating = False


def _handle_termination(signum, _frame) -> None:
    """Turn SIGTERM/SIGHUP/SIGINT into SystemExit so ``backend.close()`` runs.

    Further signals are ignored while shutting down, so a second kill from
    the Fortran side does not interrupt the cleanup.
    """
    global _terminating
    if _terminating:
        return
    _terminating = True
    print(f"fmlip-relay: received signal {signum}, shutting down",
          file=sys.stderr, flush=True)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def _limit_threads(n: int) -> None:
    """Cap the CPU thread pools used by torch / numpy / BLAS.

    The OpenMP-style env vars must be set *before* torch (and its BLAS
    backends) are first imported to size their pools; because every backend
    imports torch lazily inside its ``__init__``, calling this just before
    ``_build_backend`` is early enough.  ``torch.set_num_threads`` is also
    called afterwards as a runtime safeguard.
    """
    if n < 1:
        raise ValueError("--max-threads must be >= 1")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def _parse_cpu_list(spec: str) -> list[int]:
    """Parse a CPU spec like ``'0-3,8,10-11'`` into a sorted list of ints."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid CPU range '{part}' (start > end)")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("--cpu-affinity is empty")
    if min(cpus) < 0:
        raise ValueError("CPU indices must be >= 0")
    return sorted(cpus)


def _set_affinity(cpus: list[int]) -> None:
    """Pin the current process to the given CPU cores (Linux only)."""
    if not hasattr(os, "sched_setaffinity"):
        raise ValueError("--cpu-affinity is only supported on Linux")
    ncpu = os.cpu_count() or 0
    bad = [c for c in cpus if c >= ncpu]
    if bad:
        shown = ", ".join(map(str, bad[:8])) + (", …" if len(bad) > 8 else "")
        raise ValueError(
            f"CPU indices [{shown}] exceed available cores (system has {ncpu})"
        )
    os.sched_setaffinity(0, set(cpus))


def _build_backend(args: argparse.Namespace):
    cls     = get_backend_class(args.backend)
    backend = args.backend.lower()
    kwargs: dict = {}

    if backend == "mace":
        if not args.model:
            raise ValueError("--model is required for the mace backend")
        kwargs = dict(
            model_path = args.model,
            device     = args.device,
            dtype      = args.dtype or "float64",
        )

    elif backend == "mace_mp":
        kwargs = dict(
            model      = args.mace_model or _MACE_MP_DEFAULT,
            device     = args.device,
            dtype      = args.dtype or "float32",
            dispersion = args.dispersion,
        )

    elif backend == "mace_off":
        if args.dispersion:
            raise ValueError("--dispersion is not supported for mace_off")
        kwargs = dict(
            model  = args.mace_model or _MACE_OFF_DEFAULT,
            device = args.device,
            dtype  = args.dtype or "float32",
        )

    elif backend == "uma":
        kwargs = dict(
            model  = args.uma_model or _UMA_DEFAULT,
            task   = args.uma_task  or _UMA_TASK_DEFAULT,
            device = args.device,
        )

    elif backend == "lj":
        kwargs = dict(
            epsilon = args.lj_epsilon,
            sigma   = args.lj_sigma,
            r_cut   = args.lj_cutoff,
        )

    return cls(**kwargs)


def main(argv: list[str] | None = None) -> None:
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _handle_termination)

    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.loglevel, args.log)
    log = logging.getLogger(__name__)

    # ── validate port requirement ─────────────────────────────────────────────
    if args.test is None and args.port is None:
        parser.error("--port is required when not using --test")

    # ── pin CPU cores before torch is imported (via the backend) ──────────────
    cpus = None
    if args.cpu_affinity is not None:
        try:
            cpus = _parse_cpu_list(args.cpu_affinity)
            _set_affinity(cpus)
        except ValueError as exc:
            parser.error(str(exc))
        log.info("Pinned to CPU cores: %s.", ",".join(map(str, cpus)))

    # ── cap CPU threads before torch is imported (via the backend) ────────────
    # If cores were pinned but no explicit cap given, default the cap to the
    # number of pinned cores so torch doesn't oversubscribe them.
    thread_cap = args.max_threads
    if thread_cap is None and cpus is not None:
        thread_cap = len(cpus)
    if thread_cap is not None:
        try:
            _limit_threads(thread_cap)
        except ValueError as exc:
            parser.error(str(exc))
        log.info("Capping CPU threads to %d.", thread_cap)

    # ── build backend (shared by both modes) ──────────────────────────────────
    try:
        backend = _build_backend(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    backend._e_to_ev = args.e_to_ev
    backend._l_to_ang = args.l_to_ang

    # torch is now imported (transitively via the backend); enforce the cap at
    # the runtime level too, in case it was already imported elsewhere.
    if thread_cap is not None:
        try:
            import torch
            torch.set_num_threads(thread_cap)
        except ImportError:
            pass

    log.info("Backend '%s' loaded successfully.", backend.name)

    try:
        # ── test mode ─────────────────────────────────────────────────────────
        if args.test is not None:
            sys.exit(test_mode.run_test(backend, args.test,
                                        charge=args.charge, spin=args.spin))

        # ── server mode ───────────────────────────────────────────────────────
        server.run(args.port, backend, exit_on_disconnect=args.exit_on_disconnect)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
