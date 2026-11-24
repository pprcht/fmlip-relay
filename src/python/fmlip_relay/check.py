"""
fmlip_relay.check
-----------------
Environment checker: probes every known backend and reports which ones
are usable in the current Python environment.

Invoked via:
    fmlip-relay-check
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys


# ── colour helpers (no external deps) ────────────────────────────────────────

def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOUR = _supports_colour()
_GREEN  = "\033[32m" if _COLOUR else ""
_RED    = "\033[31m" if _COLOUR else ""
_YELLOW = "\033[33m" if _COLOUR else ""
_BOLD   = "\033[1m"  if _COLOUR else ""
_RESET  = "\033[0m"  if _COLOUR else ""

OK   = f"{_GREEN}✓ ok{_RESET}"
FAIL = f"{_RED}✗ unavailable{_RESET}"
WARN = f"{_YELLOW}~ partial{_RESET}"


# ── dependency table ──────────────────────────────────────────────────────────

# key: (import name, distribution name, note shown under the row)
_DEPS: dict[str, tuple[str, str, str]] = {
    "numpy":      ("numpy",           "numpy",           ""),
    "torch":      ("torch",           "torch",           ""),
    "ase":        ("ase",             "ase",             ""),
    "mace":       ("mace",            "mace-torch",      ""),
    "fairchem":   ("fairchem.core",   "fairchem-core",   ""),
    "nequip":     ("nequip",          "nequip",          ""),
    "allegro":    ("allegro",         "nequip-allegro",  ""),
    "deepmd":     ("deepmd",          "deepmd-kit",      ""),
    "schnetpack": ("schnetpack",      "schnetpack",      ""),
    "openmm":     ("openmm",          "openmm",          ""),
    "torch_dftd": ("torch_dftd",      "torch-dftd",
                   "optional, for --dispersion with mace_mp"),
    "yaml":       ("yaml",            "pyyaml",
                   "optional, for YAML --omm-config files"),
    "opencg":     ("OpenCGChromatin", "OpenCGChromatin",
                   "optional, for --omm-source OPENCG"),
}

_CORE_DEPS = ("numpy",)
_ML_DEPS = ("torch", "ase", "mace", "fairchem", "nequip", "allegro",
            "deepmd", "schnetpack", "openmm")
_OPTIONAL_DEPS = ("torch_dftd", "yaml", "opencg")


def _probe_dep(key: str) -> tuple[bool, str]:
    """Report whether a dependency is installed, with its version.

    The module is located but not imported, so a heavy or broken package
    cannot slow down or crash the check.
    """
    module, dist, _ = _DEPS[key]
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return False, f"No module named '{module}'"
    try:
        return True, f"{dist} {importlib.metadata.version(dist)}"
    except importlib.metadata.PackageNotFoundError:
        return True, f"{dist} (version unknown)"


def _probe_cuda() -> str:
    """Return a 'CUDA available' line; this is the one probe importing torch."""
    try:
        import torch
        return f"CUDA available: {torch.cuda.is_available()}"
    except Exception as e:
        return f"import failed: {e}"


def _probe_mace_polar() -> str:
    """Return an error text if the installed mace-torch lacks mace_polar."""
    try:
        from mace.calculators import mace_polar  # noqa: F401
        return ""
    except Exception:
        return "needs mace-torch >= 0.3.16"


def _probe_backend_import(module: str, cls: str) -> tuple[bool, str]:
    """Try to import a backend class without instantiating it."""
    try:
        mod = importlib.import_module(module)
        getattr(mod, cls)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── backend table ─────────────────────────────────────────────────────────────

# The backends import their packages lazily inside __init__, so importing the
# backend class says nothing about usability. Each entry therefore lists the
# dependencies (keys of _DEPS) the backend needs at construction time.
#
# Each entry: (cli_name, module, class, description, deps, pip extra | None,
#              extra_probe_fn | None)
_BACKENDS = [
    ("dummy",      "fmlip_relay.backends.dummy",    "DummyBackend",     "Random numbers, no deps",           (),                                   None,      None),
    ("lj",         "fmlip_relay.backends.lj",       "LJBackend",        "Lennard-Jones, no deps",            (),                                   None,      None),
    ("mace",       "fmlip_relay.backends.mace",     "MACEBackend",      "Custom MACE model from file",       ("torch", "ase", "mace"),             "mace",    None),
    ("mace_mp",    "fmlip_relay.backends.mace_mp",  "MACEMPBackend",    "MACE-MP foundation (89 elements)",  ("torch", "ase", "mace"),             "mace",    None),
    ("mace_off",   "fmlip_relay.backends.mace_off", "MACEOFFBackend",   "MACE-OFF foundation (organic)",     ("torch", "ase", "mace"),             "mace",    None),
    ("uma",        "fmlip_relay.backends.uma",      "UMABackend",       "UMA universal model (fairchem)",    ("torch", "ase", "fairchem"),         "uma",     None),
    ("mace_polar", "fmlip_relay.backends.mace",     "MACEPolarBackend", "MACE-POLAR-1 electrostatic",        ("torch", "ase", "mace"),             "mace",    _probe_mace_polar),
    ("nequip",     "fmlip_relay.backends.nequip",   "NequIPBackend",    "Compiled NequIP model",             ("torch", "ase", "nequip"),           "nequip",  None),
    ("allegro",    "fmlip_relay.backends.allegro",  "AllegroBackend",   "Compiled Allegro model",            ("torch", "ase", "nequip", "allegro"), "allegro", None),
    ("deepmd",     "fmlip_relay.backends.deepmd",   "DeepMDBackend",    "DeepMD-kit model",                  ("ase", "deepmd"),                    "deepmd",  None),
    ("schnet",     "fmlip_relay.backends.schnet",   "SchNetBackend",    "SchNetPack model",                  ("torch", "ase", "schnetpack"),       "schnet",  None),
    ("openmm",     "fmlip_relay.backends.openmm",   "OpenMMBackend",    "OpenMM classical force field",      ("openmm",),                          "openmm",  None),
    ("custom",     "fmlip_relay.backends.custom",   "CustomBackend",    "ASE calculator from a user script", ("ase",),                             None,      None),
]


# ── report ────────────────────────────────────────────────────────────────────

def run_check() -> int:
    """
    Run all probes and print a formatted report.
    Returns 0 if all required deps are present, 1 otherwise.
    """
    print(f"\n{_BOLD}fmlip-relay environment check{_RESET}")
    print("=" * 52)

    present: dict[str, bool] = {}

    def report(keys) -> None:
        for key in keys:
            ok, info = _probe_dep(key)
            present[key] = ok
            if key == "torch" and ok:
                info += "\n" + _probe_cuda()
            _print_row(_DEPS[key][1], ok, info, note=_DEPS[key][2])

    print(f"\n{_BOLD}Core dependencies{_RESET}")
    report(_CORE_DEPS)
    print(f"\n{_BOLD}ML dependencies{_RESET}")
    report(_ML_DEPS)
    print(f"\n{_BOLD}Optional dependencies{_RESET}")
    report(_OPTIONAL_DEPS)

    # ── backends ──────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Available backends{_RESET}")
    extras: list[str] = []
    for cli_name, module, cls, description, deps, extra, probe in _BACKENDS:
        missing = [_DEPS[d][1] for d in deps if not present[d]]
        if missing:
            ok, err = False, "missing: " + ", ".join(missing)
        else:
            ok, err = _probe_backend_import(module, cls)
            if ok and probe is not None:
                err = probe()
                ok = not err
        if not ok and extra and extra not in extras:
            extras.append(extra)
        _print_backend_row(cli_name, ok, description, err)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    if not present["numpy"]:
        print(f"{_RED}ERROR: numpy is missing — nothing will work.{_RESET}")
        return 1
    if extras:
        print(f"{_YELLOW}Unavailable backends can be installed with e.g.  "
              f"pip install \"fmlip-relay[{extras[0]}]\"")
        print(f"(extras: {', '.join(extras)}){_RESET}")
    if "uma" not in extras:
        print("UMA checkpoints are gated: run 'huggingface-cli login' once.")
    print(f"{_GREEN}Core is functional.{_RESET}")

    return 0


def _print_row(label: str, ok: bool, info: str, note: str = "") -> None:
    status = OK if ok else FAIL
    # indent continuation lines of info
    info_fmt = info.replace("\n", "\n" + " " * 26) if info else ""
    print(f"  {status}  {label:<16} {info_fmt}")
    if note:
        print(f"             {_YELLOW}({note}){_RESET}")


def _print_backend_row(name: str, ok: bool, description: str, err: str) -> None:
    status = OK if ok else FAIL
    print(f"  {status}  --backend {name:<10}  {description}")
    if err:
        # truncate long import error chains to the last line
        last_line = err.strip().splitlines()[-1]
        print(f"             {_RED}{last_line}{_RESET}")


def main() -> None:
    sys.exit(run_check())


if __name__ == "__main__":
    main()
