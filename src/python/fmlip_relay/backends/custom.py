"""
fmlip_relay.backends.custom
---------------------------
Backend for any ASE calculator built by a user script.

The script must define ``create_calculator()`` returning an ASE calculator.

Usage (via CLI):
    fmlip-relay-server --port 54321 --backend custom --model /path/to/ase_custom.py

Without ``--model`` (or with ``--model none``) the script is taken from
``./ase_custom.py``.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from ._ase_base import _ASEComputeMixin


class CustomBackend(_ASEComputeMixin):
    def __init__(self, script_path=None):
        if script_path is None or script_path.strip().lower() == "none":
            script_path = Path.cwd() / "ase_custom.py"
        else:
            script_path = Path(script_path).expanduser().resolve()

        if not script_path.is_file():
            raise FileNotFoundError(f"Custom ASE script not found: {script_path}")

        spec = spec_from_file_location("_fmlip_user_ase_custom", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load custom calculator: {script_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        factory = getattr(module, "create_calculator", None)
        if not callable(factory):
            raise TypeError("ase_custom.py must define create_calculator().")

        calculator = factory()
        if not isinstance(calculator, BaseCalculator):
            raise TypeError("create_calculator() must return an ASE Calculator.")

        self._calc = calculator
        self._script_path = script_path

    @property
    def name(self) -> str:
        return f"custom({self._script_path})"
