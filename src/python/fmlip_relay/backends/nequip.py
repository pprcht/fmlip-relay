"""
fmlip_relay.backends.nequip
---------------------------
Backend for NequIP models compiled with ``nequip-compile``.

Install requirements:
    pip install nequip ase

Usage (via CLI):
    fmlip-relay-server --port 54321 --backend nequip \\
        --model /path/to/model.nequip.pt2 \\
        [--device cpu|cuda|cuda:0]
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile

from ._ase_base import _ASEComputeMixin


class NequIPBackend(_ASEComputeMixin):
    """
    Wraps ``NequIPCalculator.from_compiled_model`` for a compiled model.

    torch's inductor cache is pointed at a private temporary directory
    (created under ``$TMPDIR``), which ``close()`` removes again.

    Parameters
    ----------
    model_path : str
        Path to the compiled model produced by ``nequip-compile``.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
    """

    _label = "nequip"

    def __init__(self,
                 model_path: str,
                 device: str = "cpu"):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="fmlip-relay-")
        os.environ["TMPDIR"] = self._tmp_dir.name
        tempfile.tempdir = self._tmp_dir.name
        print(f"fmlip-relay: temporary directory {self._tmp_dir.name}",
              file=sys.stderr, flush=True)

        try:
            import torch._inductor.codecache  # noqa: F401  (load before nequip)
            try:
                from nequip.integrations.ase import NequIPCalculator
            except ImportError:
                try:
                    from nequip.ase import NequIPCalculator
                except ImportError as exc:
                    raise ImportError(
                        "NequIP is not installed. Run: pip install nequip"
                    ) from exc

            self._calc = NequIPCalculator.from_compiled_model(
                compile_path=model_path,
                device=device,
            )
            self._model_path = model_path
            self._device = device
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._calc = None
        gc.collect()
        tmp_dir = getattr(self, "_tmp_dir", None)
        if tmp_dir is not None:
            try:
                tmp_dir.cleanup()
            except OSError:
                pass
            self._tmp_dir = None

    @property
    def name(self) -> str:
        return f"{self._label}({self._model_path})"
