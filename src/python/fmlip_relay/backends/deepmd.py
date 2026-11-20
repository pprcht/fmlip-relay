"""
fmlip_relay.backends.deepmd
------------------------
Backend for custom-trained DeepMD models loaded from a local file.

Install requirements:
    pip install ".[deepmd]"

Usage (via CLI):
    fmlip-relay-server --port 54321 --backend deepmd \\
        --model /path/to/model.model \\
        [--device cpu|cuda|cuda:0] \\
        [--dtype float32|float64]
"""

from __future__ import annotations

import os
from ._ase_base import _ASEComputeMixin


class DeepMDBackend(_ASEComputeMixin):
    """
    Wraps ``deepmd.calculators.DeepMDCalculator`` for a custom-trained model.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file produced by DeepMD training.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
    dtype : str
        ``"float64"`` (default) or ``"float32"``.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 dtype:  str = "float64"):

        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device == "cuda":
            pass
        else:
            raise ValueError(
                f"Unsupported DeepMD device '{device}'. "
                "Use 'cpu' or 'cuda'."
            )

        if dtype == "float64":
            os.environ["DP_INTERFACE_PREC"] = "high"
        elif dtype == "float32":
            os.environ["DP_INTERFACE_PREC"] = "low"
        elif dtype is not None:
            raise ValueError(
                f"Unsupported DeepMD dtype '{dtype}'. "
                "Use 'float32' or 'float64'."
            )
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        try:
            from deepmd.calculator import DP
        except ImportError as exc:
            raise ImportError(
                "DeepMD-kit is not installed."
            ) from exc

        self._calc       = DP(model=model_path)
        self._model_path = model_path

    @property
    def name(self) -> str:
        return f"deepmd({self._model_path})"
