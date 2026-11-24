"""
fmlip_relay.backends.schnet
------------------------
Backend for custom-trained schnet models loaded from a local file.

Install requirements:
    pip install ".[schnet]"

Usage (via CLI):
    fmlip-relay-server --port 54321 --backend schnet \\
        --model /path/to/model.model \\
        [--device cpu|cuda|cuda:0] \\
        [--dtype float32|float64]
"""

from __future__ import annotations

from ._ase_base import _ASEComputeMixin


class SchNetBackend(_ASEComputeMixin):
    """
    Wraps ``schnetpack.interfaces.SpkCalculator`` for a custom-trained model.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file produced by SchNet training.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
    dtype : str
        ``"float64"`` (default) or ``"float32"``.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 dtype:  str = "float64",
                 l_to_ang: float = 1.0):
        try:
            import torch
            import schnetpack as spk
            import schnetpack.transform as trn
        except ImportError as exc:
            raise ImportError(
                "SchNet is not installed."
            ) from exc

        if dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "float64":
            torch_dtype = torch.float64
        else:
            raise ValueError(
                f"Unsupported SchNet dtype '{dtype}'. Use 'float32' or 'float64'."
            )

        cutoff = 5.0 / float(l_to_ang)
        self._calc = spk.interfaces.SpkCalculator(
            model=model_path,
            neighbor_list=trn.ASENeighborList(cutoff=cutoff),
            energy_key="energy",
            force_key="forces",
            stress_key="stress",
            # Units handled by outer relay.
            energy_unit="eV",
            position_unit="Ang",
            device=device,
            dtype=torch_dtype,
        )
        self._model_path = model_path

    @property
    def name(self) -> str:
        return f"schnet({self._model_path})"
