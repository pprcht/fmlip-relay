"""
fmlip_relay.backends.allegro
----------------------------
Backend for Allegro models compiled with ``nequip-compile``.

Allegro models load through NequIP's ASE calculator once the
``nequip-allegro`` package is installed, so this is the NequIP backend
under a different name.

Install requirements:
    pip install nequip nequip-allegro ase

Usage (via CLI):
    fmlip-relay-server --port 54321 --backend allegro \\
        --model /path/to/model.nequip.pt2 \\
        [--device cpu|cuda|cuda:0]
"""

from __future__ import annotations

from .nequip import NequIPBackend


class AllegroBackend(NequIPBackend):
    _label = "allegro"
