from __future__ import annotations

from ..c_radio_v3.config import CRADIOConfig


CRADIO_V4_CONFIGS = {
    "so400m": CRADIOConfig(
        "nvidia/C-RADIOv4-SO400M",
        "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
        1152,
        27,
        16,
        431_237_232,
        mlp_hidden_dim=4304,
        family="v4",
        prefix_tokens=10,
    ),
    "h": CRADIOConfig(
        "nvidia/C-RADIOv4-H",
        "0057b339059c0b9e1b4ba996f975410ebbfdfcc8",
        1280,
        32,
        16,
        651_645_440,
        mlp_hidden_dim=5120,
        family="v4",
        prefix_tokens=10,
    ),
}
