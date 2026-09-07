"""
PRNG fold-in salts.
"""

from __future__ import annotations

# Aperture sampling, drawn inside OpticalElementGroup.transform_to_world.
APERTURE = 0x5A3B1E

# Surface roughness on the primary (stage-0) geometry, drawn once per render.
PRIMARY_ROUGHNESS = 0xB5DF00

# Surface roughness at later stages.
ROUGHNESS = 0xB5DF01

# Per-ray wavelength draws; offset by element index in a render.
WAVELENGTH = 0xC0FFEE
