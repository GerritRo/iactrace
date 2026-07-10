"""Light concentrators and the shared wall-tracing engine.

:class:`~iactrace.camera.optics.concentrator.Concentrator` is the abstract base
(one primitive: ``to_surface``);
:class:`~iactrace.camera.optics.polygonal.PolygonalCone` is the hollow reflective
specialisation that carries the shared :func:`~iactrace.camera.optics.polygonal.trace_chain`
bounce loop, with :class:`~iactrace.camera.optics.winston.WinstonCone` and
:class:`~iactrace.camera.optics.okumura.OkumuraCone` as concrete cones.
"""

from .concentrator import Concentrator
from .okumura import OkumuraCone
from .polygonal import ChainTrace, PolygonalCone, trace_chain
from .winston import WinstonCone

__all__ = [
    "Concentrator",
    "PolygonalCone",
    "WinstonCone",
    "OkumuraCone",
    "trace_chain",
    "ChainTrace",
]
