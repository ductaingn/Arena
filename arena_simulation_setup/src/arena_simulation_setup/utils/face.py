"""The front of a model from its annotated `face`."""

from __future__ import annotations

import math

from arena_assets.impl import Face


def _front(face: Face) -> float:
    return math.remainder(math.radians(face.angle + 180.0), math.tau)


def front_angle(face: str | float | None) -> float | None:
    """Angle in rad of an annotated `face` (an arena_assets `Face` value or a legacy numeric angle) in the model frame, None when it names no face."""
    if isinstance(face, str):
        try:
            return _front(Face(face))
        except ValueError:
            try:
                face = float(face)
            except ValueError:
                return None
    if isinstance(face, (int, float)) and not isinstance(face, bool):
        return next((_front(f) for f in Face if f.angle == float(face)), None)
    return None


__all__ = ["front_angle"]
