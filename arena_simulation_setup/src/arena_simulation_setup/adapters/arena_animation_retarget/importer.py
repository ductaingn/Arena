"""`arena asset import animation smplx`: not implemented. Convert SMPL-X motion with `ros2 run arena_simulation_setup animation_retarget` instead."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from arena_assets.importers import IMPORTERS, Importer

NOT_IMPLEMENTED = "arena asset import animation smplx is not implemented, run `ros2 run arena_simulation_setup animation_retarget` instead"


class SMPLX(Importer):
    meta = IMPORTERS.meta("animation", "smplx")

    @classmethod
    def default_provider(cls) -> None:
        return None

    def items(self) -> list[str]:
        print(NOT_IMPLEMENTED)
        return []

    def import_item(self, item: str, out: Path) -> None:
        print(NOT_IMPLEMENTED)

    def import_items(self, items: Iterable[str], out: Path) -> dict[str, Path]:
        print(NOT_IMPLEMENTED)
        return {}

    def fetch(self, item: str) -> None:
        print(NOT_IMPLEMENTED)

    def convert(self, item: str, src: Path, out: Path) -> None:
        print(NOT_IMPLEMENTED)

    def attribution(self, item: str) -> None:
        print(NOT_IMPLEMENTED)
