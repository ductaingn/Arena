"""Pedestrian animation clips: one joint-angle recording per asset directory."""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPaths,
    NetResolver,
    PathView,
)

ANNOTATION = 'annotation.yaml'


@attrs.frozen(eq=False)
class AnimationClip:
    """A clip's arrays and its annotation."""

    name: str
    joint_names: tuple[str, ...]
    angles: np.ndarray  # (T, J) radians, columns follow joint_names
    t: np.ndarray  # (T,) seconds
    root_xy_yaw: np.ndarray  # (T, 3)
    animation_state: np.ndarray  # (T,) Pedestrian.msg state
    meta: dict
    root_quat: np.ndarray | None = None  # (T, 4) xyzw pelvis orientation from the start heading, clips annotated pelvis_tilt

    @property
    def fps(self) -> float:
        return float(self.meta.get('fps', 20.0))

    def frames(self, joints: Sequence[str] = ()) -> list[dict]:
        """Per-frame dicts in the shape the players consume, `joints` the clip lacks filled with 0.0."""
        return [
            {
                't': float(self.t[k]),
                'angles': {**dict.fromkeys(joints, 0.0), **dict(zip(self.joint_names, self.angles[k].tolist(), strict=True))},
                'root_xy_yaw': self.root_xy_yaw[k].tolist(),
                'animation_state': int(self.animation_state[k]),
                **({'root_quat': self.root_quat[k].tolist()} if self.root_quat is not None else {}),
            }
            for k in range(len(self.t))
        ]


def from_start_heading(root_quat: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Pelvis orientations (T, 4 xyzw) given in each frame's heading frame, turned into the first frame's.

    A lying pelvis has no clear heading, so the per-frame heading can jump; the orientation it leaves is smooth.
    """
    half = 0.5 * (np.asarray(yaw, dtype=float) - float(yaw[0]))
    s, c = np.sin(half), np.cos(half)  # yaw quaternion (0, 0, s, c)
    x, y, z, w = np.asarray(root_quat, dtype=float).T
    q = np.stack([c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z], axis=1)
    return q


class AnimationView(PathView):
    """View around a resolved Animation asset directory."""

    @cached_property
    def meta(self) -> dict:
        return yaml.safe_load((self.path / ANNOTATION).read_text()) or {}

    @cached_property
    def clip(self) -> AnimationClip:
        """The clip; its root_quat only when the annotation says the body is pelvis-relative (pelvis_tilt), as clips
        that fold the tilt into the waist and hips carry one too."""
        with np.load(self.path / f'{self.path.name}.npz', allow_pickle=False) as z:
            tilted = bool(self.meta.get('pelvis_tilt')) and 'root_quat' in z.files
            return AnimationClip(
                name=self.path.name,
                joint_names=tuple(str(j) for j in z['joint_names']),
                angles=z['angles'],
                t=z['t'],
                root_xy_yaw=z['root_xy_yaw'],
                animation_state=z['animation_state'],
                meta=self.meta,
                root_quat=from_start_heading(z['root_quat'], z['root_xy_yaw'][:, 2]) if tilted else None,
            )

    def extra(self, filename: str) -> Path:
        """A derived file stored next to the clip."""
        return self.path / filename


@attrs.define(eq=False, hash=False)
class AnimationIdentifier(DomainAssetIdentifier[AnimationView]):
    """Represents an identifier referencing an animation clip."""

    _asset_type = 'Animation'

    def load(self, path: Path, /, **kwargs: object) -> AnimationView:
        del kwargs  # unused
        return AnimationView(path)


def write_clip(directory: Path, frames: Sequence[dict], meta: dict) -> Path:
    """Write per-frame dicts (the players' shape) as an Animation asset at directory/, named after it."""
    name = directory.name
    joint_names = sorted({j for f in frames for j in f['angles']})
    tilted = any('root_quat' in f for f in frames)
    root_xy_yaw = np.array([f.get('root_xy_yaw', (0.0, 0.0, 0.0)) for f in frames], dtype=np.float64).reshape(len(frames), 3)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f'{name}.npz',
        allow_pickle=False,
        joint_names=np.array(joint_names),
        angles=np.array([[f['angles'].get(j, 0.0) for j in joint_names] for f in frames], dtype=np.float64),
        t=np.array([f.get('t', k / meta.get('fps', 20.0)) for k, f in enumerate(frames)], dtype=np.float64),
        root_xy_yaw=root_xy_yaw,
        animation_state=np.array([f.get('animation_state', 0) for f in frames], dtype=np.int16),
        # stored per frame heading, as the retarget writes it: the inverse of the loader's turn
        **({'root_quat': from_start_heading(np.array([f.get('root_quat', (0.0, 0.0, 0.0, 1.0)) for f in frames], dtype=np.float64), -root_xy_yaw[:, 2])} if tilted else {}),
    )
    (directory / ANNOTATION).write_text(yaml.safe_dump({'name': name, 'path': f'Animation/{name}', **meta, **({'pelvis_tilt': True} if tilted else {})}, sort_keys=False))
    return directory


def referenced_clips(config: object) -> Iterator[str]:
    """Clip names a humansim agent config plays: every attention clip plus the default clip of each interaction step."""
    if isinstance(config, list):
        for item in config:
            yield from referenced_clips(item)
        return
    if not isinstance(config, dict):
        return
    for key, value in config.items():
        if key == 'clip':
            name = value.get('name') if isinstance(value, dict) else value
            if isinstance(name, str) and name:
                yield name
        else:
            yield from referenced_clips(value)
    interaction, attention = config.get('interaction'), config.get('attention')
    if isinstance(interaction, str) and not (isinstance(attention, dict) and attention.get('clip') is not None):
        if default := _interaction_clip(interaction):
            yield default


@functools.cache
def _interaction_clip(interaction: str) -> str | None:
    """The clip an interaction kind plays while bound (humansim's InteractionKind.clip), None without arena_humansim."""
    try:
        from arena_humansim.core.interaction_kinds import InteractionType
    except ImportError:
        return None
    try:
        return InteractionType[interaction].kind.clip or None
    except KeyError:
        return None


AnimationIdentifier.use(*DynamicPaths.as_resolvers(AnimationIdentifier))
AnimationIdentifier.use(*NetResolver.all(AnimationIdentifier, formats=()))
