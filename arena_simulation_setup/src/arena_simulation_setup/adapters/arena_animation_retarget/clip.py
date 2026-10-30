"""Arena Animation clip files: ``<out>/<name>/<name>.npz`` + ``annotation.yaml``.

The npz carries what Arena's loader reads (``joint_names``, ``angles`` (N, DOF), ``t``, ``root_xy_yaw``,
``animation_state``) plus ``root_z`` and ``root_quat`` (pelvis height and orientation). A clip retargeted without folding
the pelvis tilt is annotated ``pelvis_tilt: true``: Arena's player then turns the ped's body by ``root_quat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .retarget import Clip


def write_clip(clip: Clip, out_dir: str | Path, name: str, *, loop: bool = False, desc: str | None = None, tags: list[str] | None = None, note: str = "", extra: dict | None = None) -> Path:
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(clip.angles)
    np.savez_compressed(d / f"{name}.npz", joint_names=np.array(names), angles=np.stack([clip.angles[k] for k in names], 1), t=clip.t, root_xy_yaw=clip.root_xy_yaw, animation_state=np.zeros(len(clip.t), dtype=np.int16), root_z=clip.root_z, root_quat=clip.root_quat)
    tilt = "pelvis tilt folded into waist/hips" if clip.fold_tilt else "pelvis-relative body, apply root_quat"
    ann = {"name": name, "path": f"Animation/{name}", "desc": desc or name.replace("_", " "), "tags": list(tags or ["domain::common"]), "fps": round(clip.fps, 6), "loop": bool(loop), "note": " ".join(x for x in (note, f"Retargeted from SMPL-X by arena_animation_retarget ({tilt}; root_z/root_quat = pelvis).") if x)}
    if not clip.fold_tilt:
        ann["pelvis_tilt"] = True
    ann.update(extra or {})
    (d / "annotation.yaml").write_text(yaml.safe_dump(ann, sort_keys=False))
    return d


def load_clip(path: str | Path) -> Clip:
    """Read a clip npz (or its directory) back. Clips without root_z/root_quat get zeros / identity."""
    p = Path(path)
    if p.is_dir():
        p = p / f"{p.name}.npz"
    z = np.load(p)
    names = [str(n) for n in z["joint_names"]]
    n = len(z["t"])
    return Clip(
        angles={k: z["angles"][:, i].astype(float) for i, k in enumerate(names)},
        t=z["t"].astype(float),
        root_xy_yaw=z["root_xy_yaw"].astype(float) if "root_xy_yaw" in z.files else np.zeros((n, 3)),
        root_z=z["root_z"].astype(float) if "root_z" in z.files else np.zeros(n),
        root_quat=z["root_quat"].astype(float) if "root_quat" in z.files else np.tile([0.0, 0, 0, 1], (n, 1)),
    )
