# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from depth_anything_3.specs import Prediction

# Loading the inference API must not load optional exporters.
_EXPORTERS = {
    "glb": ("glb", "export_to_glb", "export"),
    "mini_npz": ("npz", "export_to_mini_npz", None),
    "npz": ("npz", "export_to_npz", None),
    "feat_vis": ("feat_vis", "export_to_feat_vis", "video"),
    "depth_vis": ("depth_vis", "export_to_depth_vis", "export"),
    "gs_ply": ("gs", "export_to_gs_ply", "gs"),
    "gs_video": ("gs", "export_to_gs_video", "gs"),
    "colmap": ("colmap", "export_to_colmap", "colmap"),
}
_OPTIONAL_MODULES = {
    "trimesh", "matplotlib", "pycolmap", "gsplat", "e3nn", "plyfile", "scipy", "imageio_ffmpeg"
}

# Canonical set of export formats this dispatcher understands. Treat this as the
# server-trusted enum for any request-facing validation (e.g. the HTTP backend) instead
# of letting a client-supplied export_format string reach the dispatcher unchecked.
SUPPORTED_EXPORT_FORMATS = frozenset(
    {
        "glb",
        "mini_npz",
        "npz",
        "feat_vis",
        "depth_vis",
        "gs_ply",
        "gs_video",
        "colmap",
    }
)


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if not isinstance(export_format, str):
        raise ValueError(f"Unsupported export format: {export_format}")
    formats = export_format.split("-")
    # Validate all components before any output is written.
    if any(fmt not in SUPPORTED_EXPORT_FORMATS for fmt in formats):
        raise ValueError(f"Unsupported export format: {export_format}")
    for fmt in formats:
        module, function, extra = _EXPORTERS[fmt]
        try:
            exporter = getattr(import_module(f".{module}", __name__), function)
            if fmt in {"npz", "mini_npz", "depth_vis"}:
                exporter(prediction, export_dir)
            else:
                exporter(prediction, export_dir, **kwargs.get(fmt, {}))
        except ModuleNotFoundError as exc:
            if extra and exc.name and exc.name.split(".")[0] in _OPTIONAL_MODULES:
                raise ImportError(
                    f"Export '{fmt}' requires optional dependencies. "
                    f"Install depth-anything-3[{extra}] from this fork."
                ) from exc
            raise


__all__ = [
    "export",
    "SUPPORTED_EXPORT_FORMATS",
]
