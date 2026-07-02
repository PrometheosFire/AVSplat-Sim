"""Dynamic rigid-object support for gsplat training (vehicles via 3D tracks)."""

from .alignment import (
    SimilarityTransform,
    compute_colmap_to_training,
    resolve_colmap_sparse_dir,
)
from .rigid_densify import RigidDensifier
from .rigid_nodes import RigidNodes
from .rigid_tracks import RigidTracks, load_rigid_tracks

__all__ = [
    "SimilarityTransform",
    "compute_colmap_to_training",
    "resolve_colmap_sparse_dir",
    "RigidTracks",
    "load_rigid_tracks",
    "RigidNodes",
    "RigidDensifier",
]
