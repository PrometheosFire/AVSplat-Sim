"""Trajectory manipulation for standalone rendering.

This module provides utilities to transform camera trajectories in local camera
coordinate frames. Transformations include translations along the camera's
local axes (forward/backward, left/right, up/down).

Coordinate System Conventions
-----------------------------
Camera-to-world (c2w) matrices are 4x4 homogeneous transforms where:
- The rotation part R (3x3) has columns representing camera axes in world coords:
    - R[:, 0]: local X axis (forward direction)
    - R[:, 1]: local Y axis (down direction, -Y = up)
    - R[:, 2]: local Z axis (forward-ish / depth)
- The translation part t (3x1) is the camera position in world coordinates

Sign conventions for shifts (from camera's perspective):
- +X: right, -X: left
- +Y: down, -Y: up
- +Z: forward, -Z: backward

Note: The up direction in this coordinate system is -Y.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrajectoryShift:
    """Configuration for a single trajectory transformation.

    Attributes:
        name: Human-readable name for this shift configuration (e.g., "left_0.5m")
        x_m: Lateral shift in meters (+ = right, - = left)
        y_m: Vertical shift in meters (+ = down, - = up)
        z_m: Longitudinal shift in meters (+ = forward, - = backward)
    """

    name: str
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    def is_identity(self) -> bool:
        """Check if this shift is effectively no transformation."""
        return abs(self.x_m) < 1e-6 and abs(self.y_m) < 1e-6 and abs(self.z_m) < 1e-6


class TrajectoryManipulator:
    """Transforms camera trajectories by applying translations in local camera frame.

    This class operates on camera-to-world (c2w) pose matrices and applies
    transformations in each camera's local coordinate frame, not the world frame.
    This means a "right shift" always moves the camera to its own right, regardless
    of the camera's orientation in the world.

    Example:
        >>> manipulator = TrajectoryManipulator()
        >>> # Original trajectory: [N, 4, 4] c2w matrices
        >>> c2ws = np.load("camtoworlds.npy")
        >>> # Shift 0.5m to the right (local Y axis)
        >>> shifted = manipulator.apply_translation(c2ws, y_shift_m=0.5)
    """

    def apply_translation(
        self,
        camtoworlds: np.ndarray,
        x_shift_m: float = 0.0,
        y_shift_m: float = 0.0,
        z_shift_m: float = 0.0,
        world_to_normalized_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Apply translation along camera's local axes.

        The shift is applied in the camera's local coordinate frame:
        - X axis (column 0 of rotation): right/left
        - Y axis (column 1 of rotation): down/up (-Y = up)
        - Z axis (column 2 of rotation): forward/backward

        Shift values are specified in real-world meters. If the scene was
        normalized during training, pass world_to_normalized_scale to convert
        automatically.

        Args:
            camtoworlds: Camera-to-world matrices of shape [N, 4, 4] or [N, 3, 4]
            x_shift_m: Lateral shift in meters (+ = right)
            y_shift_m: Vertical shift in meters (+ = down, - = up)
            z_shift_m: Longitudinal shift in meters (+ = forward)
            world_to_normalized_scale: Scale factor from normalization (meters -> normalized units).
                If provided, shifts are multiplied by this factor.

        Returns:
            Shifted camera-to-world matrices with same shape as input
        """
        if world_to_normalized_scale is not None:
            x_shift_m *= world_to_normalized_scale
            y_shift_m *= world_to_normalized_scale
            z_shift_m *= world_to_normalized_scale
        if camtoworlds.shape[-2:] == (3, 4):
            # Convert [N, 3, 4] to [N, 4, 4] for uniform handling
            n = camtoworlds.shape[0]
            c2ws_4x4 = np.zeros((n, 4, 4), dtype=camtoworlds.dtype)
            c2ws_4x4[:, :3, :] = camtoworlds
            c2ws_4x4[:, 3, 3] = 1.0
            was_3x4 = True
        else:
            c2ws_4x4 = camtoworlds.copy()
            was_3x4 = False

        # Extract local axes from rotation part (columns of R)
        x_axis = c2ws_4x4[:, :3, 0]  # [N, 3] - forward direction
        y_axis = c2ws_4x4[:, :3, 1]  # [N, 3] - right direction
        z_axis = c2ws_4x4[:, :3, 2]  # [N, 3] - up direction

        # Apply shifts in local frame (expressed in world coordinates)
        c2ws_4x4[:, :3, 3] += (
            x_axis * x_shift_m + y_axis * y_shift_m + z_axis * z_shift_m
        )

        if was_3x4:
            return c2ws_4x4[:, :3, :]
        return c2ws_4x4

    def apply_shift(
        self,
        camtoworlds: np.ndarray,
        shift: TrajectoryShift,
        world_to_normalized_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Apply a TrajectoryShift configuration to poses.

        Args:
            camtoworlds: Camera-to-world matrices of shape [N, 4, 4] or [N, 3, 4]
            shift: TrajectoryShift configuration specifying the transformation
            world_to_normalized_scale: Scale factor from normalization (meters -> normalized units)

        Returns:
            Shifted camera-to-world matrices with same shape as input
        """
        return self.apply_translation(
            camtoworlds,
            x_shift_m=shift.x_m,
            y_shift_m=shift.y_m,
            z_shift_m=shift.z_m,
            world_to_normalized_scale=world_to_normalized_scale,
        )

    def apply_shifts(
        self,
        camtoworlds: np.ndarray,
        shifts: List[TrajectoryShift],
    ) -> dict[str, np.ndarray]:
        """Apply multiple shift configurations to generate variant trajectories.

        Args:
            camtoworlds: Camera-to-world matrices of shape [N, 4, 4] or [N, 3, 4]
            shifts: List of TrajectoryShift configurations

        Returns:
            Dictionary mapping shift names to transformed trajectory arrays
        """
        results = {}
        for shift in shifts:
            results[shift.name] = self.apply_shift(camtoworlds, shift)
        return results
