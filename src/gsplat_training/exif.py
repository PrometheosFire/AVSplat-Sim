# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import piexif  # type: ignore


def _extract_shutter_time(exif: Dict) -> Optional[float]:
    """Extract shutter time in seconds from EXIF metadata.

    Tries, in order:
    1. ``ExposureTime`` (direct seconds),
    2. ``ShutterSpeedValue`` (APEX Tv, converted via ``2**(-Tv)``).

    Args:
        exif: EXIF dictionary returned by ``piexif.load``.

    Returns:
        Positive finite shutter time in seconds, or ``None`` if unavailable/invalid.
    """
    # EXIF tag IDs (decimal)
    TAG_EXPOSURE_TIME = 33434  # ExposureTime (seconds)
    TAG_SHUTTER_SPEED_VALUE = 37377  # ShutterSpeedValue (APEX Tv)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    # Try ExposureTime first, as it's a direct measure of shutter time.
    if TAG_EXPOSURE_TIME in exif_ifd:
        num, den = exif_ifd[TAG_EXPOSURE_TIME]
        seconds = num / den
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds
    # If ExposureTime is unavailable, try ShutterSpeedValue (Tv).
    if TAG_SHUTTER_SPEED_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_SHUTTER_SPEED_VALUE]
        tv = num / den
        seconds = math.pow(2.0, -tv)
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    return None


def _extract_aperture_fnumber(exif: Dict) -> Optional[float]:
    """Extract aperture as f-number from EXIF metadata.

    Tries, in order:
    1. ``FNumber`` (direct f-stop),
    2. ``ApertureValue`` (APEX Av, converted via ``2**(Av/2)``).

    Args:
        exif: EXIF dictionary returned by ``piexif.load``.

    Returns:
        Positive finite f-number, or ``None`` if unavailable/invalid.
    """
    # EXIF tag IDs (decimal)
    TAG_FNUMBER = 33437  # FNumber (f-number)
    TAG_APERTURE_VALUE = 37378  # ApertureValue (APEX Av)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    # Try FNumber first, as it directly stores the f-stop.
    if TAG_FNUMBER in exif_ifd:
        num, den = exif_ifd[TAG_FNUMBER]
        fnum = num / den
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    # If FNumber is unavailable, try ApertureValue (Av).
    if TAG_APERTURE_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_APERTURE_VALUE]
        av = num / den
        fnum = math.pow(2.0, av / 2.0)
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    return None


def _extract_iso(exif: Dict) -> Optional[float]:
    """Extract ISO sensitivity from EXIF metadata.

    Checks common ISO-related EXIF tags in priority order and returns the first
    valid value.

    Args:
        exif: EXIF dictionary returned by ``piexif.load``.

    Returns:
        Positive finite ISO value, or ``None`` if unavailable/invalid.
    """
    # EXIF tag IDs (decimal)
    # PhotographicSensitivity / ISOSpeedRatings
    TAG_PHOTOGRAPHIC_SENSITIVITY = 34855
    TAG_STANDARD_OUTPUT_SENSITIVITY = 34857  # StandardOutputSensitivity (SOS)
    TAG_RECOMMENDED_EXPOSURE_INDEX = 34858  # RecommendedExposureIndex (REI)
    TAG_ISO_SPEED = 34859  # ISOSpeed
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    candidates: List[int] = [
        TAG_PHOTOGRAPHIC_SENSITIVITY,
        TAG_RECOMMENDED_EXPOSURE_INDEX,
        TAG_STANDARD_OUTPUT_SENSITIVITY,
        TAG_ISO_SPEED,
    ]

    # Return the first valid ISO-like value found in priority order.
    for tag in candidates:
        if tag in exif_ifd:
            value = float(exif_ifd[tag])
            if value > 0.0 and math.isfinite(value):
                return value

    return None


def compute_exposure_from_exif(path: Path) -> Optional[float]:
    """Compute relative exposure from image EXIF and return it in log2 stops.

    The function extracts shutter time, aperture f-number, and ISO from EXIF, then
    computes relative exposure:

    ``rel_exposure = (seconds / f_number^2) * iso``

    and returns ``log2(rel_exposure)``.

    Missing components are treated as ``1.0`` if at least one component is present,
    so partial EXIF metadata can still produce a usable relative exposure. If no
    exposure-related metadata is available, or if the computed exposure is invalid,
    the function returns ``None``.

    Args:
        path: Path to the input image.

    Returns:
        Relative exposure in EV-like stops (log2 scale), or ``None`` when EXIF data
        is unavailable/invalid.
    """
    try:
        exif = piexif.load(str(path))
    except piexif.InvalidImageDataError:
        # File format doesn't support EXIF (e.g., PNG)
        return None

    # Extract each exposure component from EXIF.
    shutter_s = _extract_shutter_time(exif)
    aperture_f = _extract_aperture_fnumber(exif)
    iso_value = _extract_iso(exif)

    # If none of the components are available, we cannot compute exposure
    if shutter_s is None and aperture_f is None and iso_value is None:
        return None

    # Use available components; treat missing ones as 1 for exposure calculation
    seconds = shutter_s if shutter_s is not None else 1.0
    f_number = aperture_f if aperture_f is not None else 1.0
    iso = iso_value if iso_value is not None else 1.0

    rel_exposure = (seconds / (f_number * f_number)) * iso
    if rel_exposure <= 0.0 or not math.isfinite(rel_exposure):
        return None

    # Convert multiplicative exposure to additive stops.
    return math.log2(rel_exposure)
