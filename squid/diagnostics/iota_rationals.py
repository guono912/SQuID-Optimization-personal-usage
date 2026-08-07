"""Low-order rational surface scans of the VMEC iota profile.

Pure numpy logic (no netCDF4/scipy); operates on any object exposing
``vmec.wout.iotaf``. Moved here from scripts/diagnose.py so the calculation
is reusable outside the diagnostic CLI.
"""

from fractions import Fraction

import numpy as np


def _low_order_rationals(max_denominator, max_value):
    """Return unique positive p/q values up to *max_denominator*."""
    rationals = set()
    max_value = max(float(max_value), 0.0)
    for q in range(1, int(max_denominator) + 1):
        p_max = int(np.ceil(max_value * q)) + 1
        for p in range(1, p_max + 1):
            rationals.add(Fraction(p, q))
    return sorted(rationals, key=lambda x: (float(x), x.denominator))


def _scan_iota_rationals(vmec, max_denominator=8, warn_distance=0.01,
                         s_min=0.05, s_max=0.95):
    """
    Scan the VMEC iota profile against low-order rational surfaces.

    This function only reports geometric facts relative to the supplied
    threshold. It does not decide whether a configuration should be accepted.
    """
    iotaf = np.array(getattr(vmec.wout, "iotaf", []), dtype=float)
    if iotaf.size < 2 or not np.any(np.isfinite(iotaf)):
        return {
            "available": False,
            "reason": "iota profile missing or non-finite",
        }

    s_grid = np.linspace(0.0, 1.0, iotaf.size)
    mask = (
        (s_grid >= float(s_min)) &
        (s_grid <= float(s_max)) &
        np.isfinite(iotaf)
    )
    if not np.any(mask):
        return {
            "available": False,
            "reason": "no finite iota points in requested scan range",
        }

    s_scan = s_grid[mask]
    iota_scan = iotaf[mask]
    abs_iota = np.abs(iota_scan)
    signed_values = iota_scan[np.abs(iota_scan) > 1e-12]
    signs_present = sorted({-1.0 if value < 0 else 1.0 for value in signed_values})
    sign_flip = len(signs_present) > 1

    rationals = _low_order_rationals(
        max_denominator=max_denominator,
        max_value=np.nanmax(abs_iota) + abs(warn_distance),
    )

    nearest = None
    near_points = []
    crossings = []

    zero_crossings = []
    zero_diff = iota_scan
    for j in range(len(zero_diff) - 1):
        if not (np.isfinite(zero_diff[j]) and np.isfinite(zero_diff[j + 1])):
            continue
        if zero_diff[j] == 0:
            zero_crossings.append(float(s_scan[j]))
        elif zero_diff[j] * zero_diff[j + 1] < 0:
            t = -zero_diff[j] / (zero_diff[j + 1] - zero_diff[j])
            zero_crossings.append(float(s_scan[j] + t * (s_scan[j + 1] - s_scan[j])))

    for rat in rationals:
        rat_value = float(rat)
        # Scan signed targets. The previous implementation scanned |iota| and
        # applied the median sign to the whole profile, which mislabels or
        # misses resonances when iota changes sign.
        targets = [rat_value * sign for sign in signs_present]
        for target in targets:
            distances = np.abs(iota_scan - target)
            idx = int(np.nanargmin(distances))
            distance = float(distances[idx])
            signed_num = int(np.sign(target) * rat.numerator)
            record = {
                "s": float(s_scan[idx]),
                "iota": float(iota_scan[idx]),
                "rational": f"{signed_num}/{rat.denominator}",
                "rational_abs": rat_value,
                "denominator": int(rat.denominator),
                "distance": distance,
            }
            if nearest is None or distance < nearest["distance"]:
                nearest = record
            if distance <= warn_distance:
                near_points.append(record)

            diff = iota_scan - target
            for j in range(len(diff) - 1):
                if not (np.isfinite(diff[j]) and np.isfinite(diff[j + 1])):
                    continue
                if diff[j] == 0:
                    s_cross = float(s_scan[j])
                elif diff[j] * diff[j + 1] > 0:
                    continue
                else:
                    denom = diff[j + 1] - diff[j]
                    if abs(denom) < 1e-30:
                        continue
                    t = -diff[j] / denom
                    if t < 0 or t > 1:
                        continue
                    s_cross = float(s_scan[j] + t * (s_scan[j + 1] - s_scan[j]))
                crossings.append({
                    "s": s_cross,
                    "rational": f"{signed_num}/{rat.denominator}",
                    "rational_abs": rat_value,
                    "denominator": int(rat.denominator),
                })

    near_points.sort(key=lambda item: item["distance"])
    crossings.sort(key=lambda item: (item["s"], item["denominator"]))
    shear = np.gradient(iota_scan, s_scan) if len(s_scan) > 1 else np.array([np.nan])

    return {
        "available": True,
        "settings": {
            "max_denominator": int(max_denominator),
            "warn_distance": float(warn_distance),
            "s_min": float(s_min),
            "s_max": float(s_max),
        },
        "s": s_scan,
        "iota": iota_scan,
        "iota_min": float(np.nanmin(iota_scan)),
        "iota_max": float(np.nanmax(iota_scan)),
        "abs_iota_min": float(np.nanmin(abs_iota)),
        "abs_iota_max": float(np.nanmax(abs_iota)),
        "sign_flip": sign_flip,
        "zero_crossings": sorted(set(zero_crossings)),
        "shear_min": float(np.nanmin(shear)),
        "shear_max": float(np.nanmax(shear)),
        "nearest_low_order": nearest,
        "near_low_order": near_points,
        "crossings": crossings,
    }
