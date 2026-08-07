"""OOPS / piecewise-omnigenity residuals evaluated on a VMEC wout."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


INVALID_RESIDUAL = 1.0e3


def _install_jax_compat() -> None:
    """Patch older DESC-OOPS imports against newer JAX tree_util."""
    try:
        import jax.tree_util as jtu
    except Exception:
        return
    if hasattr(jtu, "tree_broadcast"):
        return

    def _compat_tree_broadcast(prefix, full_tree, is_leaf=None):
        def fill_like(value, tree):
            treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
            return jtu.tree_unflatten(treedef, [value] * treedef.num_leaves)

        return jtu.tree_map(fill_like, prefix, full_tree, is_leaf=is_leaf)

    jtu.tree_broadcast = _compat_tree_broadcast


def _maybe_add_source_path(source_path: str | None) -> None:
    if not source_path:
        return
    path = str(Path(source_path).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _as_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def oops_backend_status(source_path: str | None = None):
    """Return whether the optional DESC-OOPS API required here is available."""
    _maybe_add_source_path(source_path)
    _install_jax_compat()
    try:
        from desc.magnetic_fields import OmnigenousFieldOOPS  # noqa: F401
        from desc.objectives import OmnigenityHarmonics  # noqa: F401
    except Exception as exc:
        return False, repr(exc)
    return True, None


def oops_harmonics_residuals(wout_path: str, args):
    """Return compact OOPS harmonics residuals and metrics for one VMEC wout."""
    source_path = getattr(args, "oops_desc_source_path", None)
    available, reason = oops_backend_status(source_path)
    if not available:
        return np.array([INVALID_RESIDUAL]), {
            "ok": False,
            "error": f"DESC-OOPS backend unavailable: {reason}",
        }

    try:
        from desc.grid import LinearGrid
        from desc.magnetic_fields import OmnigenousFieldOOPS
        from desc.objectives import ObjectiveFunction, OmnigenityHarmonics
        from desc.vmec import VMECIO
    except Exception as exc:
        return np.array([INVALID_RESIDUAL]), {
            "ok": False,
            "error": f"import failed: {exc!r}",
        }

    try:
        eq = VMECIO.load(
            wout_path,
            L=int(getattr(args, "oops_desc_L", getattr(args, "desc_L", 4))),
            M=int(getattr(args, "oops_desc_M", getattr(args, "desc_M", 4))),
            N=int(getattr(args, "oops_desc_N", getattr(args, "desc_N", 4))),
        )
        rhos = _as_list(getattr(args, "oops_rhos", [0.5]), [0.5])
        grid_m = int(getattr(args, "oops_grid_m", 6))
        grid_n = int(getattr(args, "oops_grid_n", 6))
        m_harm = int(getattr(args, "oops_m_harmonics", 8))
        n_harm = int(getattr(args, "oops_n_harmonics", 8))
        s_list = np.asarray(
            _as_list(getattr(args, "oops_S_list", [0.3, 0.0]), [0.3, 0.0]),
            dtype=float,
        )
        d_list = np.asarray(
            _as_list(getattr(args, "oops_D_list", [0.0, 0.0]), [0.0, 0.0]),
            dtype=float,
        )
        helicity = tuple(
            int(x)
            for x in _as_list(getattr(args, "oops_helicity", [0, 1]), [0, 1])[:2]
        )
        field = OmnigenousFieldOOPS(
            S_len=int(s_list.size),
            D_len=int(d_list.size),
            NFP=int(eq.NFP),
            helicity=helicity,
            S_list=s_list,
            D_list=d_list,
        )

        residuals = []
        scalars = []
        scale = max(float(getattr(args, "oops_scalar_scale", 1.0)), 1e-12)
        for rho in rhos:
            grid = LinearGrid(
                rho=np.array([float(rho)]),
                M=grid_m,
                N=grid_n,
                NFP=int(eq.NFP),
                sym=False,
            )
            obj = ObjectiveFunction(
                OmnigenityHarmonics(
                    eq=eq,
                    field=field,
                    field_type="oops",
                    eq_grid=grid,
                    field_grid=grid,
                    M_harmonics=m_harm,
                    N_harmonics=n_harm,
                    eq_fixed=True,
                )
            )
            obj.build(verbose=0)
            scalar = float(np.asarray(obj.compute_scalar(obj.x(field))))
            scalars.append(scalar)
            residuals.append(
                math.sqrt(max(scalar, 0.0) / scale)
                if math.isfinite(scalar)
                else INVALID_RESIDUAL
            )

        arr = np.asarray(residuals, dtype=float)
        return arr, {
            "ok": bool(np.all(np.isfinite(arr))),
            "rhos": [float(x) for x in rhos],
            "scalars": [float(x) for x in scalars],
            "scalar_mean": float(np.mean(scalars)) if scalars else np.nan,
            "scalar_max": float(np.max(scalars)) if scalars else np.nan,
            "residual_rms": float(np.sqrt(np.mean(arr**2))) if arr.size else np.nan,
        }
    except Exception as exc:
        return np.array([INVALID_RESIDUAL]), {
            "ok": False,
            "error": repr(exc),
        }
