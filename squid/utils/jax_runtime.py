"""Small helpers for deterministic JAX backend initialization."""

import os


def initialize_requested_jax_backend():
    """Eagerly initialize an explicitly requested CUDA backend.

    VMEC/SIMSOPT may be initialized before the first DESC operation. On the
    workstation this can make a later, lazy CUDA initialization lose device
    visibility. Initializing JAX first avoids that ordering-dependent failure.
    CPU/default selections remain lazy.
    """
    # DESC 0.16 EffectiveRipple uses nufft2, which has no CUDA lowering here.
    requested = os.environ.get("JAX_PLATFORMS", "").split(",", 1)[0].strip()
    if requested != "cuda":
        return None

    import jax
    import jax.numpy as jnp

    probe = jnp.ones((1,), dtype=jnp.float32).block_until_ready()
    devices = jax.devices()
    if not devices or probe.device.platform != "gpu":
        raise RuntimeError(
            "JAX_PLATFORMS=cuda was requested, but no CUDA device initialized"
        )
    return str(probe.device)
