# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the minimum arch of CUDA devices on the system.

# Specification

From CEP 46 (https://github.com/conda/ceps/blob/main/cep-0046.md):

Implementing the `__cuda_arch` virtual package is RECOMMENDED. If a conda-compatible client
chooses to implement the `__cuda_arch` virtual package, it MUST follow these specifications:

The `__cuda_arch` virtual package MUST be absent when the `__cuda` virtual package is
absent.

When present, the version value MUST be set to the lowest compute capability of all CUDA
devices detected on the system, formatted as `{major}.{minor}`; subarchitecture letters
(e.g. `a`, `f`) are excluded. The build string MUST be `0`.

The `__cuda_arch` virtual package MUST be present when a CUDA device is detected EXCEPT when
`CONDA_OVERRIDE_CUDA_ARCH` is set as described below.

For systems without CUDA devices (e.g. a driver is installed but no devices are present),
the virtual package MUST be absent EXCEPT when `CONDA_OVERRIDE_CUDA_ARCH` is set as
described below.

If the `CONDA_OVERRIDE_CUDA_ARCH` environment variable is set to a non-empty value that can
be parsed as a compute capability string, the `__cuda_arch` virtual package MUST be exposed
with that version with the build string set to `0` EXCEPT when the `__cuda` virtual package
is absent as described above.

If the `CONDA_OVERRIDE_CUDA_ARCH` environment variable is set to the empty string, the
`__cuda_arch` virtual package MUST be absent.
"""

from __future__ import annotations

import os

from nvidia_virtual_packages.cuda.detection import detect
from nvidia_virtual_packages.cuda.version import get_cuda_version


def _get_override(name: str) -> str | None:
    """Return the override for virtual package ``name``, as conda resolves it.

    Like ``CondaVirtualPackage.to_virtual_package``, the ``CONDA_OVERRIDE_<NAME>`` environment
    variable takes precedence over ``override_virtual_packages`` in the conda configuration.
    """
    value = os.getenv(f"CONDA_OVERRIDE_{name.upper()}")
    if value is None:
        from conda.base.context import context

        if context.override_virtual_packages:
            value = context.override_virtual_packages.get(name)
    return None if value is None else value.strip()


def get_minimum_sm():
    """Try to detect the minimum SM of CUDA devices on the system.

    Returns NULL (rather than None) when ``__cuda_arch`` must be absent, so that conda's
    ``to_virtual_package`` drops it; conda calls this lazily, only when it is needed.
    """
    from conda.auxlib import NULL

    cuda_override = _get_override("cuda")
    if cuda_override == "":
        return NULL

    override = _get_override("cuda_arch")
    if override is not None:
        if override == "":
            return NULL
        import re

        if not re.fullmatch(r"[0-9]+\.[0-9]+", override):
            import warnings

            warnings.warn(
                f"Invalid compute capability ({override}) provided in CONDA_OVERRIDE_CUDA_ARCH. "
                f"The __cuda_arch virtual package will not be created. "
                f"Overrides must be of the form: CONDA_OVERRIDE_CUDA_ARCH=0.1"
            )
            return NULL
        # __cuda must be present for __cuda_arch to be exposed. If the user has asserted
        # CUDA via CONDA_OVERRIDE_CUDA, trust them; otherwise __cuda must be detected.
        if cuda_override is None and get_cuda_version() is NULL:
            import warnings

            warnings.warn(
                f"CONDA_OVERRIDE_CUDA_ARCH is set ({override}), but neither the CUDA driver "
                f"or CONDA_OVERRIDE_CUDA were detected. "
                f"The __cuda_arch virtual package will not be created."
            )
            return NULL
        return override

    result = detect()
    if result is None or not result.compute_capabilities:
        return NULL

    minimum_sm_major, minimum_sm_minor = min(result.compute_capabilities)
    return f"{minimum_sm_major}.{minimum_sm_minor}"
