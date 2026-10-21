# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the minimum arch of CUDA devices on the system.

This implementation uses ctypes to call the CUDA driver API.

# Specification

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

import functools
import os

from nvidia_virtual_packages.cuda._ctypes import detect


@functools.cache
def get_minimum_sm():
    """Try to detect the minimum SM of CUDA devices on the system.

    Returns NULL (rather than None) when ``__cuda_arch`` must be absent, so that conda's
    ``to_virtual_package`` drops it; conda calls this lazily, only when it is needed.
    """
    from conda.auxlib import NULL

    if (
        "CONDA_OVERRIDE_CUDA" in os.environ
        and os.environ["CONDA_OVERRIDE_CUDA"].strip() == ""
    ):
        return NULL

    if "CONDA_OVERRIDE_CUDA_ARCH" in os.environ:
        override = os.environ["CONDA_OVERRIDE_CUDA_ARCH"].strip()
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
        # CUDA via CONDA_OVERRIDE_CUDA, trust them; otherwise require a real driver.
        if "CONDA_OVERRIDE_CUDA" not in os.environ and detect() is None:
            import warnings

            warnings.warn(
                f"CONDA_OVERRIDE_CUDA_ARCH is set ({override}), but neither the CUDA driver "
                f"or CONDA_OVERRIDE_CUDA were detected. "
                f"The __cuda_arch virtual package will not be created."
            )
            return NULL
        return override

    result = detect()
    if result is None:
        return NULL
    _, capabilities = result
    if not capabilities:
        return NULL

    minimum_sm_major, minimum_sm_minor = min(capabilities)
    return f"{minimum_sm_major}.{minimum_sm_minor}"
