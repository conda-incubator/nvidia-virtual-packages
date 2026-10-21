# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the CUDA versions supported by the system.

conda ships a built-in ``__cuda`` virtual package; :mod:`nvidia_virtual_packages.cuda.plugin`
replaces it with this implementation so the driver is probed only once, shared with
``__cuda_arch``.

# Specification

From CEP 30 (https://github.com/conda/ceps/blob/main/cep-0030.md):

This virtual package MUST be present when the system exhibits NVIDIA GPU drivers compatible
with the CUDA runtimes. For systems without such support, the virtual package MUST NOT be
present.

When available, the version value MUST be set to the latest CUDA version supported by the
detected drivers (i.e. the formatted value of `cuDriverGetVersion()`), constrained to the first
two components (major and minor) and formatted as `{major}.{minor}`.

The build string MUST be `0`.

If the `CONDA_OVERRIDE_CUDA` environment variable is set to a non-empty value that can be
parsed as a version string, the `__cuda` virtual package MUST be exposed with that version.
"""

from nvidia_virtual_packages.cuda._ctypes import detect


def get_cuda_version():
    """Try to detect the newest CUDA version with full driver support.

    The probe itself is cached by :func:`detect`. NULL is returned (rather than None) so that conda's ``to_virtual_package`` suppresses the ``__cuda``
    package when no driver is present.
    """
    from conda.auxlib import NULL

    result = detect()
    if result is None:
        return NULL
    (major, minor), _ = result
    return f"{major}.{minor}"
