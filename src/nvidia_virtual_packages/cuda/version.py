# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the CUDA versions supported by the system.

This implementation uses ctypes to call the CUDA driver API.

# Specification

conda ships a built-in ``__cuda`` virtual package; :mod:`nvidia_virtual_packages.cuda.plugin`
replaces it with this implementation so the driver is probed only once, shared with
``__cuda_arch``.

When exposed, the version value is set to the newest CUDA version fully supported by the
driver formatted as ``{major}.{minor}`` and the build string is ``0``. Detection honors the
``CONDA_OVERRIDE_CUDA`` environment variable with conda's built-in semantics: an empty value
suppresses ``__cuda``, and a non-empty value overrides the version.
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
