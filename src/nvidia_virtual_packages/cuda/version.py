# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the CUDA versions supported by the system.

This implementation uses ctypes to call the CUDA driver API.

# Specification

conda ships a built-in ``__cuda`` virtual package, and raises ``PluginError`` when two plugins
provide the same virtual package. This plugin replaces conda's built-in provider using a pluggy
hook wrapper: it removes the other ``cuda`` result before conda evaluates it, so conda's own
driver probe never runs, and adds its own. The driver is then initialised only once, shared
with ``__cuda_arch``.

When exposed, the version value is set to the newest CUDA version fully supported by the
driver formatted as ``{major}.{minor}`` and the build string is ``0``. Detection honors the
``CONDA_OVERRIDE_CUDA`` environment variable with conda's built-in semantics: an empty value
suppresses ``__cuda``, and a non-empty value overrides the version.
"""

import functools

from conda import plugins
from conda.auxlib import NULL

from nvidia_virtual_packages.cuda._ctypes import detect


@functools.cache
def get_cuda_version():
    """Try to detect the newest CUDA version with full driver support.

    The result is cached so the driver is probed only once per process. NULL is returned
    (rather than None) so that conda's ``to_virtual_package`` suppresses the ``__cuda``
    package when no driver is present.
    """
    result = detect()
    if result is None:
        return NULL
    (major, minor), _ = result
    return f"{major}.{minor}"


# Old-style hookwrapper (rather than ``wrapper=True``) for compatibility with older pluggy.
@plugins.hookimpl(hookwrapper=True)
def conda_virtual_packages():
    outcome = yield
    # Drop any other __cuda provider (conda's built-in). Its version is a deferred callable,
    # so discarding it here means its driver probe never runs.
    results = [
        [package for package in packages if package.name != "cuda"]
        for packages in outcome.get_result()
    ]
    results.append(
        [
            plugins.types.CondaVirtualPackage(
                name="cuda",
                version=get_cuda_version,
                build="0",
                # conda reads CONDA_OVERRIDE_CUDA itself: empty -> NULL -> absent,
                # non-empty -> that version.
                override_entity="version",
            )
        ]
    )
    outcome.force_result(results)
