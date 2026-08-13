# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the CUDA versions supported by the system.

This implementation uses ctypes to call the CUDA driver API.

# Specification

conda ships a built-in ``__cuda`` virtual package. To avoid a fatal plugin conflict (conda
raises ``PluginError`` when two plugins provide the same virtual package), this plugin only
exposes ``__cuda`` when conda's built-in provider is NOT active. Once a conda release
delegates driver detection to this plugin by dropping its built-in provider, this
implementation takes over transparently.

When exposed, the version value is set to the newest CUDA version fully supported by the
driver formatted as ``{major}.{minor}`` and the build string is ``0``. Detection honors the
``CONDA_OVERRIDE_CUDA`` environment variable with conda's built-in semantics: an empty value
suppresses ``__cuda``, and a non-empty value overrides the version.
"""

import functools

from conda import plugins
from conda.auxlib import NULL

from nvidia_virtual_packages.cuda._ctypes import (
    NVIDIAVirtualPackageError,
    driver_get_version,
    init_driver,
)

# Canonical plugin name under which conda registers its built-in __cuda provider.
_BUILTIN_CUDA_PLUGIN = "conda.plugins.virtual_packages.cuda"


@functools.cache
def get_cuda_version():
    """Try to detect the newest CUDA version with full driver support.

    The result is cached so the driver is probed only once per process. NULL is returned
    (rather than None) so that conda's ``to_virtual_package`` suppresses the ``__cuda``
    package when no driver is present.
    """
    try:
        major, minor = driver_get_version(init_driver())
    except NVIDIAVirtualPackageError:
        return NULL
    return f"{major}.{minor}"


def _builtin_cuda_active() -> bool:
    """Return whether conda's built-in ``__cuda`` provider is registered."""
    from conda.base.context import context

    return context.plugin_manager.get_plugin(_BUILTIN_CUDA_PLUGIN) is not None


@plugins.hookimpl
def conda_virtual_packages():
    # Defer to conda's built-in __cuda provider when present to avoid a fatal
    # plugin conflict. Only expose our own provider once the built-in is gone.
    if _builtin_cuda_active():
        return
    yield plugins.CondaVirtualPackage(
        name="cuda",
        version=get_cuda_version,
        build="0",
        # conda reads CONDA_OVERRIDE_CUDA itself: empty -> NULL -> absent,
        # non-empty -> that version.
        override_entity="version",
    )
