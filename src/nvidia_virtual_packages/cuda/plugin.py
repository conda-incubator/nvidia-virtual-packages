# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The conda plugin providing the ``__cuda`` and ``__cuda_arch`` virtual packages.

conda ships a built-in ``__cuda`` virtual package, and raises ``PluginError`` when two plugins
provide the same virtual package. This plugin replaces conda's built-in provider using a pluggy
hook wrapper: it removes the other ``cuda`` result before conda evaluates it, so conda's own
driver probe never runs, and adds its own. Both virtual packages share one driver probe.
"""

from conda import plugins

from nvidia_virtual_packages.cuda.arch import get_minimum_sm
from nvidia_virtual_packages.cuda.version import get_cuda_version


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
    # Versions are deferred callables, so detection only runs when conda needs the records.
    results.append(
        [
            plugins.types.CondaVirtualPackage(
                name="cuda",
                version=get_cuda_version,
                build="0",
                # conda reads CONDA_OVERRIDE_CUDA itself: empty -> NULL -> absent,
                # non-empty -> that version.
                override_entity="version",
            ),
            # According to CEP 46, __cuda_arch is only created if the driver and devices are
            # detected; get_minimum_sm returns NULL otherwise.
            plugins.types.CondaVirtualPackage(
                name="cuda_arch", version=get_minimum_sm, build="0"
            ),
        ]
    )
    outcome.force_result(results)
