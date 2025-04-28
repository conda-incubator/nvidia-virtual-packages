# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the arch of CUDA devices on the system.

The virtual package will be `__cuda_arch`, and will provide the minimum SM of CUDA devices
detected on the system. The version will be the SM version and the build string will be the
device model.

This implementation is based on cuda.core which returns both major and minor compute
capability. nvidia-ml-py is also an option, but only returns the major compute capability
which means you couldn't tune at compile time to a minor architecture.
"""

import typing

import functools
import os

import cuda.core.experimental as ccx
from conda import plugins


def get_minimum_sm() -> tuple[str, typing.Union[None, str]]:
    """Try to detect the minimum SM of CUDA devices on the system."""
    if "CONDA_OVERRIDE_CUDA_ARCH" in os.environ:
        override = os.environ["CONDA_OVERRIDE_CUDA_ARCH"].strip().split("=")
        return override[0] or "0.0", None if len(override) < 2 else override[1]

    minimum_sm_major: int = 999
    minimum_sm_minor: int = 999
    device_name: str = "None"
    for device in ccx.system.devices:
        if (
            device.compute_capability.major < minimum_sm_major
            and device.compute_capability.minor < minimum_sm_minor
        ):
            minimum_sm_major = device.compute_capability.major
            # FIXME: How to handle those special devices such as 9.0a?
            minimum_sm_minor = device.compute_capability.minor
            device_name = device.name
    stripped_name = device_name.replace(" ", "").replace("NVIDIA", "")
    return f"{minimum_sm_major}.{minimum_sm_minor}", stripped_name


@functools.cache
def cached_minimum_sm():
    """Return a cached version of the minimum_sm."""
    return get_minimum_sm()


@plugins.hookimpl
def conda_virtual_packages():
    minimum_sm, device_model_name = cached_minimum_sm()
    yield plugins.CondaVirtualPackage(
        name="cuda_arch", version=minimum_sm, build=device_model_name
    )
