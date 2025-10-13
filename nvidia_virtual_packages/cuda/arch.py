# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the arch of CUDA devices on the system.

The virtual package will be `__cuda_arch`, and will provide the minimum SM of CUDA devices
detected on the system. The version will be the SM version and the build string will be the
device model.

This implementation uses ctypes to call the CUDA driver API.
"""

import ctypes
import ctypes.util
import enum
import functools
import os
import typing

from conda import plugins


library: typing.Union[ctypes.WinDLL, ctypes.CDLL]
if os.name == "nt":
    library = ctypes.WinDLL(ctypes.util.find_library("nvcuda"))
elif os.name == "posix":
    library = ctypes.CDLL(ctypes.util.find_library("cuda"))
else:
    raise RuntimeError(f"Unsupported OS: {os.name}")


class CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0


class CUdevice_attribute(enum.IntEnum):
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


library.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
library.cuDriverGetVersion.restype = ctypes.c_int
library.cuInit.argtypes = [ctypes.c_uint]
library.cuInit.restype = ctypes.c_int
library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
library.cuDeviceGetCount.restype = ctypes.c_int
library.cuDeviceGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
library.cuDeviceGetAttribute.restype = ctypes.c_int
library.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
library.cuDeviceGetName.restype = ctypes.c_int


def init_driver():
    """Initialize the CUDA driver API"""
    status = library.cuInit(0)
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to initialize CUDA driver: {status}")


def driver_get_version() -> tuple[int, int]:
    """Return the driver version as a tuple of (major, minor)"""
    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA driver version: {status}")
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count() -> int:
    """Return the number of CUDA devices"""
    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device count: {status}")
    return device_count.value


def device_get_attributes(device: int) -> tuple[int, int, str]:
    """Return a tuple of (cc_major, cc_minor, device_model)"""
    cc_major = ctypes.c_int(0)
    cc_minor = ctypes.c_int(0)
    status = library.cuDeviceGetAttribute(
        ctypes.byref(cc_major),
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device,
    )
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device compute capability: {status}")
    status = library.cuDeviceGetAttribute(
        ctypes.byref(cc_minor),
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device compute capability: {status}")
    name = ctypes.create_string_buffer(256)
    status = library.cuDeviceGetName(name, 256, device)
    if status != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device name: {status}")
    return (cc_major.value, cc_minor.value, name.value.decode("utf-8"))


def get_minimum_sm() -> tuple[str, typing.Union[None, str]]:
    """Try to detect the minimum SM of CUDA devices on the system."""
    if "CONDA_OVERRIDE_CUDA_ARCH" in os.environ:
        override = os.environ["CONDA_OVERRIDE_CUDA_ARCH"].strip().split("=")
        return override[0] or "0.0", None if len(override) < 2 else override[1]

    init_driver()

    minimum_sm_major: int = 999
    minimum_sm_minor: int = 999
    device_name: str = "None"
    for device in range(device_get_count()):
        compute_capability_major, compute_capability_minor, name = (
            device_get_attributes(device)
        )
        if (
            compute_capability_major < minimum_sm_major
            and compute_capability_minor < minimum_sm_minor
        ):
            minimum_sm_major = compute_capability_major
            minimum_sm_minor = compute_capability_minor
            device_name = name
    stripped_name = device_name.replace(" ", "").replace("NVIDIA", "")
    # FIXME: Figure out what to do if any of the queries fail
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
