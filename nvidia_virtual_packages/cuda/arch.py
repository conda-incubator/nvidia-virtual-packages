# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the arch of CUDA devices on the system.

This implementation uses ctypes to call the CUDA driver API.

# Specification

The virtual package MUST be named `__cuda_arch`.

The virtual package MUST be present when a CUDA device is detected. For systems without CUDA
devices (maybe driver is installed but no devices are present) the virtual package MUST NOT
be present.

When available, the version value MUST be set to the lowest compute capability of all CUDA
devices detected on the system, formatted as {major}.{minor}; subarchitecture letters (a,f)
excluded.

When available, the build string MUST be the device model of the lowest compute capability
device as reported by cuDeviceGetName with chars except for [a-zA-Z0-9] removed, "NVIDIA"
replaced with an empty string, then limited to 64 characters.

If the CONDA_OVERRIDE_CUDA_ARCH environment variable is set to a non-empty value that can be
parsed as a compute capability string, the __cuda_arch virtual package MUST be exposed with
that version with the build string set to "0".

If the CONDA_OVERRIDE_CUDA_ARCH environment variable is set to a non-empty value that can be
parsed as a compute capability string, the __cuda_arch virtual package MUST be exposed with
that version with the build string set to "0".
"""

import ctypes
import ctypes.util
import enum
import functools
import os
import re
import warnings

from conda import plugins


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


class NVIDIAVirtualPackageError(RuntimeError):
    """A unique RuntimeError for NVIDIA virtual package errors, so we can catch errors specific to this plugin."""


def init_driver():
    """Initialize the CUDA driver API"""
    status = library.cuInit(0)
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to initialize CUDA driver: {status}")


def driver_get_version() -> tuple[int, int]:
    """Return the driver version as a tuple of (major, minor)"""
    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA driver version: {status}")
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count() -> int:
    """Return the number of CUDA devices"""
    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA device count: {status}")
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
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA device compute capability: {status}"
        )
    status = library.cuDeviceGetAttribute(
        ctypes.byref(cc_minor),
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA device compute capability: {status}"
        )
    name = ctypes.create_string_buffer(256)
    status = library.cuDeviceGetName(name, 256, device)
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA device name: {status}")
    return (cc_major.value, cc_minor.value, name.value.decode("utf-8"))


def get_minimum_sm() -> tuple[str | None, str | None]:
    """Try to detect the minimum SM of CUDA devices on the system."""

    default_name = "0"
    example_override = "Overrides must be of the form: CONDA_OVERRIDE_CUDA_ARCH=0.1 or CONDA_OVERRIDE_CUDA_ARCH=0.1=RTX2345DeviceModelName"

    if "CONDA_OVERRIDE_CUDA_ARCH" in os.environ:
        override = os.environ["CONDA_OVERRIDE_CUDA_ARCH"].strip().split("=")
        if not re.fullmatch(r"^[0-9]+\.[0-9]+$", override[0]):
            warnings.warn(
                f"Invalid compute capability ({override[0]}) provided in CONDA_OVERRIDE_CUDA_ARCH. "
                f"The __cuda_arch virtual package will not be created. "
                f"{example_override}"
            )
            return None, None
        else:
            sm = override[0]
        if len(override) < 2:
            warnings.warn(
                f"A device model was not provided in CONDA_OVERRIDE_CUDA_ARCH. "
                f"The default model of '{sm}={default_name}' will be used instead. "
                f"{example_override}"
            )
            name = default_name
        elif not re.fullmatch(r"[a-zA-Z0-9_.+]*", override[1]):
            warnings.warn(
                f"Invalid device model ({override[1]}) provided in CONDA_OVERRIDE_CUDA_ARCH. "
                f"The default model of '{sm}={default_name}' will be used instead. "
                f"{example_override}"
            )
            name = default_name
        else:
            name = override[1]
        return sm, name

    init_driver()

    minimum_sm_major: int = 999
    minimum_sm_minor: int = 999
    device_name: str = default_name
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
    # Strip out all characters disallowed by CEP-26 and replace "NVIDIA" with an empty
    # string to save space. Limit the length to 64 characters because of CEP-26.
    stripped_name = re.sub(
        "NVIDIA", "", re.sub(r"[^a-zA-Z0-9]", "", device_name), flags=re.IGNORECASE
    )[:64]
    return f"{minimum_sm_major}.{minimum_sm_minor}", stripped_name


@functools.cache
def cached_minimum_sm() -> tuple[str | None, str | None]:
    """Return a cached version of the minimum_sm."""
    try:
        return get_minimum_sm()
    except NVIDIAVirtualPackageError:
        return None, None


@plugins.hookimpl
def conda_virtual_packages():
    minimum_sm, device_model_name = cached_minimum_sm()
    if minimum_sm is not None and device_model_name is not None:
        # According to CEP-26, we should only create the virtual package if we can
        # detect the driver and devices
        yield plugins.CondaVirtualPackage(
            name="cuda_arch", version=minimum_sm, build=device_model_name
        )
