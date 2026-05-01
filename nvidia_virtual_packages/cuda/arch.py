# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Define a virtual package exposing the arch of CUDA devices on the system.

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

import ctypes
import ctypes.util
import enum
import functools
import os
import re
import typing
import warnings

from conda import plugins

# WinDLL only exists on Windows; use a type alias based on platform so annotations work everywhere.
if os.name == "nt":
    DLL: typing.TypeAlias = ctypes.WinDLL  # type: ignore
else:
    DLL: typing.TypeAlias = ctypes.CDLL  # type: ignore


class CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0


class CUdevice_attribute(enum.IntEnum):
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class NVIDIAVirtualPackageError(RuntimeError):
    """A unique RuntimeError for NVIDIA virtual package errors, so we can catch errors specific to this plugin."""


def init_driver() -> DLL:
    """Initialize the CUDA driver API"""

    if os.name == "nt":
        library_path = ctypes.util.find_library("nvcuda")
        if library_path is None:
            raise NVIDIAVirtualPackageError("Failed to find nvcuda library")
        library = ctypes.WinDLL(library_path)  # type: ignore[unused-ignore,attr-defined]
    elif os.name == "posix":
        library_path = ctypes.util.find_library("cuda")
        if library_path is None:
            raise NVIDIAVirtualPackageError("Failed to find cuda library")
        library = ctypes.CDLL(library_path)  # type: ignore[unused-ignore,attr-defined]
    else:
        raise NVIDIAVirtualPackageError(f"Unsupported OS: {os.name}")

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

    status = library.cuInit(0)
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to initialize CUDA driver: {status}")

    return library


def driver_get_version(library: DLL) -> tuple[int, int]:
    """Return the driver version as a tuple of (major, minor)"""
    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA driver version: {status}")
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count(library: DLL) -> int:
    """Return the number of CUDA devices"""
    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA device count: {status}")
    return device_count.value


def device_get_attributes(library: DLL, device: int) -> tuple[int, int]:
    """Return a tuple of (cc_major, cc_minor)"""
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
    return cc_major.value, cc_minor.value


@functools.cache
def get_minimum_sm() -> tuple[str | None, str | None]:
    """Try to detect the minimum SM of CUDA devices on the system."""

    if (
        "CONDA_OVERRIDE_CUDA" in os.environ
        and os.environ["CONDA_OVERRIDE_CUDA"].strip() == ""
    ):
        return None, None

    default_name = "0"
    example_override = "Overrides must be of the form: CONDA_OVERRIDE_CUDA_ARCH=0.1"

    if "CONDA_OVERRIDE_CUDA_ARCH" in os.environ:
        override = os.environ["CONDA_OVERRIDE_CUDA_ARCH"].strip()
        if override == "":
            return None, None
        if not re.fullmatch(r"^[0-9]+\.[0-9]+$", override):
            warnings.warn(
                f"Invalid compute capability ({override}) provided in CONDA_OVERRIDE_CUDA_ARCH. "
                f"The __cuda_arch virtual package will not be created. "
                f"{example_override}"
            )
            return None, None
        return override, default_name

    library = init_driver()

    device_count = device_get_count(library)
    if device_count == 0:
        return None, None

    minimum_sm_major: int = 999
    minimum_sm_minor: int = 999
    for device in range(device_count):
        compute_capability_major, compute_capability_minor = device_get_attributes(
            library, device
        )
        if (compute_capability_major, compute_capability_minor) < (
            minimum_sm_major,
            minimum_sm_minor,
        ):
            minimum_sm_major = compute_capability_major
            minimum_sm_minor = compute_capability_minor

    return f"{minimum_sm_major}.{minimum_sm_minor}", default_name


@plugins.hookimpl
def conda_virtual_packages():
    try:
        minimum_sm, device_model_name = get_minimum_sm()
    except NVIDIAVirtualPackageError:
        minimum_sm, device_model_name = None, None
    if minimum_sm is not None and device_model_name is not None:
        # According to CEP-26, we should only create the virtual package if we can
        # detect the driver and devices
        yield plugins.CondaVirtualPackage(
            name="cuda_arch", version=minimum_sm, build=device_model_name
        )
