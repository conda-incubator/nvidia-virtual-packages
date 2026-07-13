# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Low-level ctypes bindings for the CUDA driver API.

This module is the single place that loads the CUDA driver library and wraps the handful of
driver entry points needed by the virtual-package plugins.
"""

import ctypes
import ctypes.util
import enum
import functools
import os
import typing

# WinDLL only exists on Windows. For type-checking, treat the handle as its
# common base class CDLL (WinDLL subclasses it) so annotations resolve on every
# platform; select the concrete loader at runtime.
if typing.TYPE_CHECKING:
    DLL: typing.TypeAlias = ctypes.CDLL
elif os.name == "nt":
    DLL = ctypes.WinDLL
else:
    DLL = ctypes.CDLL


class CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0


class CUdevice_attribute(enum.IntEnum):
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class NVIDIAVirtualPackageError(RuntimeError):
    """A unique RuntimeError for NVIDIA virtual package errors, so we can catch errors specific to this plugin."""


@functools.cache
def init_driver() -> DLL:
    """Initialize the CUDA driver API.

    The loaded, initialized library handle is cached so the driver is loaded and
    ``cuInit`` is called at most once per process, shared across every virtual
    package that needs it. ``functools.cache`` does not memoize exceptions, so a
    failed load is retried on the next call.
    """

    library: DLL
    if os.name == "nt":
        library_path = ctypes.util.find_library("nvcuda")
        if library_path is None:
            raise NVIDIAVirtualPackageError("Failed to find nvcuda library")
        library = ctypes.WinDLL(library_path)  # type: ignore[attr-defined,unused-ignore]
    elif os.name == "posix":
        library_path = ctypes.util.find_library("cuda")
        if library_path is None:
            raise NVIDIAVirtualPackageError("Failed to find cuda library")
        library = ctypes.CDLL(library_path)
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
