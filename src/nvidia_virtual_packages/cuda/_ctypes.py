# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""Low-level ctypes bindings for the CUDA driver API.

This module is the single place that loads the CUDA driver library and wraps the handful of
driver entry points needed by the virtual-package plugins.
"""

from __future__ import annotations

import contextlib
import enum
import typing

from nvidia_virtual_packages.cuda.detection import CUDASystemInfo

# ctypes is imported lazily inside the functions that use it to keep plugin import time low.
if typing.TYPE_CHECKING:
    import ctypes

    # WinDLL subclasses CDLL, so CDLL works as the handle type on every platform.
    DLL: typing.TypeAlias = ctypes.CDLL


class CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0


class CUdevice_attribute(enum.IntEnum):
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class NVIDIAVirtualPackageError(RuntimeError):
    """A unique RuntimeError for NVIDIA virtual package errors, so we can catch errors specific to this plugin."""


def _candidate_library_names(system: str) -> list[str]:
    """Return the ordered CUDA driver library candidates for a platform.

    Parameters
    ----------
    system : str
        The platform name as returned by :func:`platform.system`.

    Returns
    -------
    list of str
        Candidate library filenames/paths to try in order. Mirrors conda's
        built-in ``__cuda`` provider so detection covers the same driver-only,
        vendor-dir, and WSL locations. There is intentionally no
        ``find_library`` fallback: the list is exhaustive and deterministic.
    """
    import platform

    if system == "Windows":
        bits = platform.architecture()[0].replace("bit", "")
        return [f"nvcuda{bits}.dll", "nvcuda.dll"]
    if system == "Darwin":
        return [
            "libcuda.1.dylib",
            "libcuda.dylib",
            "/usr/local/cuda/lib/libcuda.1.dylib",
            "/usr/local/cuda/lib/libcuda.dylib",
        ]
    if system == "Linux":
        bases = [
            "libcuda.so",  # check library path first
            "/usr/lib64/nvidia/libcuda.so",  # RHEL/CentOS/Fedora
            "/usr/lib/x86_64-linux-gnu/libcuda.so",  # Debian/Ubuntu multiarch
            "/usr/lib/wsl/lib/libcuda.so",  # WSL
        ]
        # Try the versioned SONAME (``.1``) before the bare name; a driver-only
        # install often lacks the unversioned symlink from the ``-dev`` package.
        return [name for base in bases for name in (f"{base}.1", base)]
    return []


def init_driver() -> DLL:
    """Initialize the CUDA driver API.

    Called only in the detector child process; see :func:`detect`.
    """
    import ctypes
    import platform

    library: DLL
    system = platform.system()
    candidates = _candidate_library_names(system)
    if not candidates:
        raise NVIDIAVirtualPackageError(f"Unsupported OS: {system}")
    loader = ctypes.WinDLL if system == "Windows" else ctypes.CDLL  # type: ignore[attr-defined,unused-ignore]
    for candidate in candidates:
        with contextlib.suppress(OSError):
            library = loader(candidate)
            break
    else:
        raise NVIDIAVirtualPackageError("Failed to load the CUDA driver library")

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
    import ctypes

    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA driver version: {status}")
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count(library: DLL) -> int:
    """Return the number of CUDA devices"""
    import ctypes

    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(f"Failed to get CUDA device count: {status}")
    return device_count.value


def device_get_attributes(library: DLL, device: int) -> tuple[int, int]:
    """Return a tuple of (cc_major, cc_minor)"""
    import ctypes

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


def probe() -> CUDASystemInfo:
    """Query the CUDA driver API in-process.

    Raises if no usable driver is found. The device list is empty if the devices cannot be
    queried, so that the driver version is still reported.

    The driver only enumerates the devices listed in ``CUDA_VISIBLE_DEVICES``, so, like conda's
    built-in ``__cuda`` detector, it is removed from the environment to see every device. Only
    call this in the isolated process that :func:`~nvidia_virtual_packages.cuda.detection.detect`
    spawns, since it modifies that process's environment.
    """
    import os

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    library = init_driver()
    version = driver_get_version(library)
    try:
        capabilities = [
            device_get_attributes(library, device)
            for device in range(device_get_count(library))
        ]
    except Exception:
        capabilities = []
    return CUDASystemInfo(version, capabilities)
