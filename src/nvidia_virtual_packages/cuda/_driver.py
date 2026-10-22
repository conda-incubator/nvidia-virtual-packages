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
    """The ``CUresult`` codes that ``cuInit`` may return, from ``cuda.h``.

    ``cuGetErrorName`` fails until ``cuInit`` succeeds, so hardcode these likely errors.
    """

    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_STUB_LIBRARY = 34
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
    CUDA_ERROR_OPERATING_SYSTEM = 304
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
    CUDA_ERROR_UNKNOWN = 999


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


def _error_text(library: DLL, function_name: str, status: int) -> str | None:
    """Return what ``cuGetErrorName`` or ``cuGetErrorString`` reports for ``status``, if anything."""
    import ctypes

    with contextlib.suppress(AttributeError):
        function = getattr(library, function_name)
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        function.restype = ctypes.c_int
        value = ctypes.c_char_p()
        if (
            function(status, ctypes.byref(value)) == CUresult.CUDA_SUCCESS
            and value.value
        ):
            return value.value.decode(errors="replace")
    return None


def _error_string(library: DLL, status: int) -> str:
    """Describe ``status``, such as
    ``CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected (100)``.

    The driver only describes errors after ``cuInit`` succeeds; before that, the name comes
    from :class:`CUresult` and there is no description.
    """
    name = _error_text(library, "cuGetErrorName", status)
    if name is None:
        with contextlib.suppress(ValueError):
            name = CUresult(status).name
    description = _error_text(library, "cuGetErrorString", status)
    label = f"{name}: {description}" if name and description else name or description
    return f"{label} ({status})" if label else str(status)


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
        raise NVIDIAVirtualPackageError(
            f"Failed to initialize CUDA driver: {_error_string(library, status)}"
        )

    return library


def driver_get_version(library: DLL) -> tuple[int, int]:
    """Return the driver version as a tuple of (major, minor)"""
    import ctypes

    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA driver version: {_error_string(library, status)}"
        )
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count(library: DLL) -> int:
    """Return the number of CUDA devices"""
    import ctypes

    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA device count: {_error_string(library, status)}"
        )
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
            f"Failed to get CUDA device compute capability: {_error_string(library, status)}"
        )
    status = library.cuDeviceGetAttribute(
        ctypes.byref(cc_minor),
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    if status != CUresult.CUDA_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA device compute capability: {_error_string(library, status)}"
        )
    return cc_major.value, cc_minor.value


def probe() -> CUDASystemInfo:
    """Query the CUDA driver API in-process.

    Raises if no usable driver is found. Devices that cannot be queried are skipped, so the
    driver version and the remaining devices are still reported.

    The driver only enumerates the devices listed in ``CUDA_VISIBLE_DEVICES``, so, like conda's
    built-in ``__cuda`` detector, it is removed from the environment to see every device. Only
    call this in the isolated process that :func:`~nvidia_virtual_packages.cuda.detection.detect`
    spawns, since it modifies that process's environment.
    """
    import os

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    library = init_driver()
    version = driver_get_version(library)
    count = 0
    with contextlib.suppress(Exception):
        count = device_get_count(library)
    capabilities = []
    for device in range(count):
        with contextlib.suppress(NVIDIAVirtualPackageError):
            capabilities.append(device_get_attributes(library, device))
    return CUDASystemInfo(version, capabilities)
