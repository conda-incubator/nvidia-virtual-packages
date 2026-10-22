# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The NVIDIA Management Library (NVML) detection method.

Low-level ctypes bindings that load NVML and wrap the handful of entry points needed to fill
:class:`~nvidia_virtual_packages.cuda.detection.CUDASystemInfo`. Unlike the CUDA driver API,
NVML enumerates every device regardless of ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import contextlib
import enum
import typing

from nvidia_virtual_packages.cuda.detection import CUDASystemInfo

# ctypes is imported lazily inside the functions that use it to keep plugin import time low.
if typing.TYPE_CHECKING:
    import ctypes

    DLL: typing.TypeAlias = ctypes.CDLL


class NVMLReturn(enum.IntEnum):
    """The ``nvmlReturn_t`` codes, from ``nvml.h`` (CUDA 13.4)."""

    NVML_SUCCESS = 0
    NVML_ERROR_UNINITIALIZED = 1
    NVML_ERROR_INVALID_ARGUMENT = 2
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_ERROR_NO_PERMISSION = 4
    NVML_ERROR_ALREADY_INITIALIZED = 5
    NVML_ERROR_NOT_FOUND = 6
    NVML_ERROR_INSUFFICIENT_SIZE = 7
    NVML_ERROR_INSUFFICIENT_POWER = 8
    NVML_ERROR_DRIVER_NOT_LOADED = 9
    NVML_ERROR_TIMEOUT = 10
    NVML_ERROR_IRQ_ISSUE = 11
    NVML_ERROR_LIBRARY_NOT_FOUND = 12
    NVML_ERROR_FUNCTION_NOT_FOUND = 13
    NVML_ERROR_CORRUPTED_INFOROM = 14
    NVML_ERROR_GPU_IS_LOST = 15
    NVML_ERROR_RESET_REQUIRED = 16
    NVML_ERROR_OPERATING_SYSTEM = 17
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
    NVML_ERROR_IN_USE = 19
    NVML_ERROR_MEMORY = 20
    NVML_ERROR_NO_DATA = 21
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25
    NVML_ERROR_DEPRECATED = 26
    NVML_ERROR_NOT_READY = 27
    NVML_ERROR_GPU_NOT_FOUND = 28
    NVML_ERROR_INVALID_STATE = 29
    NVML_ERROR_RESET_TYPE_NOT_SUPPORTED = 30
    NVML_ERROR_UNKNOWN = 999


class NVIDIAVirtualPackageError(RuntimeError):
    """A unique RuntimeError for NVIDIA virtual package errors, so we can catch errors specific to this plugin."""


def _candidate_library_names(system: str) -> list[str]:
    """Return the ordered NVML library candidates for a platform.

    Parameters
    ----------
    system : str
        The platform name as returned by :func:`platform.system`.

    Returns
    -------
    list of str
        Candidate library filenames/paths to try in order. Combines the locations searched by
        rattler (``rattler_virtual_packages``) and by nvidia-ml-py for Windows drivers.
    """
    import os

    if system == "Windows":
        return [
            "nvml.dll",  # check library path first
            # DCH drivers
            os.path.join(os.getenv("WINDIR", "C:\\Windows"), "System32", "nvml.dll"),
            # Older drivers
            os.path.join(
                os.getenv("ProgramFiles", "C:\\Program Files"),
                "NVIDIA Corporation",
                "NVSMI",
                "nvml.dll",
            ),
        ]
    if system == "Darwin":
        return [
            "libnvidia-ml.1.dylib",
            "libnvidia-ml.dylib",
            "/usr/local/cuda/lib/libnvidia-ml.1.dylib",
            "/usr/local/cuda/lib/libnvidia-ml.dylib",
        ]
    if system == "Linux":
        bases = [
            "libnvidia-ml.so",  # check library path first
            "/usr/lib64/nvidia/libnvidia-ml.so",  # RHEL/CentOS/Fedora
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",  # Debian/Ubuntu multiarch
            "/usr/lib/wsl/lib/libnvidia-ml.so",  # WSL
        ]
        return [name for base in bases for name in (f"{base}.1", base)]
    return []


def _function(library: DLL, *names: str) -> typing.Any:
    """Return the first of ``names`` that ``library`` exports, newest version first."""
    for name in names:
        with contextlib.suppress(AttributeError):
            return getattr(library, name)
    raise NVIDIAVirtualPackageError(f"NVML does not export {names[-1]}")


def _error_string(library: DLL, status: int) -> str:
    """Describe ``status``, such as ``NVML_ERROR_DRIVER_NOT_LOADED: Driver Not Loaded (9)``.

    The name comes from :class:`NVMLReturn` and the description from ``nvmlErrorString``,
    which works without initializing NVML.
    """
    import ctypes

    name = None
    with contextlib.suppress(ValueError):
        name = NVMLReturn(status).name
    description = None
    with contextlib.suppress(NVIDIAVirtualPackageError):
        function = _function(library, "nvmlErrorString")
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_char_p
        if value := function(status):
            description = value.decode(errors="replace")
    label = f"{name}: {description}" if name and description else name or description
    return f"{label} ({status})" if label else str(status)


def init_nvml() -> DLL:
    """Load and initialize NVML.

    Called only in the detector child process; the caller must call ``nvmlShutdown``.
    """
    import ctypes
    import platform

    library: DLL
    system = platform.system()
    candidates = _candidate_library_names(system)
    if not candidates:
        raise NVIDIAVirtualPackageError(f"Unsupported OS: {system}")
    # NVML uses the C calling convention on every platform, including Windows.
    for candidate in candidates:
        with contextlib.suppress(OSError):
            library = ctypes.CDLL(candidate)
            break
    else:
        raise NVIDIAVirtualPackageError("Failed to load the NVML library")

    init = _function(library, "nvmlInit_v2", "nvmlInit")
    init.argtypes = []
    init.restype = ctypes.c_int
    status = init()
    if status != NVMLReturn.NVML_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to initialize NVML: {_error_string(library, status)}"
        )
    return library


def shutdown_nvml(library: DLL) -> None:
    """Release the resources that :func:`init_nvml` acquired."""
    import ctypes

    library.nvmlShutdown.argtypes = []
    library.nvmlShutdown.restype = ctypes.c_int
    library.nvmlShutdown()


def system_get_cuda_driver_version(library: DLL) -> tuple[int, int]:
    """Return the newest CUDA version supported by the driver as a tuple of (major, minor)"""
    import ctypes

    function = _function(
        library, "nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"
    )
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int(0)
    status = function(ctypes.byref(version))
    if status != NVMLReturn.NVML_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get CUDA driver version: {_error_string(library, status)}"
        )
    return version.value // 1000, (version.value % 1000) // 10


def device_get_count(library: DLL) -> int:
    """Return the number of devices"""
    import ctypes

    function = _function(library, "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount")
    function.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    function.restype = ctypes.c_int
    count = ctypes.c_uint(0)
    status = function(ctypes.byref(count))
    if status != NVMLReturn.NVML_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get device count: {_error_string(library, status)}"
        )
    return count.value


def device_get_compute_capability(library: DLL, index: int) -> tuple[int, int]:
    """Return a tuple of (cc_major, cc_minor) for the device at ``index``"""
    import ctypes

    get_handle = _function(
        library, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"
    )
    # nvmlDevice_t is an opaque pointer.
    get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    get_handle.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    status = get_handle(index, ctypes.byref(handle))
    if status != NVMLReturn.NVML_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get device handle: {_error_string(library, status)}"
        )

    get_capability = _function(library, "nvmlDeviceGetCudaComputeCapability")
    get_capability.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    get_capability.restype = ctypes.c_int
    cc_major = ctypes.c_int(0)
    cc_minor = ctypes.c_int(0)
    status = get_capability(handle, ctypes.byref(cc_major), ctypes.byref(cc_minor))
    if status != NVMLReturn.NVML_SUCCESS:
        raise NVIDIAVirtualPackageError(
            f"Failed to get device compute capability: {_error_string(library, status)}"
        )
    return cc_major.value, cc_minor.value


def probe() -> CUDASystemInfo:
    """Query NVML in-process.

    Raises if no usable driver is found. Devices that cannot be queried are skipped, so the
    driver version and the remaining devices are still reported.
    """
    library = init_nvml()
    try:
        version = system_get_cuda_driver_version(library)
        count = 0
        with contextlib.suppress(Exception):
            count = device_get_count(library)
        capabilities = []
        for index in range(count):
            with contextlib.suppress(NVIDIAVirtualPackageError):
                capabilities.append(device_get_compute_capability(library, index))
    finally:
        shutdown_nvml(library)
    return CUDASystemInfo(version, capabilities)
