# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import ctypes
import ctypes.util
import logging
import enum
import os

logger = logging.getLogger(__name__)

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


def init():
    """Initialize the CUDA driver API"""
    status = library.cuInit(0)
    assert status == CUresult.CUDA_SUCCESS, status


def driver_get_version() -> tuple[int, int]:
    """Return the driver version as a tuple of (major, minor)"""
    driver_version = ctypes.c_int(0)
    status = library.cuDriverGetVersion(ctypes.byref(driver_version))
    assert status == CUresult.CUDA_SUCCESS, status
    major = int(driver_version.value / 1000)
    minor = (driver_version.value % 1000) // 10
    return major, minor


def device_get_count() -> int:
    """Return the number of CUDA devices"""
    device_count = ctypes.c_int(0)
    status = library.cuDeviceGetCount(ctypes.byref(device_count))
    assert status == CUresult.CUDA_SUCCESS, status
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
    assert status == CUresult.CUDA_SUCCESS, status
    status = library.cuDeviceGetAttribute(
        ctypes.byref(cc_minor),
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    assert status == CUresult.CUDA_SUCCESS, status
    name = ctypes.create_string_buffer(256)
    status = library.cuDeviceGetName(name, 256, device)
    assert status == CUresult.CUDA_SUCCESS, status
    return (cc_major.value, cc_minor.value, name.value.decode("utf-8"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Found library: %s", library._name)
    logger.debug("Driver version: %s", driver_get_version())
    init()
    device_count = device_get_count()
    logger.info("Device count: %s", device_count)
    for device in range(device_count):
        logger.info("Device %s: %s", device, device_get_attributes(device))
