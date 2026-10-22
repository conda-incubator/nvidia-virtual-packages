# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from nvidia_virtual_packages.cuda._driver import _candidate_library_names


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Windows", ["nvcuda64.dll", "nvcuda.dll"]),
        (
            "Darwin",
            [
                "libcuda.1.dylib",
                "libcuda.dylib",
                "/usr/local/cuda/lib/libcuda.1.dylib",
                "/usr/local/cuda/lib/libcuda.dylib",
            ],
        ),
        (
            # Every base name tries the ``.1`` SONAME before the bare name.
            "Linux",
            [
                "libcuda.so.1",
                "libcuda.so",
                "/usr/lib64/nvidia/libcuda.so.1",
                "/usr/lib64/nvidia/libcuda.so",
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
                "/usr/lib/wsl/lib/libcuda.so.1",
                "/usr/lib/wsl/lib/libcuda.so",
            ],
        ),
        ("", []),
        ("FreeBSD", []),
        ("Java", []),
    ],
)
def test_candidate_library_names(monkeypatch, system, expected):
    # Pin the reported bitness so the Windows candidate name is deterministic;
    # harmless for the other platforms, which never call ``architecture()``.
    monkeypatch.setattr("platform.architecture", lambda: ("64bit", ""))
    assert _candidate_library_names(system) == expected


def test_probe_reports_driver_without_devices(monkeypatch):
    from nvidia_virtual_packages.cuda import _driver
    from nvidia_virtual_packages.cuda.detection import CUDASystemInfo

    def no_devices(library):
        raise _driver.NVIDIAVirtualPackageError("no devices")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(_driver, "init_driver", object)
    monkeypatch.setattr(_driver, "driver_get_version", lambda library: (12, 4))
    monkeypatch.setattr(_driver, "device_get_count", no_devices)
    assert _driver.probe() == CUDASystemInfo((12, 4), [])


def test_probe_skips_failing_devices(monkeypatch):
    from nvidia_virtual_packages.cuda import _driver
    from nvidia_virtual_packages.cuda.detection import CUDASystemInfo

    def attributes(library, device):
        if device == 1:
            raise _driver.NVIDIAVirtualPackageError("unavailable")
        return [(9, 0), None, (7, 5)][device]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(_driver, "init_driver", object)
    monkeypatch.setattr(_driver, "driver_get_version", lambda library: (12, 4))
    monkeypatch.setattr(_driver, "device_get_count", lambda library: 3)
    monkeypatch.setattr(_driver, "device_get_attributes", attributes)
    assert _driver.probe() == CUDASystemInfo((12, 4), [(9, 0), (7, 5)])


def test_error_string():
    from nvidia_virtual_packages.cuda import _driver

    def returns(text):
        def function(status, value):
            # ``value`` is ``byref(c_char_p)``; its target is the referenced object.
            value._obj.value = text
            return _driver.CUresult.CUDA_SUCCESS

        return staticmethod(function)

    class Library:
        cuGetErrorName = returns(b"CUDA_ERROR_NOT_PERMITTED")
        cuGetErrorString = returns(b"operation not permitted")

    # The driver's name is used even for codes that CUresult lacks.
    assert (
        _driver._error_string(Library(), 800)
        == "CUDA_ERROR_NOT_PERMITTED: operation not permitted (800)"
    )
    # Without the driver's functions, fall back to CUresult, then the number.
    assert _driver._error_string(object(), 100) == "CUDA_ERROR_NO_DEVICE (100)"
    assert _driver._error_string(object(), 12345) == "12345"


def test_error_string_before_init():
    """The driver describes no errors until cuInit succeeds, so cuInit's are named by CUresult."""
    from nvidia_virtual_packages.cuda import _driver

    class Library:
        # CUDA_ERROR_NOT_INITIALIZED
        cuGetErrorName = staticmethod(lambda status, value: 3)
        cuGetErrorString = staticmethod(lambda status, value: 3)

    assert _driver._error_string(Library(), 100) == "CUDA_ERROR_NO_DEVICE (100)"
    assert _driver._error_string(Library(), 12345) == "12345"
