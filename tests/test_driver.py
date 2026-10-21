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
