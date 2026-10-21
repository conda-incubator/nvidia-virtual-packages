# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os

import pytest

from nvidia_virtual_packages.cuda import _nvml
from nvidia_virtual_packages.cuda.detection import CUDASystemInfo


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        (
            "Windows",
            [
                "nvml.dll",
                os.path.join("C:\\Win", "System32", "nvml.dll"),
                os.path.join("C:\\PF", "NVIDIA Corporation", "NVSMI", "nvml.dll"),
            ],
        ),
        (
            "Darwin",
            [
                "libnvidia-ml.1.dylib",
                "libnvidia-ml.dylib",
                "/usr/local/cuda/lib/libnvidia-ml.1.dylib",
                "/usr/local/cuda/lib/libnvidia-ml.dylib",
            ],
        ),
        (
            # Every base name tries the ``.1`` SONAME before the bare name.
            "Linux",
            [
                "libnvidia-ml.so.1",
                "libnvidia-ml.so",
                "/usr/lib64/nvidia/libnvidia-ml.so.1",
                "/usr/lib64/nvidia/libnvidia-ml.so",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
                "/usr/lib/wsl/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/lib/libnvidia-ml.so",
            ],
        ),
        ("", []),
        ("FreeBSD", []),
    ],
)
def test_candidate_library_names(monkeypatch, system, expected):
    monkeypatch.setenv("WINDIR", "C:\\Win")
    monkeypatch.setenv("ProgramFiles", "C:\\PF")
    assert _nvml._candidate_library_names(system) == expected


@pytest.fixture
def shutdowns(monkeypatch):
    """Fake NVML initialization, and record each shutdown."""
    calls = []
    monkeypatch.setattr(_nvml, "init_nvml", object)
    monkeypatch.setattr(_nvml, "shutdown_nvml", calls.append)
    return calls


def test_probe(monkeypatch, shutdowns):
    capabilities = {0: (9, 0), 1: (8, 0)}
    monkeypatch.setattr(
        _nvml, "system_get_cuda_driver_version", lambda library: (12, 4)
    )
    monkeypatch.setattr(_nvml, "device_get_count", lambda library: 2)
    monkeypatch.setattr(
        _nvml,
        "device_get_compute_capability",
        lambda library, index: capabilities[index],
    )
    assert _nvml.probe() == CUDASystemInfo((12, 4), [(9, 0), (8, 0)])
    assert len(shutdowns) == 1


def test_probe_reports_driver_without_devices(monkeypatch, shutdowns):
    def no_devices(library):
        raise _nvml.NVIDIAVirtualPackageError("no devices")

    monkeypatch.setattr(
        _nvml, "system_get_cuda_driver_version", lambda library: (12, 4)
    )
    monkeypatch.setattr(_nvml, "device_get_count", no_devices)
    assert _nvml.probe() == CUDASystemInfo((12, 4), [])
    assert len(shutdowns) == 1


def test_probe_shuts_down_after_error(monkeypatch, shutdowns):
    def no_version(library):
        raise _nvml.NVIDIAVirtualPackageError("no version")

    monkeypatch.setattr(_nvml, "system_get_cuda_driver_version", no_version)
    with pytest.raises(_nvml.NVIDIAVirtualPackageError):
        _nvml.probe()
    assert len(shutdowns) == 1


def test_function_falls_back_to_older_symbol():
    class Library:
        nvmlInit = "v1"

    assert _nvml._function(Library(), "nvmlInit_v2", "nvmlInit") == "v1"
    with pytest.raises(_nvml.NVIDIAVirtualPackageError):
        _nvml._function(Library(), "nvmlShutdown")
