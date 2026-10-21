# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from nvidia_virtual_packages.cuda._ctypes import _candidate_library_names


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
    monkeypatch.setattr(
        "nvidia_virtual_packages.cuda._ctypes.platform.architecture",
        lambda: ("64bit", ""),
    )
    assert _candidate_library_names(system) == expected
