# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os

import pytest

from nvidia_virtual_packages.cuda import detection


@pytest.fixture(autouse=True)
def _cold_detect():
    """Clear the detection cache before and after each test."""
    detection.detect.cache_clear()
    yield
    detection.detect.cache_clear()


def test_detect_leaves_parent_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = detection.detect()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert result is None or isinstance(result.compute_capabilities, list)


def test_detect_permission_error(monkeypatch):
    def deny(method):
        raise PermissionError("denied")

    # Detection, and so the permission error, is skipped on Apple silicon.
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("multiprocessing.get_context", deny)
    with pytest.warns(UserWarning, match="permission error"):
        assert detection.detect() is None


def test_detect_skips_apple_silicon(monkeypatch):
    def no_spawn(method):
        raise AssertionError("must not spawn a detector on Apple silicon")

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("multiprocessing.get_context", no_spawn)
    assert detection.detect() is None


@pytest.fixture
def spawned(monkeypatch):
    """Record the method module that detect() passes to the child, without spawning."""
    import multiprocessing

    modules = []
    context = multiprocessing.get_context("spawn")

    class Process:
        def __init__(self, target, args, **kwargs):
            modules.append(args[1])

        def start(self):
            pass

        def join(self, timeout):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(context, "Process", Process)
    monkeypatch.setattr("multiprocessing.get_context", lambda method: context)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    return modules


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "nvidia_virtual_packages.cuda._driver"),
        ("", "nvidia_virtual_packages.cuda._driver"),
        ("driver", "nvidia_virtual_packages.cuda._driver"),
        (" NVML ", "nvidia_virtual_packages.cuda._nvml"),
    ],
)
def test_detect_selects_method(monkeypatch, spawned, value, expected):
    if value is None:
        monkeypatch.delenv("NVIDIA_VIRTUAL_PACKAGES_DETECTION", raising=False)
    else:
        monkeypatch.setenv("NVIDIA_VIRTUAL_PACKAGES_DETECTION", value)
    assert detection.detect() is None
    assert spawned == [expected]


def test_detect_unknown_method_uses_driver(monkeypatch, spawned):
    monkeypatch.setenv("NVIDIA_VIRTUAL_PACKAGES_DETECTION", "bogus")
    with pytest.warns(UserWarning, match="Unknown CUDA detection method"):
        assert detection.detect() is None
    assert spawned == ["nvidia_virtual_packages.cuda._driver"]
