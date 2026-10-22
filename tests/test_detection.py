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
        exitcode = 0

        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args
            modules.append(args[1])

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

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


def test_detect_logs_failure_reason(monkeypatch, spawned, caplog):
    """The reason detection failed in the child is logged in the parent."""
    import multiprocessing

    from nvidia_virtual_packages.cuda import _nvml

    def not_loaded():
        raise _nvml.NVIDIAVirtualPackageError(
            "Failed to initialize NVML: NVML_ERROR_DRIVER_NOT_LOADED: Driver Not Loaded (9)"
        )

    # Run the detector target in this process instead of spawning a child.
    Process = multiprocessing.get_context("spawn").Process
    monkeypatch.setattr(Process, "start", lambda self: self.target(*self.args))
    monkeypatch.setattr(_nvml, "init_nvml", not_loaded)
    monkeypatch.setenv("NVIDIA_VIRTUAL_PACKAGES_DETECTION", "nvml")
    with caplog.at_level("DEBUG", logger=detection.__name__):
        assert detection.detect() is None
    assert "NVML_ERROR_DRIVER_NOT_LOADED: Driver Not Loaded (9)" in caplog.text


def test_detect_logs_missing_result(spawned, caplog):
    with caplog.at_level("DEBUG", logger=detection.__name__):
        assert detection.detect() is None
    assert "exited with code 0 and no result" in caplog.text
