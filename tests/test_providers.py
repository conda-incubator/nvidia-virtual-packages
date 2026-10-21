# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os

import pytest

from conda.plugins import manager
from conda.plugins.virtual_packages import cuda as builtin_cuda

from nvidia_virtual_packages.cuda import _ctypes, arch, version

DETECTED = ((12, 4), [(9, 0), (8, 0)])


@pytest.fixture
def records(monkeypatch):
    """Return a function that builds the virtual package records with a fake detector."""
    for name in ("CONDA_OVERRIDE_CUDA", "CONDA_OVERRIDE_CUDA_ARCH"):
        monkeypatch.delenv(name, raising=False)

    def builtin_probe():
        raise AssertionError("conda's built-in __cuda probe must not run")

    monkeypatch.setattr(builtin_cuda, "cuda_version", builtin_probe)

    def build(detected=DETECTED):
        calls = []

        def fake_detect():
            calls.append(1)
            return detected

        monkeypatch.setattr(version, "detect", fake_detect)
        monkeypatch.setattr(arch, "detect", fake_detect)
        for cached in (
            version.get_cuda_version,
            arch.get_minimum_sm,
            builtin_cuda.cached_cuda_version,
            manager.get_plugin_manager,
        ):
            cached.cache_clear()
        pm = manager.get_plugin_manager()
        pm.load_plugins(version, arch)  # no-op when already loaded via entry points
        found = {r.name: r for r in pm.get_virtual_package_records()}
        return found, calls

    yield build
    manager.get_plugin_manager.cache_clear()


def test_replaces_builtin_cuda(records):
    found, _ = records()
    assert found["__cuda"].version == "12.4"
    assert found["__cuda"].build == "0"
    assert found["__cuda_arch"].version == "8.0"


def test_no_driver(records):
    found, _ = records(detected=None)
    assert "__cuda" not in found
    assert "__cuda_arch" not in found


def test_no_devices(records):
    found, _ = records(detected=((12, 4), []))
    assert found["__cuda"].version == "12.4"
    assert "__cuda_arch" not in found


def test_overrides_skip_detection(records, monkeypatch):
    monkeypatch.setenv("CONDA_OVERRIDE_CUDA", "12.0")
    monkeypatch.setenv("CONDA_OVERRIDE_CUDA_ARCH", "7.5")
    found, calls = records()
    assert found["__cuda"].version == "12.0"
    assert found["__cuda_arch"].version == "7.5"
    assert not calls


def test_empty_cuda_override_hides_both(records, monkeypatch):
    monkeypatch.setenv("CONDA_OVERRIDE_CUDA", "")
    found, calls = records()
    assert "__cuda" not in found
    assert "__cuda_arch" not in found
    assert not calls


def test_detect_leaves_parent_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _ctypes.detect.cache_clear()
    try:
        result = _ctypes.detect()
    finally:
        _ctypes.detect.cache_clear()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert result is None or isinstance(result[1], list)


def test_detect_permission_error(monkeypatch):
    def deny(method):
        raise PermissionError("denied")

    monkeypatch.setattr("multiprocessing.get_context", deny)
    _ctypes.detect.cache_clear()
    try:
        with pytest.warns(UserWarning, match="permission error"):
            assert _ctypes.detect() is None
    finally:
        _ctypes.detect.cache_clear()
