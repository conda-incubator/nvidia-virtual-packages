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
