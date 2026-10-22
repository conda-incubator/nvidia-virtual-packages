# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the cold cost of generating the ``__cuda`` and ``__cuda_arch`` packages.

conda invokes the virtual-package hooks on every solve, so the wall-clock cost of
loading the CUDA driver and probing devices adds latency to every conda operation on
NVIDIA machines. These benchmarks measure the *cold* cost — the cache on ``detect`` is
cleared before each timed round so every round runs detection and pays the real driver
load + probe.

Two caveats when running these:

* pytest-benchmark disables timing whenever a trace function (coverage) is active. The
  project's ``addopts`` always passes ``--cov``, so a plain ``pytest`` run still executes
  these tests (they pass) but collects no numbers. To get real measurements, disable
  coverage: ``pytest tests/test_benchmark.py --benchmark-only --no-cov``.
* They ``skip`` on machines without a CUDA driver, keeping CI green off-GPU.
"""

import pytest

from nvidia_virtual_packages.cuda.detection import detect
from nvidia_virtual_packages.cuda.arch import get_minimum_sm
from nvidia_virtual_packages.cuda.version import get_cuda_version


def _clear_caches() -> None:
    """Drop every cached driver-probe result so the next call is cold."""
    detect.cache_clear()


def _cold_setup() -> tuple[tuple[object, ...], dict[str, object]]:
    """pytest-benchmark ``setup``: clear caches, pass no args to the target."""
    _clear_caches()
    return (), {}


@pytest.fixture(autouse=True)
def _require_driver():
    """Skip the benchmarks when no CUDA driver is present."""
    try:
        if detect() is None:
            pytest.skip("no CUDA driver present")
    finally:
        # Ensure the first timed round is cold regardless of the probe above.
        _clear_caches()


@pytest.mark.timeout(0)
def test_benchmark_cuda_version(benchmark):
    # iterations=1 is required: caching makes only the first call per round cold, so we
    # time exactly one cold call and re-clear the caches via setup before the next round.
    benchmark.pedantic(
        get_cuda_version, setup=_cold_setup, rounds=20, iterations=1, warmup_rounds=0
    )


@pytest.mark.timeout(0)
def test_benchmark_cuda_arch(benchmark):
    benchmark.pedantic(
        get_minimum_sm, setup=_cold_setup, rounds=20, iterations=1, warmup_rounds=0
    )
