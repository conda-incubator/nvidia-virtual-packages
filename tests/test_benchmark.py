# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the cold cost of generating the ``__cuda`` and ``__cuda_arch`` packages.

conda invokes the virtual-package hooks on every solve, so the wall-clock cost of
loading the CUDA driver and probing devices adds latency to every conda operation on
NVIDIA machines. These benchmarks measure the *cold* cost — the ``functools`` caches on
``init_driver``/``get_cuda_version``/``get_minimum_sm`` are cleared before each timed
round so every round pays the real driver load + probe rather than a cache hit.

Two caveats when running these:

* pytest-benchmark disables timing whenever a trace function (coverage) is active. The
  project's ``addopts`` always passes ``--cov``, so a plain ``pytest`` run still executes
  these tests (they pass) but collects no numbers. To get real measurements, disable
  coverage: ``pytest tests/test_benchmark.py --benchmark-only --no-cov``.
* They ``skip`` on machines without a CUDA driver, keeping CI green off-GPU.
"""

import pytest

from nvidia_virtual_packages.cuda._ctypes import NVIDIAVirtualPackageError, init_driver
from nvidia_virtual_packages.cuda.arch import get_minimum_sm
from nvidia_virtual_packages.cuda.version import get_cuda_version


def _clear_caches() -> None:
    """Drop every cached driver-probe result so the next call is cold."""
    init_driver.cache_clear()
    get_cuda_version.cache_clear()
    get_minimum_sm.cache_clear()


def _cold_setup() -> tuple[tuple[object, ...], dict[str, object]]:
    """pytest-benchmark ``setup``: clear caches, pass no args to the target."""
    _clear_caches()
    return (), {}


@pytest.fixture(autouse=True)
def _require_driver():
    """Skip the benchmarks when no CUDA driver is present."""
    try:
        init_driver()
    except NVIDIAVirtualPackageError:
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
