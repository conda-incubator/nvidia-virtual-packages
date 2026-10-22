# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""The interface between the virtual packages and CUDA detection.

The virtual package modules depend only on this module, so the detection method can be
replaced without changing how the virtual packages or their overrides are constructed.

The policy around detection follows conda's built-in ``__cuda`` detector
(``conda/plugins/virtual_packages/cuda.py``): detection runs in a spawned child process with a
timeout, so a hung or crashing driver cannot take conda down, and Apple silicon is skipped.

The method is chosen by the ``NVIDIA_VIRTUAL_PACKAGES_DETECTION`` environment variable:
``driver`` (the default) uses the CUDA driver API (:func:`nvidia_virtual_packages.cuda._driver.probe`),
and ``nvml`` uses NVML (:func:`nvidia_virtual_packages.cuda._nvml.probe`).
"""

from __future__ import annotations

import functools
import typing

if typing.TYPE_CHECKING:
    import multiprocessing.queues

#: The modules implementing each detection method; each provides ``probe() -> CUDASystemInfo``.
_METHODS = {
    "driver": "nvidia_virtual_packages.cuda._driver",
    "nvml": "nvidia_virtual_packages.cuda._nvml",
}


class CUDASystemInfo(typing.NamedTuple):
    """The CUDA capabilities detected on the system."""

    #: The newest CUDA version supported by the driver, as ``(major, minor)``.
    driver_version: tuple[int, int]
    #: The compute capability of every device on the system that could be queried, as
    #: ``(major, minor)``, regardless of any device visibility settings; empty if none were
    #: found.
    compute_capabilities: list[tuple[int, int]]


def _detector_target(
    queue: multiprocessing.queues.SimpleQueue[CUDASystemInfo | str], module: str
) -> None:
    """Run the detection method in ``module`` in the child process; put the result on ``queue``.

    If detection fails, the reason is put on ``queue`` instead, so the parent can log it.
    """
    try:
        import importlib

        queue.put(importlib.import_module(module).probe())
    except Exception as e:
        queue.put(f"{type(e).__name__}: {e}")


@functools.cache
def detect() -> CUDASystemInfo | None:
    """Return the CUDA capabilities of the system, or ``None`` if no usable driver is found.

    The result is cached, so detection runs at most once per process. Why detection failed is
    logged at the debug level, which conda shows with ``-vvv``.
    """
    import logging
    import multiprocessing
    import os
    import platform
    import warnings

    method = (
        os.getenv("NVIDIA_VIRTUAL_PACKAGES_DETECTION", "").strip().lower() or "driver"
    )
    if method not in _METHODS:
        warnings.warn(
            f"Unknown CUDA detection method in NVIDIA_VIRTUAL_PACKAGES_DETECTION ({method}). "
            f"Using the CUDA driver API instead. Valid methods are: {', '.join(_METHODS)}",
            stacklevel=2,
        )
        method = "driver"

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return None

    try:
        context = multiprocessing.get_context("spawn")
        queue = context.SimpleQueue()
    except PermissionError as e:
        warnings.warn(
            f"Unable to detect CUDA due to permission error: {e}. "
            "Assuming CUDA is not available.",
            stacklevel=2,
        )
        return None
    detector = context.Process(
        target=_detector_target,
        args=(queue, _METHODS[method]),
        name="CUDA driver detector",
        daemon=True,
    )
    try:
        detector.start()
        detector.join(timeout=60.0)
        timed_out = detector.is_alive()
    finally:
        detector.kill()
    log = logging.getLogger(__name__)
    if queue.empty():
        if timed_out:
            log.debug("CUDA detection (%s) timed out", method)
        else:
            log.debug(
                "CUDA detection (%s) exited with code %s and no result",
                method,
                detector.exitcode,
            )
        return None
    result: CUDASystemInfo | str = queue.get()
    if isinstance(result, str):
        log.debug("CUDA detection (%s) failed: %s", method, result)
        return None
    return result
