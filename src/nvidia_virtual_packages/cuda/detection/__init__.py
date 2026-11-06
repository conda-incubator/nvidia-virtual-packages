# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA detection.

The public API is exactly what is listed in ``__all__``; everything else, including every
submodule, is private and may change without notice.
"""

from ._info import CUDASystemInfo, detect

__all__ = ["CUDASystemInfo", "detect"]
