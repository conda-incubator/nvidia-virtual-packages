# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nvidia_virtual_packages")
except PackageNotFoundError:
    # package is not installed
    pass
