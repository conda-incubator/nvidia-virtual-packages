# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def __getattr__(name):
    # Resolve __version__ lazily; importlib.metadata is costly to import and is not needed
    # on the plugin's hot path.
    if name == "__version__":
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("nvidia_virtual_packages")
        except PackageNotFoundError:
            # package is not installed
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
