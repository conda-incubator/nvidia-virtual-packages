# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import logging

from nvidia_virtual_packages.cuda.arch import (
    get_minimum_sm,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("__cuda_arch=%s=%s", *get_minimum_sm())
