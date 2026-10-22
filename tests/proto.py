# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import logging

from nvidia_virtual_packages.cuda.detection import detect

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    result = detect()
    if result is None:
        logger.info("No usable CUDA driver found")
    else:
        logger.info("Driver version: %s", result.driver_version)
        logger.info("Device count: %s", len(result.compute_capabilities))
        for device, capability in enumerate(result.compute_capabilities):
            logger.info("Device %s: %s", device, capability)
