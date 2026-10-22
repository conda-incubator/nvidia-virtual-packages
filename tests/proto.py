# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import logging

from nvidia_virtual_packages.cuda._ctypes import (
    device_get_attributes,
    device_get_count,
    driver_get_version,
    init_driver,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    library = init_driver()
    logger.debug("Found library: %s", library._name)
    logger.debug("Driver version: %s", driver_get_version(library))
    device_count = device_get_count(library)
    logger.info("Device count: %s", device_count)
    for device in range(device_count):
        logger.info("Device %s: %s", device, device_get_attributes(library, device))
