import logging

from nvidia_virtual_packages.cuda.arch import *

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Found library: %s", library._name)
    logger.debug("Driver version: %s", driver_get_version())
    init_driver()
    device_count = device_get_count()
    logger.info("Device count: %s", device_count)
    for device in range(device_count):
        logger.info("Device %s: %s", device, device_get_attributes(device))
