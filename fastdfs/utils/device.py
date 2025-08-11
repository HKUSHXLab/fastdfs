"""Device related utilities."""
from collections import namedtuple
import os

DeviceInfo = namedtuple('DeviceInfo', ['cpu_count', 'gpu_devices'])

def get_device_info():
    # For FastDFS, we focus on CPU-based processing
    # GPU support can be added later if needed
    gpu_devices = []  # No GPU support for now
    cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))
    return DeviceInfo(cpu_count, gpu_devices)
