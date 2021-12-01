"""Module of runtime utils"""
import os
import logging
import time
from typing import List, Tuple

import torch
import pynvml

# from DReinQ import Consts
def preAllocateMem(memSize: int):
    """Pre-allocate VRAM in GPUs.

    Args:
        memSize (int): Preserved VRAM amount (MiB).
    """
    devices = torch.cuda.device_count()
    for d in range(devices):
        x = torch.rand((256, 1024, memSize), device=f"cuda:{d}")
        del x
    return

def _GPUInfo():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    gpus = list()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({ "memory.used": info.used / 1048576, "memory.total": info.total / 1048576 })
    return gpus

def queryGPU(wantsMore: bool = False, givenList: list = None, needGPUs: int = -1, needVRamEachGPU: int = -1, writeOSEnv: bool = True) -> List[Tuple[int, int]]:
    """Query GPUs that meet requirements.

    Args:
        wantsMore (bool, optional): Ignore 'needGPUs' parameter. Defaults to False.
        givenList (list, optional): If given, only GPUs id in this list will be queried. Defaults to None.
        needGPUs (int, optional): How many GPUs take in demand. Defaults to -1.
        needVRamEachGPU (int, optional): It is OK to use a GPU if free VRAM (MiB) is larger than this threshold. Defaults to -1.
        writeOSEnv (bool, optional): Overwrite env parameter 'CUDA_VISIBLE_DEVICES'. Defaults to True.
        logger (logging.Logger, optional): For logging. Defaults to None.

    Raises:
        EnvironmentError: No GPU satisfied requirements.

    Returns:
        List[Tuple[int, int]]: List of available gpu id and free VRAM.
    """

    # logger = logger or Consts.Logger

    # keep the devices order same as in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    gpus = _GPUInfo()
    if needGPUs < 0:
        needGPUs = len(gpus)

    needGPUs = min(needGPUs, len(givenList))

    # logger.debug("\n" + str(gpus))
    if isinstance(givenList, list):
        it = givenList
    else:
        it = range(len(gpus))

    gpus = [(i, gpus[i]) for i in it]
    if wantsMore:
        gpus = sorted(gpus, key=lambda item: item[1]['memory.used'])

    gpuList = []
    for i, g in gpus:
        if needVRamEachGPU < 0:
            if g['memory.used'] < 64:
                # give space for basic vram
                gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
                # logger.debug("adding gpu[%d] with %f free.", i, g['memory.total'] - g['memory.used'])
        elif g['memory.total'] - g['memory.used'] > needVRamEachGPU + 64:
            gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
            # logger.debug("adding gpu[%d] with %f free.", i, g['memory.total'] - g['memory.used'])
        if len(gpuList) >= needGPUs and not wantsMore:
            break

    if len(gpuList) >= needGPUs:
        # keep order
        gpuList = sorted(gpuList, key=lambda item: item[0])
        # logger.debug("Found %d %s satisfied", len(gpuList), "gpu" if len(gpuList) == 1 else "gpus")
        if writeOSEnv:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [item[0] for item in gpuList]))
            newGPUList = []
            j = 0
            for i, mem in gpuList:
                newGPUList.append((j, mem))
                j += 1
            gpuList = newGPUList
        else:
            try:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            except KeyError:
                pass
        return gpuList
    else:
        raise EnvironmentError("Current system status is not satisfied")


class Timer:
    """A simple timer

    Example:
    ```python
        # Start the timer
        timer = Timer()
        ...
        # Last interval and total spent from start
        # 3.0,    3.0
        interval, total = timer.Tick()
        ...
        # 4.4,    7.4
        interval, total = timer.Tick()
    ```
    """
    def __init__(self):
        self._initial = self._tick = time.time()

    def Tick(self):
        """Tick-tock

        Returns:
            float: Interval from last tick.
            float: Total spent from init.
        """
        tock = time.time()
        interval = tock - self._tick
        self._tick = tock
        return interval, self._tick - self._initial
