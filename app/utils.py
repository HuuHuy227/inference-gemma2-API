import os
import gc
import torch
from typing_extensions import Literal, Union

import json
from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from pydantic import BaseModel

DeviceType = Literal["cuda", "mps", "xpu", "npu", "cpu"]
DEVICE_TO_ENV_NAME = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}

from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

def select_device(device):
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
        )

    if device == "auto":
        return get_available_device()
    else:
        if not is_device_available(device):
            raise ValueError(f"{device} is unavailable in your environment")


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def get_available_device() -> DeviceType:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"


def is_device_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    elif device == "mps":
        return torch.backends.mps.is_available()
    elif device == "xpu":
        return is_xpu_available()
    elif device == "npu":
        return is_npu_available()
    elif device == "cpu":
        return True

    return False


def move_model_to_available_device(model):
    device = get_available_device()

    if device == "cpu":
        return model

    return model.to(device)


def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    elif device == "cuda" or device == "mps" or device == "npu":
        return torch.float16
    elif device == "xpu":
        return torch.bfloat16

    return None


def is_hf_accelerate_supported(device: str) -> bool:
    return device == "cuda" or device == "xpu" or device == "npu"


# def empty_cache():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()
#     if is_xpu_available():
#         torch.xpu.empty_cache()
#     if is_npu_available():
#         torch.npu.empty_cache()

def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()

def get_available_device_env_name():
    return DEVICE_TO_ENV_NAME.get(get_available_device())


def gpu_count():
    if torch.cuda.is_available():
        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible_devices_env is None:
            return torch.cuda.device_count()

        cuda_visible_devices = (
            cuda_visible_devices_env.split(",") if cuda_visible_devices_env else []
        )

        return min(torch.cuda.device_count(), len(cuda_visible_devices))
    elif is_xpu_available():
        return torch.xpu.device_count()
    elif is_npu_available():
        return torch.npu.device_count()
    else:
        return 0

def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)

def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)