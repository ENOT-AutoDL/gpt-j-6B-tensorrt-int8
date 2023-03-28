from pathlib import Path

import pynvml
import tensorrt as trt

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

_GPU_NAME = None


def get_gpu_name() -> str:
    global _GPU_NAME

    if _GPU_NAME is None:
        pynvml.nvmlInit()
        handler = pynvml.nvmlDeviceGetHandleByIndex(0)
        _GPU_NAME = pynvml.nvmlDeviceGetName(handler)

    return _GPU_NAME


def get_prebuild_engine_filename(fp16: bool = False) -> str:
    filename = get_gpu_name().replace(' ', '_')
    filename += '-'
    filename += trt.__version__.replace('.', '_')
    filename += '-'
    filename += f'i8f{16 if fp16 else 32}.engine'

    return filename


def get_engine() -> Path:
    try:
        path_to_engine = hf_hub_download(
            repo_id='ENOT-AutoDL/gpt-j-6B-tensorrt-int8',
            filename=get_prebuild_engine_filename(),
        )
    except EntryNotFoundError as exc:
        raise RuntimeError('Prebuilded engine for your GPU is not found.') from exc

    return Path(path_to_engine)
