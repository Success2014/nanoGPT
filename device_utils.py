"""
Device selection helpers for CUDA / Apple MPS / CPU.
"""
import torch


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available accelerator."""
    if device != 'auto':
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError(
                "device='mps' requested but MPS is not available. "
                "Use a recent PyTorch build on Apple Silicon, or pass --device=cpu."
            )
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_device_type(device: str) -> str:
    if 'cuda' in device:
        return 'cuda'
    if device == 'mps':
        return 'mps'
    return 'cpu'


def default_dtype_for_device(device: str) -> str:
    device_type = get_device_type(device)
    if device_type == 'cuda':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if device_type == 'mps':
        # MPS has solid float16 support; bfloat16 is limited on Apple GPUs.
        return 'float16'
    return 'float32'


def setup_device_seeds(seed: int, device_type: str) -> None:
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)


def configure_backends(device_type: str) -> None:
    if device_type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
