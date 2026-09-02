"""
GPU utility functions for device selection and memory management.
"""
import torch


def get_optimal_device(requested_device: str = "auto") -> str:
    """Determine optimal compute device based on availability and request.

    Args:
        requested_device: 'auto', 'cuda', or 'cpu'

    Returns:
        Device string ('cuda' or 'cpu')

    Raises:
        RuntimeError if 'cuda' requested but unavailable.
    """
    req = (requested_device or "auto").lower()
    if req == "cpu":
        return "cpu"

    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available on this system.")

    # 'auto' mode: prefer CUDA if available, fallback to CPU
    return "cuda" if torch.cuda.is_available() else "cpu"
