"""
loader.py — Load a Pleiades checkpoint for inference.

Usage:
    from loader import load_model
    model = load_model("checkpoints/best_model.pt")
"""

import torch
from pathlib import Path
from model import Pleiades

# Keys the training code may have used to store the model state dict
_MODEL_STATE_KEYS = ("model", "model_state_dict", "state_dict")


def load_model(
    checkpoint_path: str,
    d_model: int   = 256,
    n_head: int    = 8,
    d_ff: int      = 1024,
    base_layers: int    = 6,
    region_layers: int  = 4,
    sample_layers: int  = 2,
    n_classes: int = 4,
    dropout: float = 0.0,   # set to 0 for inference; no dropout needed
    device: str | torch.device | None = None,
) -> Pleiades:
    """
    Build the Pleiades model, load weights from a checkpoint, and return
    the model in eval mode on the requested device.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved during training.
        d_model, n_head, d_ff, *_layers, n_classes: Architecture params —
            must match what was used during training.
        dropout: Overridden to 0.0 for inference by default.
        device: 'cuda', 'cpu', or a torch.device. Auto-detects CUDA if None.

    Returns:
        Pleiades model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build architecture
    model = Pleiades(
        d_model=d_model,
        n_head=n_head,
        d_ff=d_ff,
        base_layers=base_layers,
        region_layers=region_layers,
        sample_layers=sample_layers,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Find the state dict key (handles different training code conventions)
    if isinstance(ckpt, dict):
        model_key = next((k for k in _MODEL_STATE_KEYS if k in ckpt), None)
        if model_key is None:
            raise KeyError(
                f"Could not find model weights in checkpoint.\n"
                f"Available keys: {list(ckpt.keys())}\n"
                f"Expected one of: {_MODEL_STATE_KEYS}"
            )
        state_dict = ckpt[model_key]
        epoch = ckpt.get("epoch", "unknown")
        best_acc = ckpt.get("best_val_acc", ckpt.get("best_acc", None))
    else:
        # Checkpoint is the raw state dict
        state_dict = ckpt
        epoch = "unknown"
        best_acc = None

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[loader] Loaded checkpoint: {checkpoint_path}")
    print(f"[loader] Device: {device}")
    print(f"[loader] Epoch: {epoch}")
    if best_acc is not None:
        print(f"[loader] Best val accuracy: {best_acc:.4f}")

    return model