"""
inference.py — Run inference with a loaded Pleiades model.

Provides:
    predict_file()  — predict a single patient JSON file
    predict_batch() — predict a list/directory of patient JSON files
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from model import Pleiades, LABEL_MAP


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    file: str
    predicted_label: str
    predicted_class: int
    confidence: float
    probabilities: dict[str, float]
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"[ERROR] {self.file}: {self.error}"
        lines = [
            f"File         : {self.file}",
            f"Prediction   : {self.predicted_label}",
            f"Confidence   : {self.confidence*100:.1f}%",
        ]
        for label, prob in self.probabilities.items():
            lines.append(f"  {label:<14}: {prob*100:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core predict functions
# ---------------------------------------------------------------------------

def predict_file(
    model: Pleiades,
    patient_json_path: str | Path,
    device: torch.device | None = None,
    use_cached: bool = False,
) -> PredictionResult:
    """
    Run inference on a single patient JSON file.

    Args:
        model: A loaded Pleiades model (from loader.load_model).
        patient_json_path: Path to patient JSON with a 'regions' key.
        device: Device to run on. Inferred from model if None.
        use_cached: If True, use pre-computed CLS vectors in JSON ('cls_vector' field).

    Returns:
        PredictionResult dataclass.
    """
    if device is None:
        device = next(model.parameters()).device

    path = Path(patient_json_path)

    try:
        with open(path) as f:
            patient = json.load(f)
    except Exception as e:
        return PredictionResult(
            file=str(path), predicted_label="", predicted_class=-1,
            confidence=0.0, probabilities={}, error=f"Failed to load JSON: {e}"
        )

    if "regions" not in patient:
        return PredictionResult(
            file=str(path), predicted_label="", predicted_class=-1,
            confidence=0.0, probabilities={}, error="JSON missing 'regions' key"
        )

    try:
        model.eval()
        with torch.no_grad():
            logits = model({"regions": patient["regions"]}, use_cached=use_cached)
            probs  = F.softmax(logits, dim=-1)[0]
            pred   = logits.argmax(dim=-1).item()
    except Exception as e:
        return PredictionResult(
            file=str(path), predicted_label="", predicted_class=-1,
            confidence=0.0, probabilities={}, error=f"Inference failed: {e}"
        )

    return PredictionResult(
        file=str(path),
        predicted_label=LABEL_MAP[pred],
        predicted_class=pred,
        confidence=probs[pred].item(),
        probabilities={LABEL_MAP[i]: probs[i].item() for i in range(len(LABEL_MAP))},
    )


def predict_batch(
    model: Pleiades,
    inputs: list[str | Path] | str | Path,
    device: torch.device | None = None,
    use_cached: bool = False,
    verbose: bool = True,
) -> list[PredictionResult]:
    """
    Run inference on multiple patient JSON files.

    Args:
        model: A loaded Pleiades model (from loader.load_model).
        inputs: A list of JSON file paths, or a directory to scan recursively.
        device: Device to run on. Inferred from model if None.
        use_cached: Use pre-computed CLS vectors if available in JSON.
        verbose: Print each result as it's computed.

    Returns:
        List of PredictionResult, one per file.
    """
    # If a directory was passed, find all JSONs inside
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        files = sorted(Path(inputs).rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in: {inputs}")
    else:
        files = [Path(p) for p in (inputs if isinstance(inputs, list) else [inputs])]

    results = []
    for i, f in enumerate(files, 1):
        result = predict_file(model, f, device=device, use_cached=use_cached)
        results.append(result)
        if verbose:
            print(f"\n[{i}/{len(files)}] {'-'*50}")
            print(result)

    return results