"""
run_inference.py — Inference engine, FastAPI-ready.

Exposes:
    PleiadesInferenceEngine   — singleton class: load once, call many times
    PredictionResult          — typed output dataclass
    get_engine()              — returns the global engine instance (for FastAPI lifespan)

FastAPI usage:
    from run_inference import get_engine, PredictionResult
    engine = get_engine()
    result = engine.predict(patient_dict)

Direct Python usage:
    engine = PleiadesInferenceEngine("best_model.pt")
    result = engine.predict(patient_dict)
    result = engine.predict_json_file("patient.json")
"""

import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from loader import load_model
from model import Pleiades, LABEL_MAP


# ---------------------------------------------------------------------------
# Output schema (FastAPI can use this directly as a response_model)
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    predicted_label: str
    predicted_class: int
    confidence: float
    probabilities: dict[str, float]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine — load once, reuse forever
# ---------------------------------------------------------------------------

class PleiadesInferenceEngine:
    """
    Wraps the model. Instantiate once at startup, call predict() per request.

    Args:
        checkpoint_path: Path to trained .pt checkpoint.
        device: 'cuda', 'cpu', or None (auto-detect).
        **model_kwargs: Architecture params (d_model, d_ff, etc.) — must match training.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        **model_kwargs,
    ):
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Pleiades = load_model(
            checkpoint_path=checkpoint_path,
            device=self._device,
            **model_kwargs,
        )
        self._model.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Core predict — accepts a dict (the 'regions' structure directly)
    # ------------------------------------------------------------------

    def predict(
        self,
        patient: dict,
        use_cached: bool = False,
    ) -> PredictionResult:
        """
        Run inference on a patient dict.

        Args:
            patient: Dict with a 'regions' key. This is exactly what you'd
                     get from json.load() on a patient JSON file.
            use_cached: If True, use 'cls_vector' fields from the JSON
                        instead of running the base encoder.

        Returns:
            PredictionResult dataclass (call .to_dict() for JSON serialization).
        """
        if "regions" not in patient:
            return PredictionResult(
                predicted_label="", predicted_class=-1, confidence=0.0,
                probabilities={}, error="Input dict missing 'regions' key"
            )

        try:
            with torch.no_grad():
                logits = self._model({"regions": patient["regions"]}, use_cached=use_cached)
                probs  = F.softmax(logits, dim=-1)[0]
                pred   = logits.argmax(dim=-1).item()
        except Exception as e:
            return PredictionResult(
                predicted_label="", predicted_class=-1, confidence=0.0,
                probabilities={}, error=f"Inference failed: {e}"
            )

        return PredictionResult(
            predicted_label=LABEL_MAP[pred],
            predicted_class=int(pred),
            confidence=float(probs[pred]),
            probabilities={LABEL_MAP[i]: float(probs[i]) for i in range(len(LABEL_MAP))},
        )

    # ------------------------------------------------------------------
    # Convenience: predict directly from a JSON file path
    # ------------------------------------------------------------------

    def predict_json_file(
        self,
        path: str | Path,
        use_cached: bool = False,
    ) -> PredictionResult:
        """Load a patient JSON file and run predict()."""
        try:
            with open(path) as f:
                patient = json.load(f)
        except Exception as e:
            return PredictionResult(
                predicted_label="", predicted_class=-1, confidence=0.0,
                probabilities={}, error=f"Failed to load JSON: {e}"
            )
        return self.predict(patient, use_cached=use_cached)


# ---------------------------------------------------------------------------
# Global singleton — for FastAPI lifespan / dependency injection
# ---------------------------------------------------------------------------

_engine: Optional[PleiadesInferenceEngine] = None


def init_engine(checkpoint_path: str, device: str | None = None, **model_kwargs) -> None:
    """
    Call this once at app startup (e.g. inside FastAPI lifespan).

    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            init_engine("checkpoints/best_model.pt")
            yield

        app = FastAPI(lifespan=lifespan)
    """
    global _engine
    _engine = PleiadesInferenceEngine(checkpoint_path, device=device, **model_kwargs)


def get_engine() -> PleiadesInferenceEngine:
    """
    FastAPI dependency — returns the global engine.

    Example:
        @app.post("/predict")
        def predict(body: PatientInput, engine: PleiadesInferenceEngine = Depends(get_engine)):
            return engine.predict(body.dict()).to_dict()
    """
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() at startup.")
    return _engine