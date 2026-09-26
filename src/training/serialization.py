"""Portable serialization for trusted TRACE tabular model snapshots."""
from __future__ import annotations

import io
import pickle


class _TrustedTabularUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Older ``python -m`` runs wrote these classes under ``__main__``.
        # This compatibility mapping does not make untrusted pickle safe.
        if module == "__main__" and name in {
            "CheapBaseline", "TabularBaseline", "SafeFeatureExtractor"
        }:
            from src.training import tabular
            return getattr(tabular, name)
        return super().find_class(module, name)


def load_trusted_tabular(payload: bytes):
    """Read our own snapshots, including legacy CLI class references."""
    return _TrustedTabularUnpickler(io.BytesIO(payload)).load()


def serialize_tabular_model(model) -> bytes:
    """Write canonical importable classes while retaining all fitted state."""
    original = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    canonical = load_trusted_tabular(original)
    return pickle.dumps(canonical, protocol=pickle.HIGHEST_PROTOCOL)
