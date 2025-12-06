"""
Ontology Declaration - Intelligence Algorithm (Version 2)

Existence Nature: Systemic Capacity Module
Purpose:
  - Provide the system-wide "Intelligence" capacities: learning potential, pattern-abstraction,
    memory consolidation, generalization potential, metacognitive monitors, neuromodulatory
    balance controls, temporal coordination, and efficiency constraints.
  - Expose capacity vectors (interpretable 0..1 dimensions) usable by Orchestrator, Evolution,
    Symmetry and Reasoning subsystems.
  - Provide auditable learning updates, memory operations, and capacity mutation APIs.

Granted Capacities:
  - Learning update (Bayesian + plasticity proxy) producing posterior capacity estimates.
  - Pattern abstraction primitives and hierarchical feature smoothing.
  - Memory consolidation (short-term -> long-term) with replay & prioritization hooks.
  - Metacognitive monitoring (confidence, uncertainty, error-detection) telemetry.
  - Neuromodulatory controls for exploration/exploitation and temporal coordination.
  - Efficiency metrics: compute/energy proxies and sparsity enforcement.
  - Integration hooks for Symmetry and Evolution (callbacks).

Interactions:
  - Consumes sensory/representation vectors (from perception/vision modules).
  - Emits capacity vectors, learning traces, and metacognitive signals to orchestrator.
  - Receives mutation events from Evolution engine (to update capacity vector) and reports back.
"""

from typing import Dict, Any, Callable, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import uuid
import copy
import math
import logging

# Setup logger
logger = logging.getLogger("intelligence_v2")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------
# Utilities
# ----------------------------
def now_ts() -> float:
    return time.time()


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


# ----------------------------
# Dataclasses for traces & telemetry
# ----------------------------
@dataclass
class LearningTrace:
    trace_id: str
    prior: np.ndarray
    data_summary: np.ndarray
    posterior: np.ndarray
    plasticity_delta: Optional[np.ndarray]
    confidence: float
    timestamp: float = field(default_factory=now_ts)


# ----------------------------
# IntelligenceAlgorithmV2
# ----------------------------
class IntelligenceAlgorithmV2:
    """
    Intelligence (Capacity) v2
    """

    def __init__(self,
                 capacity_dim: int = 16,
                 rng_seed: Optional[int] = None,
                 base_learning_rate: float = 0.05,
                 sparsity_threshold: float = 0.05):
        self.capacity_dim = int(capacity_dim)
        self._rng = np.random.default_rng(rng_seed)

        # core parameters
        self.base_learning_rate = float(base_learning_rate)
        self.sparsity_threshold = float(sparsity_threshold)

        # capacity vector describes latent capacity dimensions (0..1)
        # dims might correspond to: learning, pattern_abstraction, memory_consolidation,
        # social_internalization, generalization, metacognition, embodiment, temporal_coordination, efficiency...
        self.capacity_vector = (0.5 * np.ones(self.capacity_dim)).tolist()

        # memory stores (simple representation for now)
        self.short_term_store: List[np.ndarray] = []
        self.long_term_store: List[np.ndarray] = []

        # neuromodulatory state: dict of modulatory gains (0..1)
        self.neuromodulatory: Dict[str, float] = {
            'dopamine_like': 0.5,   # exploration/exploitation bias
            'serotonin_like': 0.5,  # value stability / mood proxy
            'acetylcholine_like': 0.5  # attention / uncertainty signal
        }

        # telemetry
        self.telemetry: Dict[str, Any] = {
            'learning_traces': [],
            'capacity_history': [],
            'memory_sizes': {'short_term': 0, 'long_term': 0},
            'efficiency_metrics': []
        }

        # callbacks / integration hooks
        # signature: cb(event_dict)
        self.symmetry_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.evolution_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.orchestrator_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    # ----------------------------
    # Callback registration
    # ----------------------------
    def register_symmetry_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.symmetry_callbacks.append(cb)

    def register_evolution_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.evolution_callbacks.append(cb)

    def register_orchestrator_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.orchestrator_callbacks.append(cb)

    def _emit_symmetry(self, payload: Dict[str, Any]) -> None:
        for cb in self.symmetry_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("symmetry callback failed")

    def _emit_evolution(self, payload: Dict[str, Any]) -> None:
        for cb in self.evolution_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("evolution callback failed")

    def _emit_orchestrator(self, payload: Dict[str, Any]) -> None:
        for cb in self.orchestrator_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("orchestrator callback failed")

    # ----------------------------
    # Capacity APIs
    # ----------------------------
    def get_capacity_vector(self) -> np.ndarray:
        return np.array(self.capacity_vector, dtype=float)

    def set_capacity_vector(self, v: np.ndarray) -> None:
        v = np.array(v, dtype=float)
        if v.size != self.capacity_dim:
            raise ValueError("capacity vector dimension mismatch")
        self.capacity_vector = np.clip(v, 0.0, 1.0).tolist()
        self.telemetry['capacity_history'].append({'ts': now_ts(), 'capacity': self.capacity_vector.copy()})

    # ----------------------------
    # Learning update (variational + plasticity proxy)
    # ----------------------------
    def learn_update(self,
                     data_vector: np.ndarray,
                     prior: Optional[np.ndarray] = None,
                     num_samples: int = 128,
                     plasticity_gain: Optional[float] = None) -> LearningTrace:
        """
        Perform a Bayesian-style posterior estimate with Hebbian-like plasticity delta.
        Returns LearningTrace that includes posterior and confidence.
        """
        data_vector = np.array(data_vector, dtype=float)
        dim = self.capacity_dim

        if prior is None:
            prior = np.ones(dim) * 0.5
        prior = np.array(prior, dtype=float)

        # sample around prior (simple normal approx)
        samples = self._rng.normal(loc=prior, scale=0.1, size=(int(num_samples), dim))
        posterior = np.mean(samples, axis=0)

        # plasticity: scaled by novelty (difference between data summary and prior)
        data_summary = np.atleast_1d(data_vector)
        if data_summary.size != dim:
            # compress/expand with simple procrustes-like mapping (mean-based)
            data_summary = np.resize(data_summary, dim)

        novelty = float(np.mean(np.abs(data_summary - prior)))
        learning_rate = float(plasticity_gain if plasticity_gain is not None else self.base_learning_rate)
        plasticity_delta = learning_rate * (data_summary - prior) * (1.0 + novelty)

        # sparsity gating (encourage sparse updates if variance high)
        sample_std = float(np.std(samples))
        if sample_std > self.sparsity_threshold:
            posterior = posterior + 0.5 * plasticity_delta
        else:
            posterior = posterior + 0.25 * plasticity_delta

        # clamp and set as capacity update suggestion (not forced)
        posterior_clamped = np.clip(posterior, 0.0, 1.0)

        # confidence estimate (inverse uncertainty)
        confidence = clamp01(1.0 - sample_std)

        trace = LearningTrace(
            trace_id=str(uuid.uuid4()),
            prior=prior,
            data_summary=data_summary,
            posterior=posterior_clamped,
            plasticity_delta=plasticity_delta,
            confidence=float(confidence)
        )

        # store telemetry
        self.telemetry['learning_traces'].append({
            'id': trace.trace_id,
            'ts': now_ts(),
            'confidence': trace.confidence,
            'novelty': novelty
        })

        # optionally apply small update to capacity_vector (conservative)
        current = self.get_capacity_vector()
        applied = current + 0.1 * (posterior_clamped - current)
        self.set_capacity_vector(applied)

        # emit metacognitive signal if low confidence
        if trace.confidence < 0.3:
            self._emit_symmetry({'type': 'low_confidence', 'trace_id': trace.trace_id, 'confidence': trace.confidence})

        return trace

    # ----------------------------
    # Pattern abstraction and hierarchical smoothing
    # ----------------------------
    def enhance_pattern_abstraction(self, features: np.ndarray, window: int = 3) -> np.ndarray:
        """
        A light-weight hierarchical smoothing / abstraction using convolutional smoothing
        and top-k sparsification to mimic cortical column pooling behavior.
        """
        f = np.array(features, dtype=float)
        if f.ndim > 1:
            # collapse time dimension by mean
            f = f.mean(axis=0)
        # simple moving average as hierarchy proxy
        kernel = np.ones(window) / float(window)
        padded = np.pad(f, (window//2, window//2), mode='edge')
        abstracted = np.convolve(padded, kernel, mode='valid')
        # sparsify: keep
