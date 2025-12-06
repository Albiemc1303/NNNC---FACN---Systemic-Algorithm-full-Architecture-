"""
Ontology Declaration - Reasoning Algorithm (Version 2)

Existence Nature: Systemic Process Module (Inference Engine)
Purpose:
  - Provide the formalized "Reasoning" process: deductive, inductive, abductive, analogical,
    causal inference, mental simulation, strategy planning, context adaptation, epistemic monitoring,
    and self-correction.
  - Offer structured reasoning traces and confidence assessments suitable for Symmetry & CriticalThinking.
  - Integrate with Intelligence capacities and emit audit-friendly reasoning traces and provenance.

Granted Capacities:
  - Multi-mode inference primitives (deduction / induction / abduction / analogy).
  - Causal inference scorer and temporal factoring.
  - Mental simulation engine (lightweight vectorized forward-model rollouts).
  - Strategic planner with means-ends breakdown (abstract).
  - Context-adaptive rule selector and heuristic switching.
  - Epistemic monitoring hooks: source weighting, uncertainty calibration, and contradiction detection.
  - Self-correction: re-evaluate with alternative premises, request additional evidence, or escalate.

Interactions:
  - Accepts inputs from Intelligence (representations & capacity signals).
  - Emits reasoning traces to CriticalThinking & Symmetry for fact-check and credibility evaluation.
  - Accepts enforcement signals (e.g., re-evaluate, quarantine) from Symmetry.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import uuid
import copy
import math
import logging

# Setup logger
logger = logging.getLogger("reasoning_v2")
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
# Dataclasses for traces
# ----------------------------
@dataclass
class ReasoningTrace:
    trace_id: str
    mode: str
    input_vectors: List[np.ndarray]
    output_vector: np.ndarray
    confidence: float
    justification: Dict[str, Any]
    timestamp: float = field(default_factory=now_ts)


# ----------------------------
# ReasoningAlgorithmV2
# ----------------------------
class ReasoningAlgorithmV2:
    """
    Multi-modal reasoning engine with epistemic monitoring and self-correction.
    """

    def __init__(self,
                 vector_dim: int = 16,
                 rng_seed: Optional[int] = None,
                 inference_threshold: float = 0.6):
        self.vector_dim = int(vector_dim)
        self._rng = np.random.default_rng(rng_seed)
        self.inference_threshold = float(inference_threshold)

        # callbacks
        self.symmetry_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.ct_callbacks: List[Callable[[Dict[str, Any]], None]] = []  # critical thinking / consumer
        self.orchestrator_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # telemetry & history
        self.telemetry: Dict[str, Any] = {'traces': [], 'contradictions': []}

    # ----------------------------
    # registration
    # ----------------------------
    def register_symmetry_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.symmetry_callbacks.append(cb)

    def register_critical_thinking_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.ct_callbacks.append(cb)

    def register_orchestrator_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.orchestrator_callbacks.append(cb)

    def _emit_symmetry(self, payload: Dict[str, Any]) -> None:
        for cb in self.symmetry_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("symmetry cb failed")

    def _emit_ct(self, payload: Dict[str, Any]) -> None:
        for cb in self.ct_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("ct cb failed")

    def _emit_orchestrator(self, payload: Dict[str, Any]) -> None:
        for cb in self.orchestrator_callbacks:
            try:
                cb(payload)
            except Exception:
                logger.exception("orchestrator cb failed")

    # ----------------------------
    # Basic inference modes
    # ----------------------------
    def deductive_reasoning(self, premises: List[np.ndarray], rule_matrix: Optional[np.ndarray] = None) -> ReasoningTrace:
        """
        Simple vectorized deduction: combines premises and applies transformation (rule_matrix).
        rule_matrix maps premise-space -> conclusion-space (if None, use average).
        """
        if not premises:
            out = np.zeros(self.vector_dim)
            confidence = 0.0
            just = {'reason': 'no_premises'}
        else:
            stacked = np.stack([np.array(p, dtype=float) for p in premises])
            combined = np.mean(stacked, axis=0)
            if rule_matrix is not None:
                try:
                    out = np.dot(rule_matrix, combined)
                except Exception:
                    out = combined
            else:
                out = combined
            # confidence heuristic: inverse variance across premises
            var = float(np.mean(np.var(stacked, axis=0)))
            confidence = clamp01(1.0 - var)
            just = {'method': 'deduction', 'premise_var': var}

        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            mode='deductive',
            input_vectors=premises,
            output_vector=safe_norm(np.array(out)),
            confidence=confidence,
            justification=just
        )
        self._record_trace(trace)
        return trace

    def inductive_reasoning(self, observations: List[np.ndarray]) -> ReasoningTrace:
        """
        Infer generalization from examples. Provide generalization vector + confidence
        (higher when low observation variance and enough samples).
        """
        if not observations:
            out = np.zeros(self.vector_dim)
            confidence = 0.0
            just = {'reason': 'no_observations'}
        else:
            stacked = np.stack([np.array(o, dtype=float) for o in observations])
            mean = np.mean(stacked, axis=0)
            var = float(np.mean(np.var(stacked, axis=0)))
            # sample-size adjustment
            n = len(observations)
            confidence = clamp01((1.0 - var) * math.log(1 + n) / math.log(1 + n + 1e-9))
            out = safe_norm(mean) * confidence

            just = {'method': 'induction', 'n': n, 'var': var}

        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            mode='inductive',
            input_vectors=observations,
            output_vector=np.array(out),
            confidence=float(confidence),
            justification=just
        )
        self._record_trace(trace)
        return trace

    def abductive_reasoning(self, observations: List[np.ndarray], hypotheses: List[np.ndarray]) -> ReasoningTrace:
        """
        Abductive: select best hypothesis that explains observations (maximizes similarity).
        Returns chosen hypothesis vector and justification.
        """
        if not (observations and hypotheses):
            out = np.zeros(self.vector_dim)
            conf = 0.0
            just = {'reason': 'insufficient_data'}
        else:
            obs_mean = np.mean(np.stack([np.array(o, dtype=float) for o in observations]), axis=0)
            sims = [float(np.dot(safe_norm(np.array(h)), safe_norm(obs_mean))) for h in hypotheses]
            best_idx = int(np.argmax(sims))
            out = safe_norm(np.array(hypotheses[best_idx], dtype=float))
            conf = clamp01(sims[best_idx])
            just = {'method': 'abduction', 'best_idx': best_idx, 'similarities': sims}

        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            mode='abductive',
            input_vectors=observations + hypotheses,
            output_vector=np.array(out),
            confidence=float(conf),
            justification=just
        )
        self._record_trace(trace)
        return trace

    def analogical_reasoning(self, source: np.ndarray, target_prototype: np.ndarray) -> ReasoningTrace:
        """
        Transfer features from source domain to target prototype by similarity weighting.
        """
        s = safe_norm(np.array(source, dtype=float))
        t = safe_norm(np.array(target_prototype, dtype=float))
        similarity = float(np.dot(s, t))
        transfer = s * similarity
        conf = clamp01(similarity)
        just = {'method': 'analogy', 'similarity': similarity}
        trace = ReasoningTrace(trace_id=str(uuid.uuid4()),
                               mode='analogical',
                               input_vectors=[source, target_prototype],
                               output_vector=safe_norm(transfer),
                               confidence=conf,
                               justification=just)
        self._record_trace(trace)
        return trace

    # ----------------------------
    # Causal reasoning & temporal factors
    # ----------------------------
    def causal_reasoning(self, event_a: np.ndarray, event_b: np.ndarray, temporal_gap: float = 1.0) -> Dict[str, Any]:
        """
        Compute a causal strength proxy: similarity * temporal_decay * directional_indicator.
        Returns a structured dict with 'strength' and 'relationship' label.
        """
        a = safe_norm(np.array(event_a, dtype=float))
        b = safe_norm(np.array(event_b, dtype=float))
        sim = float(np.dot(a, b))
        temporal_factor = math.exp(-abs(temporal_gap) / (1.0 + 0.5))  # tunable decay
        # directional indicator: dot with gradient (simple heuristic)
        # here we use element-wise difference
        directionality = float(np.mean(np.sign(b - a) * (b - a)))
        strength = clamp01(sim * temporal_factor * (0.5 + 0.5 * abs(directionality)))
        if strength > 0.75:
            rel = 'strong_causal'
        elif strength > 0.4:
            rel = 'weak_causal'
        else:
            rel = 'no_causal'
        result = {'strength': float(strength), 'relationship': rel, 'sim': sim, 'temporal_factor': temporal_factor}
        # emit to telemetry
        self.telemetry.setdefault('causal_checks', []).append({'ts': now_ts(), **result})
        return result

    # ----------------------------
    # Mental simulation
    # ----------------------------
    def mental_simulation(self, initial_state: np.ndarray, dynamics_fn: Callable[[np.ndarray], np.ndarray], steps: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Lightweight forward rollout using provided vector-dynamics function.
        Returns final state and list of intermediate states.
        """
        s = np.array(initial_state, dtype=float)
        states = [s.copy()]
        for _ in range(max(1, int(steps))):
            s = dynamics_fn(s)
            states.append(s.copy())
        final = safe_norm(s)
        # record trace
        self.telemetry.setdefault('simulation_runs', []).append({'ts': now_ts(), 'steps': steps})
        return final, states

    # ----------------------------
    # Strategic planning (abstract)
    # ----------------------------
    def strategic_plan(self, goal_vec: np.ndarray, current_state_vec: np.ndarray, max_depth: int = 4) -> Dict[str, Any]:
        """
        Produce a minimal abstract plan as a sequence of intermediate vectors (means-ends).
        This is an abstract heuristic planner using linear interpolation + simple pruning.
        """
        cur = np.array(current_state_vec, dtype=float)
        goal = np.array(goal_vec, dtype=float)
        plan = []
        for d in range(1, max_depth + 1):
            alpha = d / float(max_depth)
            step = safe_norm(cur * (1 - alpha) + goal * alpha)
            plan.append(step)
        # score plan plausibility by average similarity chaining
        sims = [float(np.dot(safe_norm(plan[i]), safe_norm(plan[i+1]))) for i in range(len(plan)-1)] if len(plan) > 1 else [1.0]
        plausibility = clamp01(np.mean(sims))
        result = {'plan': plan, 'plausibility': plausibility, 'steps': len(plan)}
        self.telemetry.setdefault('plans', []).append({'ts': now_ts(), 'plausibility': plausibility})
        return result

    # ----------------------------
    # Epistemic monitoring & contradiction detection
    # ----------------------------
    def epistemic_monitor(self, reasoning_trace: ReasoningTrace, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Assess quality of trace by source reliability and internal coherence.
        sources: optional list of {'source_id':..., 'reliability':0..1}
        """
        base_conf = float(reasoning_trace.confidence)
        source_weight = 0.5
        if sources:
            avg_rel = float(np.mean([s.get('reliability', 0.5) for s in sources]))
            adjusted_conf = clamp01(0.6 * base_conf + 0.4 * avg_rel)
        else:
            adjusted_conf = base_conf

        # check contradictions against recent traces
        contradictions = []
        for t in self.telemetry.get('traces', [])[-50:]:
            # simple contradiction: strong negative dot product
            dot = float(np.dot(safe_norm(np.array(t['output_vector'])), safe_norm(reasoning_trace.output_vector)))
            if dot < -0.2:
                contradictions.append({'trace_id': t['trace_id'], 'dot': dot})

        if contradictions:
            # emit to orchestrator/symmetry
            self._emit_symmetry({'type': 'contradiction_detected', 'trace_id': reasoning_trace.trace_id, 'details': contradictions})
            self.telemetry.setdefault('contradictions', []).extend(contradictions)

        result = {'trace_id': reasoning_trace.trace_id, 'adjusted_confidence': adjusted_conf, 'contradictions': contradictions}
        return result

    # ----------------------------
    # Self-correction & re-evaluation
    # ----------------------------
    def self_correct(self, trace: ReasoningTrace, alternative_premises: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Attempt re-evaluation when low-confidence or contradictions exist. Returns result dict and
        possibly a replacement trace suggestion.
        """
        if trace.confidence > 0.6 and not self.telemetry.get('contradictions'):
            return {'action': 'no_correction_needed', 'trace_id': trace.trace_id}

        # re-evaluate using alternative premises if provided
        if alternative_premises:
            new_trace = self.inductive_reasoning(alternative_premises)
            # compare confidence improvement
            improved = new_trace.confidence > trace.confidence
            # emit to critical thinking consumers
            self._emit_ct({'type': 'self_correction', 'original': trace.trace_id, 'replacement': new_trace.trace_id, 'improved': improved})
            return {'action': 're_eval', 'original': trace.trace_id, 'replacement': new_trace.trace_id, 'improved': improved}
        else:
            # request more evidence via orchestrator
            req = {'type': 'request_evidence', 'trace_id': trace.trace_id, 'reason': 'low_confidence_or_contradiction'}
            self._emit_orchestrator(req)
            return {'action': 'request_evidence', 'trace_id': trace.trace_id}

    # ----------------------------
    # Trace recording
    # ----------------------------
    def _record_trace(self, trace: ReasoningTrace) -> None:
        # store minimal serializable info
        serial = {
            'trace_id': trace.trace_id,
            'mode': trace.mode,
            'output_vector': np.asarray(trace.output_vector).tolist(),
            'confidence': float(trace.confidence),
            'justification': trace.justification,
            'ts': now_ts()
        }
        self.telemetry.setdefault('traces', []).append(serial)
        # push to critical thinking / symmetry consumers
        self._emit_ct({'type': 'new_trace', 'trace': serial})
        self._emit_symmetry({'type': 'new_trace_symmetry_check', 'trace_id': trace.trace_id, 'confidence': trace.confidence})

    # ----------------------------
    # Introspection
    # ----------------------------
    def get_telemetry(self) -> Dict[str, Any]:
        return copy.deepcopy(self.telemetry)
