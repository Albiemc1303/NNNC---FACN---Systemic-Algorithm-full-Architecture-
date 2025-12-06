"""
Ontology Declaration - Symmetry Algorithm

Existence Nature: Systemic Imperative Module (Sentient Imperative)
Purpose: Implements Symmetry as the system-level truth-seeking, epistemic-curation,
         and alignment enforcement mechanism. Symmetry continuously sorts, verifies,
         and organizes system knowledge into verified-fact vs recognized-fiction domains,
         computes system-wide symmetry metrics, and issues corrective enforcement
         signals when epistemic disorder is detected.

Granted Capacities:
  - Truth-Alignment Measurement (vector-space & evidence-based)
  - Active Data Curation & Categorization (fact / fiction / uncertain)
  - Epistemic Consistency Enforcement across beliefs and subsystems
  - Anomaly & Reality-Drift Detection
  - Bias Correction & Reconciliation Mechanisms
  - Ethical & Policy Constraint Enforcement (integration hook)
  - Symmetry Scoring & Telemetry (truth-seeking efficiency metrics)

Interactions:
  - Consumes candidate beliefs/inferences from Reasoning and CriticalThinking.
  - Uses evidence traces & provenance from CriticalThinking to validate claims.
  - Instructs Intelligence/Reasoning to re-evaluate or revise beliefs via self-correction hooks.
  - Emits enforcement actions to the Orchestrator/Evolution (e.g., quarantine, re-testing, corrective mutation).
  - Provides telemetry to Monitoring & Governance subsystems.

Notes:
  - This module emphasizes auditable, explainable decisions — all checks produce traces and recommendations.
  - Symmetry is active and continuous: calls to `ingest_observation`/`curate` update internal state and scores.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cosine
import math
import time
import uuid

# -------------------------
# Utilities
# -------------------------
def safe_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    val = np.dot(a, b) / (na * nb)
    val = float(np.clip(val, -1.0, 1.0))
    # map -1..1 -> 0..1 similarity
    return (val + 1.0) / 2.0

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

# -------------------------
# Symmetry Algorithm
# -------------------------
class SymmetryAlgorithm:
    """
    Symmetry (Sentient Imperative) - Version 2
    """

    def __init__(self,
                 truth_threshold: float = 0.7,
                 consistency_weight: float = 0.5,
                 uncertainty_margin: float = 0.15):
        # thresholds
        self.truth_threshold = float(truth_threshold)
        self.consistency_weight = float(consistency_weight)
        self.uncertainty_margin = float(uncertainty_margin)

        # evidence / belief registries
        # evidence_registry: evidence_id -> {'vector':..., 'source_id':..., 'timestamp':..., 'metadata':...}
        self.evidence_registry: Dict[str, Dict[str, Any]] = {}
        # belief_registry: belief_id -> {'vector':..., 'provenance':..., 'status': 'fact'|'fiction'|'uncertain', ...}
        self.belief_registry: Dict[str, Dict[str, Any]] = {}

        # telemetry metrics
        self.metrics: Dict[str, Any] = {
            'total_evidence': 0,
            'total_beliefs': 0,
            'facts_count': 0,
            'fiction_count': 0,
            'uncertain_count': 0,
            'symmetry_score_history': []
        }

        # policy hooks (registered callbacks)
        self.enforcement_callbacks: List[Any] = []
        # short term cache for recent items used in consistency checks
        self.recent_belief_ids: List[str] = []

    # -------------------------
    # Evidence & Belief management
    # -------------------------
    def ingest_evidence(self, vector: np.ndarray, source_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add evidence vector with provenance metadata and return evidence_id."""
        evidence_id = str(uuid.uuid4())
        self.evidence_registry[evidence_id] = {
            'vector': normalize(np.array(vector, dtype=float)),
            'source_id': source_id,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.metrics['total_evidence'] += 1
        return evidence_id

    def register_belief(self, belief_vector: np.ndarray, provenance: Dict[str, Any]) -> str:
        """
        Register a candidate belief/inference into the belief registry.
        Provenance should include reasoning_trace, evidence_ids (if any), source_id, and confidence.
        """
        bid = str(uuid.uuid4())
        vec = normalize(np.array(belief_vector, dtype=float))
        entry = {
            'vector': vec,
            'provenance': provenance,
            'status': 'uncertain',
            'symmetry_score': None,
            'last_checked': None,
            'timestamp': time.time()
        }
        self.belief_registry[bid] = entry
        self.metrics['total_beliefs'] += 1
        self.recent_belief_ids.append(bid)
        # keep recent cache bounded
        if len(self.recent_belief_ids) > 200:
            self.recent_belief_ids.pop(0)
        return bid

    # -------------------------
    # Core truth-seeking & categorization
    # -------------------------
    def compute_truth_alignment(self, belief_vec: np.ndarray, evidence_ids: List[str]) -> Dict[str, Any]:
        """
        Compute alignment score between belief and aggregated evidence.
        Returns support_score [0,1], alignment_metric, and explanation trace.
        """
        if not evidence_ids:
            return {'support_score': 0.0, 'alignment': 0.0, 'trace': {'reason': 'no_evidence'}}

        evidence_vectors = [self.evidence_registry[eid]['vector'] for eid in evidence_ids if eid in self.evidence_registry]
        if not evidence_vectors:
            return {'support_score': 0.0, 'alignment': 0.0, 'trace': {'reason': 'evidence_missing'}}

        agg = np.mean(np.stack(evidence_vectors), axis=0)
        support_score = safe_cosine_similarity(normalize(belief_vec), normalize(agg))
        alignment = float(support_score)  # alias
        trace = {
            'type': 'truth_alignment',
            'evidence_count': len(evidence_vectors),
            'evidence_sources': list({self.evidence_registry[eid]['source_id'] for eid in evidence_ids if eid in self.evidence_registry})
        }
        return {'support_score': support_score, 'alignment': alignment, 'trace': trace}

    def categorize_belief(self, belief_id: str, evidence_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Categorize a registered belief into 'fact', 'fiction', or 'uncertain' according to truth alignment and thresholds.
        Updates registry and metrics; returns the categorization result + trace.
        """
        if belief_id not in self.belief_registry:
            raise KeyError(f"belief_id {belief_id} not found")

        belief = self.belief_registry[belief_id]
        evidence_ids = evidence_ids or belief['provenance'].get('evidence_ids', [])
        res = self.compute_truth_alignment(belief['vector'], evidence_ids)
        support = res['support_score']

        # apply uncertainty margin around threshold to avoid flip-flopping
        lower = max(0.0, self.truth_threshold - self.uncertainty_margin)
        upper = min(1.0, self.truth_threshold + self.uncertainty_margin)

        if support >= upper:
            status = 'fact'
        elif support <= lower:
            status = 'fiction'
        else:
            status = 'uncertain'

        # update registry
        old_status = belief.get('status')
        belief.update({
            'status': status,
            'symmetry_score': support,
            'last_checked': time.time()
        })

        # update metrics counts
        if old_status != status:
            # decrement old count if applicable
            if old_status == 'fact':
                self.metrics['facts_count'] = max(0, self.metrics.get('facts_count', 0) - 1)
            if old_status == 'fiction':
                self.metrics['fiction_count'] = max(0, self.metrics.get('fiction_count', 0) - 1)
            if old_status == 'uncertain':
                self.metrics['uncertain_count'] = max(0, self.metrics.get('uncertain_count', 0) - 1)

            if status == 'fact':
                self.metrics['facts_count'] = self.metrics.get('facts_count', 0) + 1
            elif status == 'fiction':
                self.metrics['fiction_count'] = self.metrics.get('fiction_count', 0) + 1
            else:
                self.metrics['uncertain_count'] = self.metrics.get('uncertain_count', 0) + 1

        # telemetry append
        self.metrics['symmetry_score_history'].append({'belief_id': belief_id, 'score': support, 'timestamp': time.time()})

        return {'belief_id': belief_id, 'status': status, 'support_score': support, 'trace': res['trace']}

    # -------------------------
    # Epistemic consistency & anomaly detection
    # -------------------------
    def epistemic_consistency_check(self, belief_ids: List[str]) -> Dict[str, Any]:
        """
        Measure pairwise consistency across a set of beliefs (by ids).
        Returns mean consistency in [0,1], pairwise matrix summary, and outlier list.
        """
        vectors = [self.belief_registry[b]['vector'] for b in belief_ids if b in self.belief_registry]
        if not vectors:
            return {'consistency': 1.0, 'pairwise': [], 'outliers': []}

        pairwise = []
        n = len(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                sim = safe_cosine_similarity(vectors[i], vectors[j])
                pairwise.append(sim)
        mean_consistency = float(np.mean(pairwise)) if pairwise else 1.0

        # detect outliers: beliefs with average similarity well below mean
        outliers = []
        for idx, b in enumerate(belief_ids):
            if b not in self.belief_registry:
                continue
            sims = [safe_cosine_similarity(self.belief_registry[b]['vector'], self.belief_registry[other]['vector'])
                    for other in belief_ids if other in self.belief_registry and other != b]
            avg_sim = float(np.mean(sims)) if sims else 1.0
            if avg_sim < max(0.0, mean_consistency - 0.25):  # tunable threshold
                outliers.append({'belief_id': b, 'avg_similarity': avg_sim})

        return {'consistency': mean_consistency, 'pairwise': pairwise, 'outliers': outliers}

    def detect_anomaly(self, current_state_vec: np.ndarray, expected_state_vec: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies and reality-drift between current and expected state vectors.
        Returns anomaly boolean, anomaly_score (0..1), and recommended severity.
        """
        sim = safe_cosine_similarity(normalize(current_state_vec), normalize(expected_state_vec))
        anomaly_score = float(1.0 - sim)  # higher -> more anomalous
        is_anomaly = anomaly_score > 0.5  # tunable
        severity = 'low'
        if anomaly_score > 0.75:
            severity = 'high'
        elif anomaly_score > 0.6:
            severity = 'medium'
        trace = {'type': 'anomaly_detection', 'similarity': sim, 'anomaly_score': anomaly_score}
        return {'is_anomaly': is_anomaly, 'anomaly_score': anomaly_score, 'severity': severity, 'trace': trace}

    # -------------------------
    # Bias correction & reconciliation
    # -------------------------
    def bias_correction(self, biased_vec: np.ndarray, truth_vec: np.ndarray, correction_rate: float = 0.2) -> Dict[str, Any]:
        """Correct biased belief vector toward truth vector; return corrected vector + trace."""
        corrected = normalize(biased_vec + correction_rate * (normalize(truth_vec) - normalize(biased_vec)))
        trace = {'type': 'bias_correction', 'correction_rate': correction_rate}
        return {'corrected_vector': corrected, 'trace': trace}

    # -------------------------
    # Enforcement & governance hooks
    # -------------------------
    def register_enforcement_callback(self, fn):
        """Register a callback(fn(action_dict)) to handle enforcement actions."""
        self.enforcement_callbacks.append(fn)

    def enforce_symmetry(self, belief_id: str, action_policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Decide and emit enforcement actions to restore epistemic order.
        Example actions: 'request_verification', 'quarantine_belief', 'initiate_retest', 'escalate_to_human'.
        Returns action summary and calls registered callbacks.
        """
        if belief_id not in self.belief_registry:
            return {'status': 'not_found', 'belief_id': belief_id}

        belief = self.belief_registry[belief_id]
        status = belief.get('status', 'uncertain')

        # policy decision heuristics
        action = None
        reason = None
        if status == 'fiction':
            action = 'quarantine_belief'
            reason = 'symmetry_detected_as_fiction'
        elif status == 'uncertain':
            action = 'request_verification'
            reason = 'insufficient_evidence'
        elif status == 'fact':
            action = 'publish_confirmed'
            reason = 'verified_as_fact'
        else:
            action = 'request_review'
            reason = 'unknown_status'

        result = {'belief_id': belief_id, 'action': action, 'reason': reason, 'policy': action_policy or {}}

        # call callbacks
        for cb in self.enforcement_callbacks:
            try:
                cb(result)
            except Exception:
                # fail-safe: continue
                pass

        return result

    # -------------------------
    # Composite symmetry score & telemetry
    # -------------------------
    def compute_system_symmetry_score(self) -> Dict[str, Any]:
        """
        Aggregate a global symmetry score across all registered beliefs, weighted by recency and evidence strength.
        Returns score [0,1] and component breakdown for telemetry.
        """
        if not self.belief_registry:
            return {'score': 1.0, 'breakdown': {}, 'timestamp': time.time()}

        scores = []
        weights = []
        now = time.time()
        for bid, entry in self.belief_registry.items():
            s = entry.get('symmetry_score', 0.0) or 0.0
            # recency weight: fresher beliefs weigh more
            age = now - entry.get('last_checked', entry.get('timestamp', now))
            recency_w = math.exp(-age / (60.0 * 60.0))  # hour-scale decay
            scores.append(s * recency_w)
            weights.append(recency_w)

        weights = np.array(weights)
        if weights.sum() == 0:
            score = float(np.mean(scores)) if scores else 0.0
        else:
            score = float(np.sum(scores) / (weights.sum() + 1e-12))

        # ensure 0..1
        score = float(np.clip(score, 0.0, 1.0))
        breakdown = {
            'avg_symmetry': float(np.mean([entry.get('symmetry_score', 0.0) or 0.0 for entry in self.belief_registry.values()])),
            'facts_count': self.metrics.get('facts_count', 0),
            'fiction_count': self.metrics.get('fiction_count', 0),
            'uncertain_count': self.metrics.get('uncertain_count', 0)
        }
        # telemetry
        self.metrics['symmetry_score_history'].append({'score': score, 'timestamp': now})
        return {'score': score, 'breakdown': breakdown, 'timestamp': now}

    # -------------------------
    # Utilities & Introspection
    # -------------------------
    def get_belief(self, belief_id: str) -> Dict[str, Any]:
        return self.belief_registry.get(belief_id, {})

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
