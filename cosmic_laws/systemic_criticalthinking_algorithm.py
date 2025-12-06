"""
Ontology Declaration - Critical Thinking Algorithm

Existence Nature: Systemic Evaluative Module
Purpose: Implements Critical Thinking as an evaluative, reflective, and self-correcting
         process that validates, interrogates, and confirms reasoning outputs and knowledge.
Granted Capacities:
  - Analysis & Deconstruction of Claims
  - Assumption Extraction & Examination
  - Bias Detection & Mitigation (personal, source, systemic)
  - Evidence Assessment & Weighting (probabilistic/heuristic)
  - Logical Fallacy Identification (structured heuristics)
  - Synthesis of Alternative Explanations
  - Reflective Judgment & Belief Revision Hooks
  - Dispositional Regulation (calibrate open-mindedness, skepticism, humility)
  - Epistemic Vigilance & Provenance Tracking
Interactions:
  - Receives candidate inferences and traces from ReasoningAlgorithm
  - Consults IntelligenceAlgorithm for representation-level diagnostics
  - Reports flagged inconsistencies to SymmetryAlgorithm for alignment enforcement
  - Provides structured feedback to Orchestrator and Evolution for adaptation decisions
Notes:
  - Outputs are structured dicts with 'result', 'score', 'status', 'trace', and 'recommendation'.
  - This module favors explainability and auditable decisions; randomness is avoided in heuristics.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine
import math
import uuid

# ---------- utility functions ----------
def safe_cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    val = np.dot(a, b) / (na * nb)
    # clip to [-1,1], convert to similarity [0,1]
    val = float(np.clip(val, -1.0, 1.0))
    return (val + 1.0) / 2.0  # map -1..1 -> 0..1

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def kl_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """A simple abs-difference mean as a cheap proxy for surprise/KL-like divergence."""
    return float(np.mean(np.abs(a - b)))

# ---------- core module ----------
class CriticalThinkingAlgorithm:
    """
    Critical Thinking (Evaluation/Confirmation) - Version 2
    """

    def __init__(self,
                 credibility_threshold: float = 0.5,
                 fallacy_sensitivity: float = 0.6):
        # thresholds and calibration
        self.credibility_threshold = credibility_threshold
        self.fallacy_sensitivity = fallacy_sensitivity

        # provenance registry for evidence and sources
        self.provenance_registry: Dict[str, Dict[str, Any]] = {}

        # lightweight knowledge of common fallacy vectors/templates (symbolic heuristics)
        # each pattern is a small heuristic function (no learned randomness)
        self.fallacy_checks = {
            'strawman': self._heuristic_strawman,
            'ad_hominem': self._heuristic_ad_hominem,
            'false_equivalence': self._heuristic_false_equivalence,
            'appeal_to_authority': self._heuristic_appeal_to_authority,
            'circular_reasoning': self._heuristic_circular_reasoning,
            'hasty_generalization': self._heuristic_hasty_generalization
        }

        # disposition state (open-mindedness, skepticism, humility); dynamic and adjustable
        self.dispositions = {
            'open_mindedness': 0.6,
            'skepticism': 0.5,
            'intellectual_humility': 0.5
        }

    # --------------------------
    # Provenance & Evidence APIs
    # --------------------------
    def register_source(self, source_id: str, metadata: Dict[str, Any]) -> None:
        """Register source provenance info into registry."""
        self.provenance_registry[source_id] = metadata.copy()

    def source_reliability(self, source_id: str) -> float:
        """Return reliability score [0,1] if known, otherwise 0.5."""
        md = self.provenance_registry.get(source_id)
        if not md:
            return 0.5
        return float(md.get('reliability', 0.5))

    # --------------------------
    # Analysis & Assumption Extraction
    # --------------------------
    def analyze_assumptions(self, claim_vector: np.ndarray,
                            supporting_vectors: List[np.ndarray]) -> Dict[str, Any]:
        """
        Deconstruct claim into constituent support vectors and compute assumption fragility.
        Returns: {'assumptions': count, 'fragility': 0..1, 'trace': {...}}
        """
        if not supporting_vectors:
            return {'assumptions': 0, 'fragility': 1.0, 'trace': {'reason': 'no_supporting_evidence'}}

        sims = [safe_cosine_sim(claim_vector, sv) for sv in supporting_vectors]
        mean_sim = float(np.mean(sims))
        fragility = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
        trace = {'type': 'assumption_analysis', 'n_support': len(supporting_vectors), 'mean_sim': mean_sim}
        return {'assumptions': len(supporting_vectors), 'fragility': fragility, 'trace': trace}

    # --------------------------
    # Bias Detection & Mitigation
    # --------------------------
    def detect_bias(self, information: np.ndarray, known_bias_vectors: Optional[List[np.ndarray]] = None
                    ) -> Dict[str, Any]:
        """
        Detect similarity to known bias vectors, and return bias strength + suggested mitigation.
        """
        if not known_bias_vectors:
            return {'bias_present': False, 'strength': 0.0, 'details': None}

        sims = [safe_cosine_sim(information, kb) for kb in known_bias_vectors]
        max_sim = float(np.max(sims))
        bias_present = max_sim > 0.6
        details = {'most_sim': max_sim, 'index': int(np.argmax(sims))}
        mitigation = None
        if bias_present:
            # suggest mitigation: request diverse evidence, downweight confidence
            mitigation = {'action': 'request_diverse_evidence', 'downweight_factor': 0.5}
        return {'bias_present': bias_present, 'strength': max_sim, 'details': details, 'mitigation': mitigation}

    def bias_mitigation_apply(self, candidate_vec: np.ndarray, downweight_factor: float) -> np.ndarray:
        """Apply mitigation by reducing magnitude/confidence of biased candidate"""
        return candidate_vec * (1.0 - float(np.clip(downweight_factor, 0.0, 1.0)))

    # --------------------------
    # Evidence Assessment & Fact-Checking
    # --------------------------
    def evidence_weighting(self, evidence_vectors: List[np.ndarray], provenance_ids: List[str] = None
                           ) -> Dict[str, Any]:
        """
        Weights evidence by (a) source reliability, (b) independence (diverse vectors), and (c) recency if present.
        Returns aggregated support vector, support_score [0,1], and weight breakdown.
        """
        if not evidence_vectors:
            return {'support_vec': None, 'support_score': 0.0, 'weights': []}

        # base similarity matrix
        stack = np.stack(evidence_vectors)
        mean_vec = np.mean(stack, axis=0)
        sims = [safe_cosine_sim(mean_vec, v) for v in evidence_vectors]

        # provenance weighting
        weights = []
        for i, pid in enumerate(provenance_ids or [None] * len(evidence_vectors)):
            reli = self.source_reliability(pid) if pid else 0.5
            independence_penalty = 1.0 - np.mean([safe_cosine_sim(evidence_vectors[i], v) for j, v in enumerate(evidence_vectors) if j != i]) if len(evidence_vectors) > 1 else 1.0
            w = float(reli * 0.6 + independence_penalty * 0.4)
            weights.append(w)

        # normalized weighted support
        weights = np.array(weights)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / (weights.sum() + 1e-12)
        support_vec = np.sum(np.stack(evidence_vectors).T * weights, axis=1)
        support_score = float(np.clip(np.dot(np.mean(stack, axis=0), support_vec) / (np.linalg.norm(support_vec) * (np.linalg.norm(np.mean(stack, axis=0)) + 1e-9)), -1.0, 1.0))
        # map to 0..1
        support_score = (support_score + 1.0) / 2.0
        return {'support_vec': normalize(support_vec), 'support_score': support_score, 'weights': weights.tolist()}

    def fact_check(self, claim_vec: np.ndarray, evidence_vectors: List[np.ndarray], provenance_ids: List[str] = None
                   ) -> Dict[str, Any]:
        """
        Returns:
          - verified (bool)
          - support_score [0,1]
          - explanation trace
          - recommendation (e.g., seek more evidence, flag, accept)
        """
        ew = self.evidence_weighting(evidence_vectors, provenance_ids)
        support = ew['support_score']
        if support > 0.75:
            status = 'HIGH_CONFIDENCE'
            verified = True
            recommendation = 'accept'
        elif support > 0.45:
            status = 'MEDIUM_CONFIDENCE'
            verified = True
            recommendation = 'monitor'
        else:
            status = 'LOW_CONFIDENCE'
            verified = False
            recommendation = 'seek_more_evidence'

        # measure alignment between claim and support
        align = safe_cosine_sim(claim_vec, ew['support_vec']) if ew['support_vec'] is not None else 0.0

        trace = {'type': 'fact_check', 'support_score': support, 'alignment': align}
        return {'verified': verified, 'support_score': support, 'alignment': align, 'status': status, 'recommendation': recommendation, 'trace': trace}

    # --------------------------
    # Logical Fallacy Detection (structured heuristics)
    # --------------------------
    def _heuristic_strawman(self, argument_vec: np.ndarray, context_vec: Optional[np.ndarray] = None) -> float:
        """
        Strawman heuristic: detect if argument diverges sharply from average supporting evidence
        by measuring asymmetric distance to context/support.
        """
        if context_vec is None:
            return 0.0
        sim = safe_cosine_sim(argument_vec, context_vec)
        return float(np.clip(1.0 - sim, 0.0, 1.0))

    def _heuristic_ad_hominem(self, argument_text_meta: Dict[str, Any], *_) -> float:
        """
        Very simple heuristic: if metadata contains 'attacks_person' flag -> high score.
        This requires upstream text-to-meta processing to tag attack features.
        """
        return float(1.0) if argument_text_meta.get('attacks_person', False) else 0.0

    def _heuristic_false_equivalence(self, argument_pairs: Tuple[np.ndarray, np.ndarray]) -> float:
        a, b = argument_pairs
        # if both are low-similarity but presented as equivalent -> medium-high score
        sim = safe_cosine_sim(a, b)
        return float(np.clip(1.0 - sim, 0.0, 1.0))

    def _heuristic_appeal_to_authority(self, metadata: Dict[str, Any]) -> float:
        """
        If claim relies heavily on a single authority and lacks independent evidence, raise flag.
        """
        return float(metadata.get('single_authority_reliance', 0.0))

    def _heuristic_circular_reasoning(self, argument_vec: np.ndarray, supporting_vectors: List[np.ndarray]) -> float:
        """
        Detect circularity: high similarity between claim and supporting evidence but low independent support.
        """
        if not supporting_vectors:
            return 0.0
        sims = [safe_cosine_sim(argument_vec, v) for v in supporting_vectors]
        mean_sim = float(np.mean(sims))
        independence = 1.0 - np.mean([safe_cosine_sim(supporting_vectors[i], supporting_vectors[j])
                                     for i in range(len(supporting_vectors)) for j in range(i + 1, len(supporting_vectors))]
                                    ) if len(supporting_vectors) > 1 else 0.0
        # circular if claim close to evidence but evidence lacks independence
        score = float(np.clip(mean_sim * (1.0 - independence), 0.0, 1.0))
        return score

    def _heuristic_hasty_generalization(self, claim_vec: np.ndarray, supporting_vectors: List[np.ndarray]) -> float:
        """
        Flag if sample size is small and variance is high.
        """
        if not supporting_vectors:
            return 1.0
        stack = np.stack(supporting_vectors)
        var = float(np.var(stack, axis=0).mean())
        n = len(supporting_vectors)
        score = float(np.clip((1.0 - math.exp(-n / 5.0)) * var, 0.0, 1.0))
        return score

    def detect_logical_fallacies(self, argument_vec: np.ndarray, context_support: Optional[List[np.ndarray]] = None,
                                argument_meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Run structured heuristics and return fallacy scores in [0,1] for each check.
        """
        scores = {}
        context_mean = np.mean(np.stack(context_support), axis=0) if context_support else None
        for name, fn in self.fallacy_checks.items():
            try:
                if name == 'ad_hominem':
                    scores[name] = fn(argument_meta or {})
                elif name == 'false_equivalence':
                    if context_support and len(context_support) >= 2:
                        scores[name] = fn((context_support[0], context_support[1]))
                    else:
                        scores[name] = 0.0
                elif name == 'strawman':
                    scores[name] = fn(argument_vec, context_mean)
                elif name == 'circular_reasoning':
                    scores[name] = fn(argument_vec, context_support or [])
                elif name == 'hasty_generalization':
                    scores[name] = fn(argument_vec, context_support or [])
                else:
                    scores[name] = 0.0
            except Exception:
                scores[name] = 0.0
        return {k: float(v) for k, v in scores.items()}

    # --------------------------
    # End-to-end Credibility Evaluation
    # --------------------------
    def evaluate_credibility(self,
                             candidate_vec: np.ndarray,
                             reasoning_trace: Dict[str, Any],
                             evidence_vectors: List[np.ndarray],
                             provenance_ids: Optional[List[str]] = None,
                             source_id: Optional[str] = None,
                             argument_meta: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
        """
        Full credibility pipeline:
         1) Evidence weighting and fact-checking
         2) Assumption fragility
         3) Logical fallacy detection
         4) Bias detection
         5) Composite credibility score & recommendation
        """
        # 1) evidence support
        fact_check_res = self.fact_check(candidate_vec, evidence_vectors, provenance_ids)

        # 2) assumptions
        ass_res = self.analyze_assumptions(candidate_vec, evidence_vectors)

        # 3) fallacies
        fall_res = self.detect_logical_fallacies(candidate_vec, context_support=evidence_vectors, argument_meta=argument_meta)
        fall_presence = float(max(fall_res.values()) if fall_res else 0.0)

        # 4) bias
        # gather known bias vectors from provenance registry if flagged (simplified)
        known_bias_vecs = []
        for pid in provenance_ids or []:
            md = self.provenance_registry.get(pid)
            if md and 'bias_vector' in md:
                known_bias_vecs.append(md['bias_vector'])
        bias_res = self.detect_bias(candidate_vec, known_bias_vecs)

        # 5) composite credibility score
        source_rel = self.source_reliability(source_id) if source_id else 0.5
        # weights tuned to prefer direct evidence support and fallacy absence
        credibility_score = float(
            0.45 * fact_check_res['support_score'] +
            0.25 * source_rel +
            0.20 * (1.0 - fall_presence) +
            0.10 * (1.0 - ass_res['fragility'])
        )
        credibility_score = float(np.clip(credibility_score, 0.0, 1.0))

        # decision thresholds
        if credibility_score >= max(self.credibility_threshold, 0.75):
            status = 'HIGH_CONFIDENCE'
            recommendation = 'accept'
        elif credibility_score >= self.credibility_threshold:
            status = 'MEDIUM_CONFIDENCE'
            recommendation = 'monitor'
        else:
            status = 'LOW_CONFIDENCE'
            recommendation = 'flag_for_review'

        # assemble result
        trace = {
            'fact_check': fact_check_res,
            'assumptions': ass_res,
            'fallacies': fall_res,
            'bias': bias_res,
            'source_reliability': source_rel
        }

        # suggestion to downstream processes (e.g., downweight vector when flagged)
        filtered_vec = candidate_vec
        if status == 'LOW_CONFIDENCE':
            filtered_vec = self.bias_mitigation_apply(candidate_vec, downweight_factor=0.5)

        result = {
            'output': filtered_vec,
            'credibility_score': credibility_score,
            'status': status,
            'recommendation': recommendation,
            'trace': trace,
            'id': str(uuid.uuid4())
        }
        return result

    # --------------------------
    # Reflective Judgment & Dispositional Regulation
    # --------------------------
    def reflect_and_update_dispositions(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        """
        Based on feedback (outcomes of prior decisions), adjust dispositions.
        Feedback example: {'outcome_accuracy':0.8, 'overconfidence_event':True}
        """
        outcome = float(feedback.get('outcome_accuracy', 0.5))
        overconf = bool(feedback.get('overconfidence_event', False))

        # simple rules: better outcomes -> slightly more open; overconfidence -> raise humility & skepticism
        if outcome > 0.7:
            self.dispositions['open_mindedness'] = float(np.clip(self.dispositions['open_mindedness'] + 0.02, 0.0, 1.0))
        if overconf:
            self.dispositions['intellectual_humility'] = float(np.clip(self.dispositions['intellectual_humility'] + 0.05, 0.0, 1.0))
            self.dispositions['skepticism'] = float(np.clip(self.dispositions['skepticism'] + 0.03, 0.0, 1.0))

        return self.dispositions.copy()

    # --------------------------
    # Utilities
    # --------------------------
    def explain_decision(self, cred_result: Dict[str, Any]) -> str:
        """Return human-friendly explanation string synthesized from trace."""
        t = cred_result.get('trace', {})
        parts = []
        fc = t.get('fact_check', {})
        parts.append(f"support_score={fc.get('support_score'):.3f}, alignment={fc.get('alignment'):.3f}")
        parts.append(f"assumption_fragility={t.get('assumptions', {}).get('fragility') if t.get('assumptions') else 'N/A'}")
        fallies = t.get('fallacies', {})
        major_fall = max(fallies.items(), key=lambda x: x[1])[0] if fallies else 'none'
        parts.append(f"major_fallacy={major_fall}")
        parts.append(f"source_rel={t.get('source_reliability'):.3f}")
        return "; ".join(map(str, parts))
