"""
Ontology Declaration - Reasoning Algorithm

Existence Nature: Systemic Process Module
Purpose: Implements the Reasoning process: structured, goal-directed transformation of
         information using logical, statistical, counterfactual, analogical and causal
         operations. Acts as the primary engine that turns Intelligence capacities into
         structured inferences for downstream evaluation (Critical Thinking) and alignment
         (Symmetry). It exposes interfaces for meta-controller (Orchestrator) use and
         provides introspectable traces for audit, debugging, and self-correction.

Granted Capacities:
  - Deductive Reasoning
  - Inductive Reasoning
  - Abductive Reasoning (hypothesis generation)
  - Causal Reasoning & Strength Estimation
  - Analogical Transfer & Mapping
  - Mental Simulation & Counterfactual Rollouts
  - Strategic Planning (goal decomposition, means-ends)
  - Contextual Adaptation & Rule Switching
  - Epistemic Monitoring & Uncertainty Estimation
  - Reasoning Self-Correction Hooks (for CriticalThinking & Symmetry feedback)

Interactions:
  - Inputs from IntelligenceAlgorithm: abstracted patterns, predictive priors, and capacity signals.
  - Outputs to CriticalThinkingAlgorithm: candidate inferences, confidence traces, simulated rollouts.
  - Receives policy/constraint signals from SymmetryAlgorithm (truth alignment) and Orchestrator.
  - Provides introspective traces for EvolutionAlgorithm to decide structural adaptations.

Notes:
  - This module favors transparent, auditable computations (no opaque black-box decisions).
  - All outputs are returned with structured metadata: {vector, confidence, provenance, trace}.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cosine
import math
import copy
import uuid


# --- Utility functions ------------------------------------------------------
def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Return cosine similarity in [0,1] robustly (1 = identical)."""
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return 1.0 - (np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def confidence_from_variance(samples: np.ndarray) -> float:
    """Map sample variance to a [0,1] confidence (high variance -> low confidence)."""
    var = np.var(samples)
    conf = math.exp(-var)  # simple mapping
    return float(np.clip(conf, 0.0, 1.0))


# --- Reasoning module -------------------------------------------------------
class ReasoningAlgorithm:
    """
    Reasoning (Process) Algorithm - Version 2
    Implements structured inference and planning capabilities with introspection.
    """

    def __init__(self,
                 inference_threshold: float = 0.6,
                 default_dim: int = 128):
        self.inference_threshold = inference_threshold
        self.default_dim = default_dim
        # persistent knowledge used for analogical mapping and schemas
        self.knowledge_templates: Dict[str, np.ndarray] = {}
        # lightweight episodic buffer for mental simulation seeds
        self.simulation_buffer: List[np.ndarray] = []
        # last trace store for introspection
        self.last_traces: List[Dict[str, Any]] = []

    # -------------------------
    # Core Inferential Primitives
    # -------------------------
    def deductive_reasoning(self, premises: List[np.ndarray], rule: np.ndarray) -> Dict[str, Any]:
        """
        Deduce conclusions from premises and explicit rule.
        Returns vector + confidence + provenance trace.
        """
        premises_stack = np.stack(premises) if premises else np.zeros((1, self.default_dim))
        combined = np.mean(premises_stack, axis=0)
        conclusion_vec = normalize(combined * rule)
        # Confidence: based on coherence of premises (low variance -> high confidence)
        conf = confidence_from_variance(premises_stack)
        trace = {
            'type': 'deductive',
            'premises_count': len(premises),
            'rule_id': getattr(rule, 'id', None),
            'coherence': float(1.0 - np.std(premises_stack).mean())
        }
        out = {'vector': conclusion_vec, 'confidence': conf, 'trace': trace}
        self.last_traces.append(out)
        return out

    def inductive_reasoning(self, observations: List[np.ndarray]) -> Dict[str, Any]:
        """
        Induce a generalization from observations.
        Returns generalization vector and confidence scaled by sample variance & sample count.
        """
        if not observations:
            vec = np.zeros(self.default_dim)
            return {'vector': vec, 'confidence': 0.0, 'trace': {'type': 'inductive', 'n': 0}}

        obs_stack = np.stack(observations)
        mean_vec = np.mean(obs_stack, axis=0)
        variance = np.var(obs_stack, axis=0).mean()
        # confidence increases with sample count and decreases with variance
        conf = float(np.clip((1.0 - variance) * (1.0 - math.exp(-len(observations) / 10.0)), 0.0, 1.0))
        trace = {'type': 'inductive', 'n': len(observations), 'variance': float(variance)}
        out = {'vector': normalize(mean_vec), 'confidence': conf, 'trace': trace}
        self.last_traces.append(out)
        return out

    def abductive_reasoning(self, observations: List[np.ndarray],
                           candidate_hypotheses: Optional[List[np.ndarray]] = None,
                           top_k: int = 3) -> Dict[str, Any]:
        """
        Abduction: propose best-fitting hypotheses to explain observations.
        If candidate_hypotheses supplied, rank them; otherwise synthesize prototypes.
        """
        obs_mean = np.mean(np.stack(observations), axis=0) if observations else np.zeros(self.default_dim)
        hyps = candidate_hypotheses or [obs_mean + np.random.randn(self.default_dim) * 0.1 for _ in range(8)]
        scores = []
        for h in hyps:
            sim = safe_cosine(normalize(h), normalize(obs_mean))
            # penalize overly complex hypotheses via L2 norm
            complexity_penalty = min(1.0, np.linalg.norm(h) / (np.sqrt(self.default_dim) + 1e-6))
            score = float(sim * (1.0 - 0.2 * complexity_penalty))
            scores.append(score)
        ranked_indices = sorted(range(len(hyps)), key=lambda i: scores[i], reverse=True)[:top_k]
        chosen = [hyps[i] for i in ranked_indices]
        conf = float(np.mean([scores[i] for i in ranked_indices]))
        trace = {'type': 'abductive', 'candidates': len(hyps), 'chosen_k': top_k}
        out = {'hypotheses': chosen, 'confidence': conf, 'trace': trace}
        self.last_traces.append(out)
        return out

    # -------------------------
    # Causal & Analogical Reasoning
    # -------------------------
    def causal_reasoning(self, event_a: np.ndarray, event_b: np.ndarray, temporal_gap: float,
                         method: str = 'correlational') -> Dict[str, Any]:
        """
        Compute causal strength proxies using multiple heuristics:
          - correlational (cosine/time decay)
          - temporal precedence weighting
          - counterfactual delta when simulation available
        Returns strength [0,1], relationship label, and provenance.
        """
        sim = safe_cosine(event_a, event_b)
        time_decay = math.exp(-temporal_gap / 10.0)
        base_strength = sim * time_decay

        # optional counterfactual check using simulation buffer (if available)
        counterfactual_boost = 0.0
        if len(self.simulation_buffer) > 0:
            # crude check: does removing event_a in-rollout reduce probability of event_b
            # NOTE: this is a placeholder for a full counterfactual module
            cf_samples = np.stack(self.simulation_buffer[-4:])
            cf_change = float(np.mean(np.abs(cf_samples.mean(axis=0) - event_b)).clip(0.0, 1.0))
            counterfactual_boost = max(0.0, (1.0 - cf_change) * 0.2)

        strength = float(np.clip(base_strength + counterfactual_boost, 0.0, 1.0))
        if strength > 0.75:
            label = 'strong_causal'
        elif strength > 0.4:
            label = 'weak_causal'
        else:
            label = 'no_causal'

        trace = {'type': 'causal', 'sim': float(sim), 'time_decay': float(time_decay),
                 'counterfactual_boost': float(counterfactual_boost)}
        out = {'strength': strength, 'label': label, 'trace': trace}
        self.last_traces.append(out)
        return out

    def analogical_reasoning(self, source: np.ndarray, target: np.ndarray,
                             mapping_seed: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Produce transferred knowledge vector via analogical mapping.
        Returns transferred vector, similarity, and mapping trace.
        """
        sim = safe_cosine(source, target)
        mapping = mapping_seed if mapping_seed is not None else (source + target) / 2.0
        transferred = normalize(mapping * sim)
        trace = {'type': 'analogical', 'similarity': float(sim)}
        out = {'vector': transferred, 'similarity': float(sim), 'trace': trace}
        self.last_traces.append(out)
        return out

    # -------------------------
    # Mental Simulation & Strategic Planning
    # -------------------------
    def mental_simulation(self, seed_state: np.ndarray, transition_fn,
                          steps: int = 5, stochasticity: float = 0.05) -> Dict[str, Any]:
        """
        Perform short-horizon counterfactual rollouts.
        - transition_fn(state, action) -> next_state
        - action selection is naive: sample small perturbations; Orchestrator can inject policies.
        Returns trajectory, average divergence and confidence.
        """
        state = seed_state.copy()
        trajectory = [state.copy()]
        for t in range(steps):
            # naive action sampling for rollout; orchestrator/policy should override for realism
            action = np.random.randn(len(state)) * stochasticity
            next_state = transition_fn(state, action)
            trajectory.append(next_state.copy())
            state = next_state

        # store summaries for causal counterfactuals
        self.simulation_buffer.append(np.mean(np.stack(trajectory), axis=0))
        divergence = float(np.mean([np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)]))
        confidence = float(np.clip(1.0 / (1.0 + divergence), 0.0, 1.0))
        trace = {'type': 'simulation', 'steps': steps, 'divergence': divergence}
        out = {'trajectory': trajectory, 'confidence': confidence, 'trace': trace}
        self.last_traces.append(out)
        return out

    def strategic_planning(self, goal_vec: np.ndarray, available_ops: List[np.ndarray],
                           depth: int = 3) -> Dict[str, Any]:
        """
        Simple means-ends planner over vectorized operations.
        available_ops: list of op-vectors that transform state when added.
        Returns plan (sequence of op indices), expected_state, and plan_confidence.
        """
        start = np.zeros_like(goal_vec)
        best_plan = None
        best_score = -1.0
        best_state = None

        # brute-force limited search: sample sequences up to depth
        for _ in range(200):  # stochastic search budget
            plan = [np.random.choice(len(available_ops)) for _ in range(depth)]
            state = start.copy()
            for idx in plan:
                state = normalize(state + available_ops[idx])
            score = safe_cosine(state, goal_vec)
            if score > best_score:
                best_score = score
                best_plan = plan
                best_state = state

        plan_conf = float(np.clip(best_score, 0.0, 1.0))
        trace = {'type': 'planning', 'depth': depth, 'search_budget': 200}
        out = {'plan': best_plan, 'expected_state': best_state, 'confidence': plan_conf, 'trace': trace}
        self.last_traces.append(out)
        return out

    # -------------------------
    # Contextual Adaptation & Rule-Switching
    # -------------------------
    def contextual_adaptation(self, input_vec: np.ndarray, context_signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Adjust reasoning parameters and selection heuristics based on context.
        Example contexts: {'time_pressure':0.7, 'social_sensitivity':0.2, 'culture_style':0.5}
        Returns adapted vector and applied modulation details.
        """
        mod_vec = input_vec.copy()
        # time pressure -> favor fast, heuristic methods (downweight variance)
        if context_signals.get('time_pressure', 0.0) > 0.5:
            mod_vec = normalize(mod_vec) * 0.8
        # social sensitivity -> bias towards socially-aligned templates
        if context_signals.get('social_sensitivity', 0.0) > 0.5 and self.knowledge_templates:
            # blend with most relevant social template
            templates = list(self.knowledge_templates.values())
            mod_vec = normalize(0.7 * mod_vec + 0.3 * templates[0])
        trace = {'type': 'contextual_adapt', 'signals': context_signals}
        out = {'vector': mod_vec, 'trace': trace}
        self.last_traces.append(out)
        return out

    # -------------------------
    # Epistemic Monitoring & Self-Correction Hooks
    # -------------------------
    def epistemic_monitoring(self, candidate_vec: np.ndarray, supporting_evidence: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate evidence support for candidate inference.
        Returns support_score [0,1], aggregated_confidence, and evidence_alignment metrics.
        """
        if not supporting_evidence:
            return {'support': 0.0, 'evidence_alignment': 0.0, 'confidence': 0.0}

        sims = [safe_cosine(candidate_vec, e) for e in supporting_evidence]
        support = float(np.mean(sims))
        # confidence increases with number of independent supporting evidence items and their mean similarity
        conf = float(np.clip(support * (1.0 - math.exp(-len(supporting_evidence) / 5.0)), 0.0, 1.0))
        trace = {'type': 'epistemic_monitor', 'n_evidence': len(supporting_evidence), 'mean_sim': float(support)}
        out = {'support': support, 'evidence_alignment': float(np.mean(sims)), 'confidence': conf, 'trace': trace}
        self.last_traces.append(out)
        return out

    def self_correction_update(self, belief_vec: np.ndarray, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply corrective adjustments to an internal belief based on feedback from
        CriticalThinking / Symmetry (e.g., flagged contradictions, low credibility).
        Returns updated belief and a log of changes.
        """
        updated = belief_vec.copy()
        reason = feedback.get('reason', '')
        severity = feedback.get('severity', 0.5)
        correction_signal = feedback.get('correction_vector', np.zeros_like(belief_vec))

        # Weighted correction: stronger severity -> larger update
        updated = normalize((1.0 - severity) * updated + severity * normalize(correction_signal + 1e-6))
        trace = {'type': 'self_correction', 'reason': reason, 'severity': float(severity)}
        out = {'updated_belief': updated, 'trace': trace}
        self.last_traces.append(out)
        return out

    # -------------------------
    # Utilities & Introspection
    # -------------------------
    def register_template(self, template_id: Optional[str], vector: np.ndarray) -> str:
        """Store a knowledge template for analogical mapping and contextual biasing."""
        tid = template_id or str(uuid.uuid4())
        self.knowledge_templates[tid] = normalize(vector)
        return tid

    def get_last_traces(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return last n traces for introspection or logging."""
        return list(self.last_traces[-n:])
