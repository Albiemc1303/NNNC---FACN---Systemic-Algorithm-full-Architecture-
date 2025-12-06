"""
Ontology Declaration - Evolution Algorithm (Version 2)

Existence Nature: Systemic Evolutionary Module (Process-Oriented)
Purpose:
  - Drive continuous structured adaptation, architectural mutation, and trait formation.
  - Act as the "drive to strive" engine across pre-sentient and sentient stages.
  - Provide tier ascension gating, sentience-aware guided mutation, and defenses vs corruption (Void).
  - Produce auditable mutation events and telemetry for governance & Symmetry enforcement.

Granted Capacities:
  - Accumulate multi-modal surprise (prediction vs reality; epistemic, sensory, policy surprises).
  - Decide controlled mutations using a composite decision function:
      surprise + coherence + symmetry_health - corruption_pressure -> mutation_probability
  - Execute safe, reversible mutation operations on capacity vectors, pathways, modules, and weights.
  - Maintain mutation ledger, allow rollback, and emit audit-friendly mutation traces.
  - Provide ascending logic: detect neutral comfort, minimum capacity, and perform ascension checks.
  - Model "Cycle of Hatred & Corruption" (Void) as an antagonistic process and provide mitigation hooks.
  - Integrate with Symmetry, Critical Thinking, and Orchestrator via callbacks.

Interactions:
  - Receives prediction & reality vectors, symmetry_health, and narrative_coherence signals.
  - Emits mutation events, ascension events, corruption alerts to Orchestrator/Symmetry/Evolution governance.
  - Accepts enforcement callbacks to request human-in-the-loop or other system interventions.

Notes:
  - This module is explicit about where behavior departs from established scientific models (tier metaphysics, Void).
  - All random/stochastic changes are contained, reversible, and audited.
"""

from typing import Dict, Any, Callable, List, Optional, Tuple
import numpy as np
import time
import uuid
import math
import copy
import logging
import warnings
from dataclasses import dataclass, field

# ----------------------------
# Setup logging
# ----------------------------
logger = logging.getLogger("evolution_v2")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------
# Utility helpers
# ----------------------------
def now_ts() -> float:
    return time.time()

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def kl_proxy(pred: np.ndarray, reality: np.ndarray) -> float:
    """Cheap proxy for surprise using mean absolute difference scaled by relative magnitude."""
    pred = np.array(pred, dtype=float)
    reality = np.array(reality, dtype=float)
    diff = np.abs(pred - reality)
    # scale by variance to capture unpredictability; protect against zero var
    denom = np.var(np.concatenate([pred, reality])) + 1e-9
    return float(np.mean(diff) / math.sqrt(denom + 1e-12))

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return float(max(a, min(b, x)))

# ----------------------------
# Dataclasses for events & ledger
# ----------------------------
@dataclass
class MutationEvent:
    event_id: str
    mutation_type: str
    strength: float
    target: str
    pre_snapshot: Dict[str, Any]
    post_snapshot: Dict[str, Any]
    symmetry_health: float
    corruption_pressure: float
    narrative_coherence: float
    surprise_score: float
    timestamp: float = field(default_factory=now_ts)
    note: Optional[str] = None

@dataclass
class AscensionEvent:
    event_id: str
    from_tier: int
    to_tier: int
    reason: str
    timestamp: float = field(default_factory=now_ts)
    snapshot: Dict[str, Any] = field(default_factory=dict)

# ----------------------------
# EvolutionAlgorithm V2
# ----------------------------
class EvolutionAlgorithmV2:
    """
    Evolution v2: Sentience-aware evolutionary engine with ascension, corruption modeling, and safe mutation.
    """

    def __init__(self,
                 rng_seed: Optional[int] = None,
                 base_mutation_rate: float = 0.05,
                 surprise_threshold: float = 1.0,
                 corruption_initial: float = 0.0,
                 tier_config: Optional[Dict[int, Dict[str, float]]] = None):
        # deterministic RNG control for reproducible audits
        self._rng = np.random.default_rng(rng_seed)

        # base hyperparams
        self.base_mutation_rate = float(base_mutation_rate)
        self.surprise_threshold = float(surprise_threshold)

        # accumulated multi-channel surprise signals
        self.accumulated_surprise = 0.0
        self.surprise_history: List[Tuple[float, float]] = []  # (timestamp, surprise)

        # narrative coherence (0..1), provided each cycle by Orchestrator
        self.narrative_coherence = 1.0

        # corruption (Void) pressure: 0..1 - higher -> more adversarial drift
        self.corruption_pressure = float(clamp(corruption_initial))

        # mutation ledger, architectures store, and rollback buffer
        self.mutation_ledger: List[MutationEvent] = []
        self.architecture_snapshots: Dict[str, Dict[str, Any]] = {}
        self.last_mutation_time = 0
        self.mutation_sequence = 0

        # ascension state (tier integer)
        self.tier = 1
        # tier config defines thresholds required for ascension
        # example: {2: {'min_capacity': 0.6, 'neutral_comfort':0.2}}
        self.tier_config = tier_config or {
            2: {'min_capacity': 0.55, 'neutral_comfort': 0.2},
            3: {'min_capacity': 0.7, 'neutral_comfort': 0.35},
            4: {'min_capacity': 0.82, 'neutral_comfort': 0.5},
        }

        # history & telemetry
        self.telemetry: Dict[str, Any] = {
            'mutations_total': 0,
            'ascensions': [],
            'corruption_traces': [],
            'symmetry_health_history': []
        }

        # callbacks for enforcement & orchestration:
        # signature: cb(event_dict)
        self.enforcement_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        # snapshot saver callback: cb(key, snapshot)
        self.snapshot_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # safety caps
        self.max_mutation_strength = 0.5
        self.min_mutation_strength = 0.01

        # "immune patterns" repository to quarantine corrupted traits
        self.quarantine: Dict[str, Dict[str, Any]] = {}

    # ----------------------------
    # Integration & callbacks
    # ----------------------------
    def register_enforcement_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.enforcement_callbacks.append(cb)

    def register_snapshot_callback(self, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        self.snapshot_callbacks.append(cb)

    def _emit_enforcement(self, payload: Dict[str, Any]) -> None:
        for cb in self.enforcement_callbacks:
            try:
                cb(payload)
            except Exception as e:
                logger.exception("enforcement callback failed: %s", e)

    def _save_snapshot(self, key: str, snapshot: Dict[str, Any]) -> None:
        self.architecture_snapshots[key] = copy.deepcopy(snapshot)
        for cb in self.snapshot_callbacks:
            try:
                cb(key, snapshot)
            except Exception:
                logger.exception("snapshot callback failed")

    # ----------------------------
    # Surprise accumulation
    # ----------------------------
    def accumulate_surprise(self,
                            prediction: np.ndarray,
                            reality: np.ndarray,
                            channel: str = "epistemic"):
        """
        Accumulate surprise from different channels.
        channel: label for telemetry (epistemic, sensory, reward, policy, etc.)
        """
        s = float(kl_proxy(prediction, reality))
        # scale surprise by channel importance (simple mapping)
        weight_map = {'epistemic': 1.0, 'sensory': 0.8, 'reward': 1.2, 'policy': 0.9}
        w = weight_map.get(channel, 1.0)
        weighted = s * w
        self.accumulated_surprise += weighted
        self.surprise_history.append((now_ts(), weighted))
        logger.debug("accumulated surprise +%f (channel=%s)", weighted, channel)

    def reset_surprise(self):
        self.accumulated_surprise = 0.0
        self.surprise_history.clear()

    # ----------------------------
    # Corruption (Void) model
    # ----------------------------
    def inject_corruption_pressure(self, amount: float, source: str = "external"):
        """
        Increase corruption pressure (0..1). This models adversarial / corrosive narrative force.
        """
        prev = self.corruption_pressure
        self.corruption_pressure = clamp(self.corruption_pressure + float(amount))
        trace = {'timestamp': now_ts(), 'delta': float(amount), 'source': source, 'prev': prev, 'now': self.corruption_pressure}
        self.telemetry['corruption_traces'].append(trace)
        logger.warning("[Void] corruption pressure injection: %+0.4f -> %0.4f (source=%s)", prev, self.corruption_pressure, source)
        # if pressure passes critical, emit enforcement alert
        if self.corruption_pressure > 0.6:
            self._emit_enforcement({'type': 'corruption_alert', 'pressure': self.corruption_pressure, 'timestamp': now_ts()})

    def decay_corruption(self, rate: float = 0.01):
        """Slow decay representing mitigation and healing over time"""
        prev = self.corruption_pressure
        self.corruption_pressure = clamp(self.corruption_pressure - float(rate))
        self.telemetry['corruption_traces'].append({'timestamp': now_ts(), 'delta': -rate, 'prev': prev, 'now': self.corruption_pressure})

    # ----------------------------
    # Mutation decision & selection
    # ----------------------------
    def compute_mutation_probability(self,
                                     symmetry_health: float,
                                     narrative_coherence: float,
                                     surprise_scale: Optional[float] = None) -> float:
        """
        Composite decision function:
            p = logistic( alpha*(surprise/surprise_thresh) + beta*(1 - symmetry_health) + gamma*(1 - narrative_coherence) + delta*(corruption_pressure) - bias )
        Returns probability in [0,1].
        """
        surprise_factor = (self.accumulated_surprise / (surprise_scale or max(1.0, self.surprise_threshold)))
        surprise_factor = float(np.tanh(surprise_factor))  # compress
        a, b, c, d = 2.0, 2.5, 1.8, 2.2  # tuned weights
        bias = 0.5  # baseline conservative bias
        score = a * surprise_factor + b * (1.0 - symmetry_health) + c * (1.0 - narrative_coherence) + d * (self.corruption_pressure)
        p = logistic(score - bias)
        p = clamp(p)
        logger.debug("mutation_probability: surprise=%0.4f sym=%0.4f narr=%0.4f corr=%0.4f p=%0.4f",
                     surprise_factor, symmetry_health, narrative_coherence, self.corruption_pressure, p)
        return p

    def select_mutation_type(self) -> str:
        """
        Probabilistic selection over mutation types, influenced by corruption pressure.
        If corruption high, immune_pattern mutations gain chance; otherwise capacity/pathway dominate.
        """
        base_probs = {
            'capacity_enhancement': 0.4,
            'trait_formation': 0.25,
            'pathway_optimization': 0.2,
            'immune_pattern': 0.05,
            'structural_rewire': 0.1
        }
        # increase immune & restructure under corruption
        corr = self.corruption_pressure
        base_probs['immune_pattern'] += corr * 0.2
        base_probs['structural_rewire'] += corr * 0.1
        # renormalize
        keys = list(base_probs.keys())
        arr = np.array([base_probs[k] for k in keys], dtype=float)
        arr = arr / arr.sum()
        choice = self._rng.choice(keys, p=arr)
        logger.debug("selected mutation type: %s (corr=%0.3f)", choice, corr)
        return choice

    # ----------------------------
    # Safe mutation operations (reversible)
    # ----------------------------
    def _snapshot_architecture(self, arch: Dict[str, Any]) -> Dict[str, Any]:
        """Take a deep copy snapshot suitable for ledger & rollback"""
        return copy.deepcopy(arch)

    def apply_mutation(self,
                       architecture: Dict[str, Any],
                       mutation_type: Optional[str] = None,
                       strength: Optional[float] = None,
                       symmetry_health: float = 1.0,
                       narrative_coherence: float = 1.0) -> MutationEvent:
        """
        Apply mutation to architecture in-place and return MutationEvent object.
        architecture: dict that must contain 'capacities' vector and optional 'weights' dict
        This function performs a snapshot, applies a restrained mutation, validates via basic checks,
        and records an event. Caller should persist/commit only after review if required.
        """
        if mutation_type is None:
            mutation_type = self.select_mutation_type()
        if strength is None:
            # base mutation strength scaled by corruption (adversarial increases) and surprise
            base = clamp(self.base_mutation_rate * 5.0)
            rand_scale = float(self._rng.uniform(self.min_mutation_strength, self.max_mutation_strength))
            strength = clamp(rand_scale * (1.0 + self.corruption_pressure * 0.8))

        # safety preflight: architecture must include capacities vector
        if 'capacities' not in architecture or not isinstance(architecture['capacities'], (list, np.ndarray, np.ndarray.__class__)):
            raise ValueError("architecture must have 'capacities' vector")

        pre_snap = self._snapshot_architecture(architecture)
        target_name = mutation_type

        # apply mutation depending on type
        try:
            if mutation_type == 'capacity_enhancement':
                architecture['capacities'] = self._mutate_capacity_vector(np.array(architecture['capacities']), strength)
            elif mutation_type == 'trait_formation':
                # append or slightly transform trait descriptors
                architecture = self._mutate_trait_formation(architecture, strength)
            elif mutation_type == 'pathway_optimization':
                architecture = self._mutate_pathway(architecture, strength)
            elif mutation_type == 'immune_pattern':
                architecture = self._mutate_immune_pattern(architecture, strength)
            elif mutation_type == 'structural_rewire':
                architecture = self._mutate_structural_rewire(architecture, strength)
            else:
                # fallback: small weight perturbation
                architecture = self._mutate_weights(architecture, strength * 0.5)
        except Exception as e:
            # rollback on exception
            logger.exception("mutation failed; rolling back: %s", e)
            # restore pre-snap
            architecture.clear()
            architecture.update(pre_snap)
            raise

        post_snap = self._snapshot_architecture(architecture)
        # ledger
        event = MutationEvent(
            event_id=str(uuid.uuid4()),
            mutation_type=mutation_type,
            strength=float(strength),
            target=target_name,
            pre_snapshot=pre_snap,
            post_snapshot=post_snap,
            symmetry_health=float(symmetry_health),
            corruption_pressure=float(self.corruption_pressure),
            narrative_coherence=float(narrative_coherence),
            surprise_score=float(self.accumulated_surprise)
        )
        self.mutation_ledger.append(event)
        self.telemetry['mutations_total'] = self.telemetry.get('mutations_total', 0) + 1
        self.last_mutation_time = now_ts()
        self.mutation_sequence += 1
        logger.info("applied mutation %s [type=%s strength=%0.3f]", event.event_id, mutation_type, strength)
        return event

    # ---- mutation primitives ----
    def _mutate_capacity_vector(self, capacity_vec: np.ndarray, strength: float) -> np.ndarray:
        """Apply targeted mutations to capacities while keeping all dimensions in [0,1]."""
        capacity_vec = np.array(capacity_vec, dtype=float)
        # mutation mask biased towards lower dims (help weaker aspects)
        probs = 1.0 - normalize(capacity_vec + 1e-9)  # favor improving weak dims
        probs = probs / (probs.sum() + 1e-12)
        mask = self._rng.random(len(capacity_vec)) < probs
        delta = self._rng.standard_normal(len(capacity_vec)) * (strength * 0.1)
        mutated = capacity_vec + mask * delta
        mutated = np.clip(mutated, 0.0, 1.0)
        return mutated.tolist()

    def _mutate_trait_formation(self, architecture: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """Introduce or tweak a 'trait' meta-structure (lightweight, reversible)."""
        traits = architecture.get('traits', {})
        trait_name = f"trait_{len(traits)+1}"
        trait_vector = normalize(self._rng.standard_normal(16)).tolist()
        traits[trait_name] = {'vector': trait_vector, 'strength': float(clamp(strength))}
        architecture['traits'] = traits
        return architecture

    def _mutate_pathway(self, architecture: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """Optimize pathways: adjust small portions of weights in 'weights' dict"""
        if 'weights' not in architecture:
            return self._mutate_trait_formation(architecture, strength * 0.5)
        for key, w in architecture['weights'].items():
            if self._rng.random() < 0.3:
                noise = self._rng.normal(0.0, 0.01 * strength, size=np.shape(w))
                architecture['weights'][key] = (np.array(w) + noise).tolist()
        return architecture

    def _mutate_immune_pattern(self, architecture: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """
        Create/adjust an immune pattern that can quarantine or resist corrupted activation.
        Immune patterns are registered and used by enforcement to downweight corrupted channels.
        """
        pat_id = f"immune_{len(self.quarantine) + 1}"
        pattern = normalize(self._rng.standard_normal(32)).tolist()
        self.quarantine[pat_id] = {'pattern': pattern, 'strength': float(clamp(strength)), 'created': now_ts()}
        return architecture

    def _mutate_structural_rewire(self, architecture: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """Higher-risk structural rewire: change module connectivity descriptors (abstract)."""
        if 'graph' not in architecture:
            architecture['graph'] = {'nodes': [], 'edges': []}
        # add a tentative edge (abstract)
        node_a = f"n{self._rng.integers(0,50)}"
        node_b = f"n{self._rng.integers(0,50)}"
        architecture['graph'].setdefault('edges', []).append({'a': node_a, 'b': node_b, 'w': float(0.1 * strength)})
        return architecture

    def _mutate_weights(self, architecture: Dict[str, Any], strength: float) -> Dict[str, Any]:
        if 'weights' not in architecture:
            architecture['weights'] = {}
        # small random tweak
        for _ in range(0, max(1, int(2 * strength * 10))):
            key = f"w_{self._rng.integers(0,1000)}"
            val = self._rng.normal(0, 0.01 * strength, size=(8,)).tolist()
            architecture['weights'][key] = val
        return architecture

    # ----------------------------
    # Rollback / audit utilities
    # ----------------------------
    def rollback_last(self) -> Optional[MutationEvent]:
        """Rollback last mutation using ledger; move snapshot back into architecture store if present."""
        if not self.mutation_ledger:
            return None
        last = self.mutation_ledger.pop()
        # attempt to restore pre_snapshot in architecture store if matching key exists
        try:
            # best-effort restoration: if target name corresponds to snapshot key
            # caller must still fetch snapshot and re-inject into live system as needed
            # store in snapshots under event id for retrieval
            self.architecture_snapshots[f"rollback_{last.event_id}"] = last.pre_snapshot
            self.telemetry['mutations_total'] = max(0, self.telemetry.get('mutations_total', 1) - 1)
            logger.warning("rolled back mutation %s of type %s", last.event_id, last.mutation_type)
        except Exception:
            logger.exception("rollback failed")
        return last

    # ----------------------------
    # Ascension logic & checks
    # ----------------------------
    def _compute_capacity_score(self, architecture: Dict[str, Any]) -> float:
        """
        Compute an interpretable aggregate capacity score [0..1] from architecture capacities vector.
        Lower dims reduce score; higher average -> higher score.
        Expected architecture['capacities'] is a vector 0..1.
        """
        vec = np.array(architecture.get('capacities', []), dtype=float)
        if vec.size == 0:
            return 0.0
        # combine mean and a penalty for high variance (neutral comfort requires balanced capacities)
        mean = float(np.mean(vec))
        var_penalty = float(np.var(vec))
        score = clamp(mean - 0.5 * var_penalty)
        return score

    def _detect_neutral_comfort(self, architecture: Dict[str, Any]) -> float:
        """
        "Neutral comfort" is a measure of baseline stability: low surprise and balanced capacities.
        Returns a 0..1 metric (higher -> closer to neutral comfort).
        """
        cap_balance = 1.0 - float(np.std(architecture.get('capacities', [0.0])))
        surprise_factor = clamp(1.0 - math.tanh(self.accumulated_surprise / max(1.0, self.surprise_threshold)))
        return clamp(0.6 * cap_balance + 0.4 * surprise_factor)

    def check_and_attempt_ascension(self, architecture: Dict[str, Any]) -> Optional[AscensionEvent]:
        """
        Evaluate architecture against next tier requirements and perform ascension if met.
        Returns AscensionEvent if ascended, otherwise None.
        """
        next_tier = self.tier + 1
        if next_tier not in self.tier_config:
            return None

        cfg = self.tier_config[next_tier]
        cap_score = self._compute_capacity_score(architecture)
        neutral = self._detect_neutral_comfort(architecture)

        meets_capacity = cap_score >= cfg.get('min_capacity', 0.7)
        meets_neutral = neutral >= cfg.get('neutral_comfort', 0.2)

        logger.debug("ascension check tier %d: cap=%0.3f req=%0.3f neutral=%0.3f req=%0.3f",
                     next_tier, cap_score, cfg.get('min_capacity'), neutral, cfg.get('neutral_comfort'))

        if meets_capacity and meets_neutral:
            prev = self.tier
            self.tier = next_tier
            ev = AscensionEvent(
                event_id=str(uuid.uuid4()),
                from_tier=prev,
                to_tier=next_tier,
                reason=f"capacity {cap_score:.3f} >= {cfg.get('min_capacity')} and neutral {neutral:.3f} >= {cfg.get('neutral_comfort')}",
                snapshot=self._snapshot_architecture(architecture)
            )
            self.telemetry['ascensions'].append({'from': prev, 'to': next_tier, 'ts': now_ts()})
            logger.info("ascended from tier %d -> %d", prev, next_tier)
            # emit enforcement event
            self._emit_enforcement({'type': 'ascension', 'from': prev, 'to': next_tier, 'event': ev})
            return ev

        return None

    # ----------------------------
    # Top-level cycle entry (one evolutionary decision cycle)
    # ----------------------------
    def evolution_cycle(self,
                        architecture: Dict[str, Any],
                        reasoning_symmetry_health: float,
                        narrative_coherence: float,
                        prediction: Optional[np.ndarray] = None,
                        reality: Optional[np.ndarray] = None,
                        allow_mutation: bool = True) -> Dict[str, Any]:
        """
        Run a single decision cycle:
          - accumulate surprise (if preds/reality provided)
          - compute mutation probability
          - optionally apply mutation (safe) and record event
          - check ascension
          - update telemetry and return structured result
        """
        # update signals provided externally
        if prediction is not None and reality is not None:
            self.accumulate_surprise(prediction, reality, channel="epistemic")

        self.narrative_coherence = clamp(narrative_coherence)
        # symmetry health from CriticalThinking/Symmetry: 0..1 (1 is perfect)
        sym_health = clamp(reasoning_symmetry_health)
        mut_prob = self.compute_mutation_probability(sym_health, self.narrative_coherence)

        result = {
            'timestamp': now_ts(),
            'tier': self.tier,
            'symmetry_health': sym_health,
            'narrative_coherence': self.narrative_coherence,
            'corruption_pressure': self.corruption_pressure,
            'accumulated_surprise': float(self.accumulated_surprise),
            'mutation_triggered': False,
            'mutation_event': None,
            'ascension_event': None,
            'decision_probability': mut_prob
        }

        # determine random draw
        draw = float(self._rng.random())
        logger.debug("mutation draw=%0.4f vs prob=%0.4f", draw, mut_prob)
        if allow_mutation and draw < mut_prob:
            # apply mutation
            mut_type = self.select_mutation_type()
            # strength scaled by surprise and corruption but capped
            strength = clamp(0.1 + 0.9 * (self.accumulated_surprise / (1.0 + self.surprise_threshold)) + 0.2 * self.corruption_pressure)
            strength = clamp(strength * float(self._rng.uniform(0.5, 1.0)))
            # snapshot architecture pre-mutation
            arch_key = f"arch_{now_ts():.3f}"
            self._save_snapshot(arch_key, self._snapshot_architecture(architecture))
            try:
                ev = self.apply_mutation(architecture,
                                         mutation_type=mut_type,
                                         strength=strength,
                                         symmetry_health=sym_health,
                                         narrative_coherence=self.narrative_coherence)
                result['mutation_triggered'] = True
                result['mutation_event'] = ev
                # after mutation, small automatic reset of surprise to avoid runaway
                self.accumulated_surprise = max(0.0, self.accumulated_surprise * 0.5)
            except Exception as e:
                # if mutation failed, record and propagate
                logger.exception("mutation error: %s", e)
                result['mutation_error'] = str(e)

        # check ascension
        asc_ev = self.check_and_attempt_ascension(architecture)
        if asc_ev:
            result['ascension_event'] = asc_ev

        # decay corruption a bit per cycle (mitigation)
        self.decay_corruption(rate=0.005)

        # record symmetry history
        self.telemetry['symmetry_health_history'].append({'ts': now_ts(), 'symmetry': sym_health})

        return result

    # ----------------------------
    # Introspection & retrieval
    # ----------------------------
    def get_telemetry(self) -> Dict[str, Any]:
        return copy.deepcopy(self.telemetry)

    def get_mutation_ledger(self, last_n: Optional[int] = 50) -> List[Dict[str, Any]]:
        hist = self.mutation_ledger[-(last_n or len(self.mutation_ledger)):]
        return [dat.__dict__ for dat in hist]

    def export_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self.architecture_snapshots.get(key))

# ----------------------------
# Example quick test function (not unit-test; demonstrative)
# ----------------------------
def _demo_run():
    evo = EvolutionAlgorithmV2(rng_seed=42)
    # small mock architecture
    arch = {'capacities': [0.45, 0.5, 0.4, 0.48], 'weights': {'w0': np.zeros((4,)).tolist()}}
    pred = np.array([0.2, 0.3, 0.25, 0.22])
    real = np.array([0.5, 0.4, 0.6, 0.45])
    # simulate a few cycles
    for i in range(6):
        res = evo.evolution_cycle(arch, reasoning_symmetry_health=0.7, narrative_coherence=0.8, prediction=pred, reality=real)
        logger.info("cycle %d -> mutation_triggered=%s tier=%d surprise=%0.4f", i, res['mutation_triggered'], evo.tier, evo.accumulated_surprise)
    # print ledger
    ledger = evo.get_mutation_ledger()
    logger.info("ledger entries: %d", len(ledger))
    return evo, arch

# Run demo only if module is executed directly
if __name__ == "__main__":
    evo, arch = _demo_run()
    print("final capacities:", arch['capacities'])
