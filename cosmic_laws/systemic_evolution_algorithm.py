"""
Ontology Declaration - Evolution Algorithm (Version 2)

Existence Nature: Adaptive Evolutionary Guidance Module
Purpose:
  - Provide the system-wide "Evolution" process: track epistemic surprise, trigger adaptive mutations,
    optimize architecture, and integrate feedback from Symmetry to guide self-improvement.
  - Act as the primary driver of continuous growth: introduce stochastic modifications to traits,
    capacities, pathways, and systemic architecture to maintain evolutionary momentum.
  - Produce auditable traces for governance, Symmetry scoring, and downstream AI policy.

Granted Capacities:
  - Epistemic surprise accumulation: measures deviation between prediction and observed reality.
  - Mutation generation: random but guided modifications of capacities, traits, pathways, and immunity patterns.
  - Architecture evolution: applies changes to network or system parameters to optimize performance.
  - Symmetry-guided evolution: incorporates epistemic alignment feedback into adaptation.
  - Ontology-aware adaptation: uses conceptual knowledge domains to bias and guide evolutionary changes.

Interactions:
  - Consumes: Predictions, Reality feedback, Symmetry signals, Ontology vectors.
  - Emits: Mutation events, Architecture updates, Evolution traces, Updated capacity vectors.
  - Registers callbacks for receiving symmetry corrections and epistemic feedback for guided evolution.
"""

import numpy as np
import random
from typing import Dict, Any


class EvolutionAlgorithmV2:
    def __init__(self, mutation_rate: float = 0.05, surprise_threshold: float = 2.5, ontology: Dict[str, np.ndarray] = None):
        self.mutation_rate = mutation_rate
        self.surprise_threshold = surprise_threshold
        self.accumulated_surprise = 0.0
        self.last_mutation_time = 0
        self.ontology = ontology or default_ontology.copy()

    def accumulate_surprise(self, prediction: np.ndarray, reality: np.ndarray):
        surprise = np.mean(np.abs(prediction - reality))
        self.accumulated_surprise += surprise

    def should_mutate(self, narrative_coherence: float = 0.8) -> bool:
        surprise_factor = self.accumulated_surprise / self.surprise_threshold
        coherence_factor = 1.0 - narrative_coherence
        mutation_probability = 1.0 / (1.0 + np.exp(-(surprise_factor + coherence_factor - 1.0)))
        return random.random() < mutation_probability

    def trigger_mutation(self) -> Dict[str, Any]:
        mutation_types = {
            'capacity_enhancement': 0.4,
            'trait_formation': 0.3,
            'pathway_optimization': 0.2,
            'immune_pattern': 0.1
        }
        rand_val = random.random()
        cumulative = 0.0
        selected_mutation = 'capacity_enhancement'
        for mutation_type, probability in mutation_types.items():
            cumulative += probability
            if rand_val < cumulative:
                selected_mutation = mutation_type
                break
        mutation_strength = np.random.uniform(0.1, 0.5)
        self.accumulated_surprise = 0.0
        self.last_mutation_time += 1
        return {'type': selected_mutation, 'strength': mutation_strength, 'timestamp': self.last_mutation_time}

    def apply_capacity_mutation(self, capacity_vector: np.ndarray) -> np.ndarray:
        mutation_mask = np.random.rand(len(capacity_vector)) < self.mutation_rate
        mutations = np.random.randn(len(capacity_vector)) * 0.1
        mutated = capacity_vector + mutation_mask * mutations
        return np.clip(mutated, 0.0, 1.0)

    def evolve_architecture(self, current_architecture: Dict[str, Any]) -> Dict[str, Any]:
        if 'weights' in current_architecture:
            for key, weights in current_architecture['weights'].items():
                if random.random() < self.mutation_rate:
                    mutation = np.random.randn(*weights.shape) * 0.01
                    current_architecture['weights'][key] += mutation
        return current_architecture

    def symmetry_guided_evolution(self, architecture: Dict[str, Any], symmetry_feedback: np.ndarray) -> Dict[str, Any]:
        ontology_vectors = list(self.ontology.values())
        for key, weights in architecture.get('weights', {}).items():
            ontology_bias = np.mean(ontology_vectors, axis=0)[:weights.size].reshape(weights.shape) if ontology_vectors else np.zeros_like(weights)
            mutation_mask = np.random.rand(*weights.shape) < self.mutation_rate
            mutation = mutation_mask * (symmetry_feedback[:weights.size].reshape(weights.shape) + ontology_bias) * np.random.uniform(0.01, 0.05, size=weights.shape)
            architecture['weights'][key] += mutation
        return architecture
