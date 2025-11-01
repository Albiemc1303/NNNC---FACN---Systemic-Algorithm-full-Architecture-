"""
NNNC Core Implementation
Neural Neutral Network Core with Five Axis Cognition Network (FACN)

This module implements the core NNNC architecture with 5 layers and autonomous intelligence.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import cosine
from scipy.stats import norm
import hashlib
import time


@dataclass
class CapacityDimension:
    """Represents a single capacity dimension in the CogFlux graph"""
    name: str
    value: float = 0.5
    activation_history: List[float] = field(default_factory=list)
    
    def activate(self, delta: float):
        """Update capacity value"""
        self.value = np.clip(self.value + delta, 0.0, 1.0)
        self.activation_history.append(self.value)


class SubconsciousMemory:
    """
    Inaccessible Subconscious Memory Layer (M_sub)
    Stores lifelong narrative, biases, traits, and experiences
    Uses LSH for content-addressable retrieval without direct access
    """
    
    def __init__(self, hash_size: int = 16, vector_dim: int = 64):
        self.hash_size = hash_size
        self.vector_dim = vector_dim
        self.episodic_store = []
        self.concept_graph = nx.Graph()
        self.narrative_attractors = {}
        self.traits = {}
        self.source_reliability = {}
        self.timestamp = 0
        
        np.random.seed(42)
        self.stable_random_vectors = np.random.randn(self.hash_size, self.vector_dim)
        
    def _hash_vector(self, vector: np.ndarray) -> str:
        """LSH-based hashing with stable random projections for content addressing"""
        if len(vector) != self.vector_dim:
            vector = np.pad(vector, (0, max(0, self.vector_dim - len(vector))))[:self.vector_dim]
        
        hash_bits = (self.stable_random_vectors @ vector > 0).astype(int)
        return ''.join(map(str, hash_bits))
    
    def store_experience(self, content: Dict[str, Any], affective_weight: float = 0.5):
        """Store experience in episodic memory"""
        self.timestamp += 1
        experience = {
            'id': self.timestamp,
            'content': content,
            'affective_weight': affective_weight,
            'timestamp': self.timestamp,
            'hash': self._hash_vector(np.random.rand(64))
        }
        self.episodic_store.append(experience)
        
        if len(self.episodic_store) > 1000:
            self.episodic_store = self.episodic_store[-1000:]
    
    def form_narrative_bias(self, concept: str, strength: float):
        """Form or strengthen narrative bias/trait"""
        if concept in self.narrative_attractors:
            current = self.narrative_attractors[concept]
            self.narrative_attractors[concept] = current + 0.1 * (strength - current)
        else:
            self.narrative_attractors[concept] = strength
            
        self.traits[concept] = {
            'strength': self.narrative_attractors[concept],
            'formed_at': self.timestamp
        }
    
    def get_narrative_influence(self, context_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate narrative influence vector based on stored biases and current context
        This is the inaccessible -> accessible influence channel
        """
        if not self.narrative_attractors:
            return np.zeros(16)
        
        base_influence = np.zeros(16)
        
        for concept, strength in self.narrative_attractors.items():
            concept_hash = self._hash_vector(np.random.rand(self.vector_dim))
            hash_value = int(concept_hash[:8], 2) % 16
            base_influence[hash_value] += strength * 0.1
        
        if context_vector is not None and len(self.episodic_store) > 0:
            recent_memories = self.episodic_store[-10:]
            for memory in recent_memories:
                memory_list = memory['content'].get('input', [0] * 16)
                memory_vec = np.array(memory_list)
                
                if len(memory_vec) < 16:
                    memory_vec = np.pad(memory_vec, (0, 16 - len(memory_vec)))
                elif len(memory_vec) > 16:
                    memory_vec = memory_vec[:16]
                    
                affective = memory.get('affective_weight', 0.5)
                base_influence += memory_vec * affective * 0.05
        
        base_influence = np.clip(base_influence, -1.0, 1.0)
        return base_influence
    
    def update_source_reliability(self, source_id: str, reliability: float):
        """Update reliability score for information source"""
        if source_id in self.source_reliability:
            current = self.source_reliability[source_id]
            self.source_reliability[source_id] = current + 0.2 * (reliability - current)
        else:
            self.source_reliability[source_id] = reliability
    
    def query_source_reliability(self, source_id: str) -> float:
        """Query source reliability (for Critical Thinking)"""
        return self.source_reliability.get(source_id, 0.5)


class CogFluxEngine:
    """
    CogFlux Mathematical Framework
    Dynamic graph-based capacity modeling with Bayesian overlay
    """
    
    def __init__(self, num_dimensions: int = 12):
        self.graph = nx.Graph()
        self.dimensions = self._initialize_dimensions(num_dimensions)
        self.lambda_interdep = 0.3
        self._build_graph()
    
    def _initialize_dimensions(self, num_dims: int) -> Dict[str, CapacityDimension]:
        """Initialize capacity dimensions"""
        dimension_names = [
            'learning_capacity', 'pattern_abstraction', 'memory_consolidation',
            'social_internalization', 'generalization', 'metacognitive',
            'embodied_affective', 'temporal_coordination', 'metabolic_efficiency',
            'neuromodulatory_balance', 'network_efficiency', 'innovation'
        ]
        
        return {name: CapacityDimension(name=name, value=np.random.uniform(0.3, 0.7))
                for name in dimension_names[:num_dims]}
    
    def _build_graph(self):
        """Build dynamic graph of capacity dimensions"""
        dim_names = list(self.dimensions.keys())
        
        for i, dim1 in enumerate(dim_names):
            self.graph.add_node(dim1, capacity=self.dimensions[dim1].value)
            
            for dim2 in dim_names[i+1:]:
                weight = np.random.uniform(0.3, 0.9)
                self.graph.add_edge(dim1, dim2, weight=weight)
    
    def calculate_global_efficiency(self) -> float:
        """Calculate η(G) - global efficiency of capacity network with proper normalization"""
        N = len(self.graph.nodes())
        if N < 2:
            return 0.0
        
        total_efficiency = 0.0
        node_list = list(self.graph.nodes())
        
        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(
                            self.graph, node_i, node_j, weight='weight'
                        )
                        if path_length > 0:
                            total_efficiency += 1.0 / path_length
                    except nx.NetworkXNoPath:
                        continue
        
        eta_G = (1.0 / (N * (N - 1))) * total_efficiency
        return eta_G
    
    def calculate_total_capacity(self) -> float:
        """Calculate Total Capacity = η(G) + λ * Σ(Interdependencies) with proper edge weighting"""
        eta_G = self.calculate_global_efficiency()
        
        edge_interdependencies = sum([
            self.graph[u][v]['weight'] for u, v in self.graph.edges()
        ])
        
        total_capacity = eta_G + (self.lambda_interdep * edge_interdependencies)
        return total_capacity
    
    def update_dimension(self, dim_name: str, delta: float):
        """Update specific capacity dimension"""
        if dim_name in self.dimensions:
            self.dimensions[dim_name].activate(delta)
            self.graph.nodes[dim_name]['capacity'] = self.dimensions[dim_name].value
    
    def get_capacity_vector(self) -> np.ndarray:
        """Get current capacity values as vector"""
        return np.array([dim.value for dim in self.dimensions.values()])


class FACN:
    """
    Five Axis Cognition Network
    Implements the 5-layer architecture: Input → Hidden → Subconscious → Meta-Cognitive → Output
    """
    
    def __init__(self, input_size: int = 64):
        self.input_size = input_size
        
        self.input_layer = np.zeros(input_size)
        self.hidden_layers = [np.zeros(32), np.zeros(16)]
        self.subconscious = SubconsciousMemory()
        self.meta_cognitive_state = np.zeros(16)
        self.output_layer = np.zeros(10)
        
        self.weights_input_hidden = np.random.randn(input_size, 32) * 0.1
        self.weights_hidden1_hidden2 = np.random.randn(32, 16) * 0.1
        self.weights_to_meta = np.random.randn(16, 16) * 0.1
        self.weights_meta_output = np.random.randn(16, 10) * 0.1
    
    def process_input(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Process data through all 5 layers with proper Meta-Cognitive sovereignty
        Only Meta-Cognitive layer makes the final decision
        """
        
        self.input_layer = input_data
        
        self.hidden_layers[0] = np.tanh(self.weights_input_hidden.T @ input_data)
        
        self.hidden_layers[1] = np.tanh(self.weights_hidden1_hidden2.T @ self.hidden_layers[0])
        
        experience = {
            'input': input_data.tolist()[:10],
            'hidden_activation': float(np.mean(self.hidden_layers[1])),
            'timestamp': time.time()
        }
        affective_weight = float(np.mean(np.abs(self.hidden_layers[1])))
        self.subconscious.store_experience(experience, affective_weight=affective_weight)
        
        narrative_influence = self.subconscious.get_narrative_influence(self.hidden_layers[1])
        
        hidden_state = self.hidden_layers[1]
        combined = np.concatenate([hidden_state, narrative_influence[:6]])[:16]
        
        self.meta_cognitive_state = np.tanh(self.weights_to_meta.T @ combined)
        
        inner_dialogue_iterations = 3
        for _ in range(inner_dialogue_iterations):
            dialogue_feedback = self.meta_cognitive_state * 0.1
            self.meta_cognitive_state = np.tanh(
                self.meta_cognitive_state + dialogue_feedback
            )
        
        self.output_layer = np.tanh(self.weights_meta_output.T @ self.meta_cognitive_state)
        
        return {
            'input': self.input_layer,
            'hidden': self.hidden_layers,
            'subconscious_influence': narrative_influence,
            'meta_state': self.meta_cognitive_state,
            'output': self.output_layer
        }
    
    def hebbian_update(self, learning_rate: float = 0.01):
        """Apply Hebbian plasticity to weights"""
        if np.std(self.hidden_layers[0]) > 0.1:
            plasticity_delta = learning_rate * np.outer(self.input_layer, self.hidden_layers[0])
            self.weights_input_hidden += plasticity_delta.T * 0.1


class NNNCCore:
    """
    Neural Neutral Network Core
    The central autonomous intelligence system with meta-cognitive control
    """
    
    def __init__(self):
        self.facn = FACN(input_size=64)
        self.cogflux = CogFluxEngine(num_dimensions=12)
        
        self.decision_history = []
        self.autonomy_level = 0.5
        self.existence_time = 0
        
    def perceive_and_decide(self, environmental_input: np.ndarray) -> Dict[str, Any]:
        """
        Main cognitive loop: Perceive → Process → Decide
        The Meta-Cognitive layer makes the final autonomous decision
        """
        self.existence_time += 1
        
        layer_states = self.facn.process_input(environmental_input)
        
        total_capacity = self.cogflux.calculate_total_capacity()
        global_efficiency = self.cogflux.calculate_global_efficiency()
        
        meta_state = layer_states['meta_state']
        output_action = layer_states['output']
        
        decision_confidence = float(np.mean(np.abs(output_action)))
        
        decision = {
            'action': output_action,
            'confidence': decision_confidence,
            'capacity': total_capacity,
            'efficiency': global_efficiency,
            'timestamp': self.existence_time,
            'meta_state': meta_state
        }
        
        self.decision_history.append(decision)
        
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return decision
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state for monitoring"""
        return {
            'capacity_dimensions': {name: dim.value 
                                   for name, dim in self.cogflux.dimensions.items()},
            'global_efficiency': self.cogflux.calculate_global_efficiency(),
            'total_capacity': self.cogflux.calculate_total_capacity(),
            'narrative_biases': self.facn.subconscious.narrative_attractors,
            'traits': self.facn.subconscious.traits,
            'existence_time': self.existence_time,
            'decision_count': len(self.decision_history)
        }
