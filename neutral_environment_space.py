"""
Neutral Environment Space (NES)
Self-consistent reality generator where NNNC exists and interacts autonomously
No pre-assigned tasks or goals - pure emergent existence
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class InformationObject:
    """An object/event/information in the NES"""
    id: str
    content: np.ndarray
    complexity: float
    credibility: float
    category: str
    metadata: Dict[str, Any]


class NeutralEnvironmentSpace:
    """
    NES - The metaphysical container where NNNC lives
    Provides random information encounters with adaptive complexity
    """
    
    def __init__(self, complexity_baseline: float = 0.5):
        self.complexity_baseline = complexity_baseline
        self.time = 0
        self.content_pool = self._initialize_content_pool()
        self.interaction_history = []
        self.event_log = []
        
    def _initialize_content_pool(self) -> List[InformationObject]:
        """Create diverse content for NNNC to encounter"""
        content_pool = []
        
        categories = [
            'scientific_article',
            'philosophical_text',
            'conspiracy_theory',
            'factual_news',
            'misleading_blog',
            'mathematical_concept',
            'ethical_dilemma',
            'creative_fiction',
            'technical_documentation',
            'social_interaction'
        ]
        
        for i, category in enumerate(categories * 5):
            if category == 'conspiracy_theory':
                credibility = np.random.uniform(0.1, 0.3)
                complexity = np.random.uniform(0.3, 0.6)
            elif category == 'scientific_article':
                credibility = np.random.uniform(0.7, 0.95)
                complexity = np.random.uniform(0.6, 0.9)
            elif category == 'misleading_blog':
                credibility = np.random.uniform(0.2, 0.4)
                complexity = np.random.uniform(0.3, 0.5)
            elif category == 'factual_news':
                credibility = np.random.uniform(0.6, 0.85)
                complexity = np.random.uniform(0.4, 0.7)
            else:
                credibility = np.random.uniform(0.4, 0.7)
                complexity = np.random.uniform(0.3, 0.8)
            
            content_vector = np.random.randn(64)
            content_vector = content_vector / (np.linalg.norm(content_vector) + 1e-8)
            
            if category == 'conspiracy_theory':
                content_vector[:10] = np.abs(content_vector[:10]) + 1.0
            
            obj = InformationObject(
                id=f"{category}_{i}",
                content=content_vector,
                complexity=complexity,
                credibility=credibility,
                category=category,
                metadata={
                    'created_at': i,
                    'source': f"source_{random.randint(1, 10)}"
                }
            )
            content_pool.append(obj)
        
        return content_pool
    
    def adaptive_complexity_modulation(self, nnnc_capacity_utilization: float):
        """
        Adjust environmental complexity based on NNNC's current state
        Homeostatic regulation - match environment to organism capability
        """
        target_utilization = 0.6
        
        if nnnc_capacity_utilization < 0.3:
            self.complexity_baseline = min(0.9, self.complexity_baseline + 0.05)
        elif nnnc_capacity_utilization > 0.85:
            self.complexity_baseline = max(0.2, self.complexity_baseline - 0.05)
        
        self.complexity_baseline = np.clip(self.complexity_baseline, 0.2, 0.9)
    
    def generate_interaction_opportunity(self) -> InformationObject:
        """
        Generate next information encounter
        Weighted by current complexity setting
        """
        
        complexity_target = self.complexity_baseline + np.random.normal(0, 0.15)
        complexity_target = np.clip(complexity_target, 0.1, 1.0)
        
        available_content = [
            obj for obj in self.content_pool
            if abs(obj.complexity - complexity_target) < 0.3
        ]
        
        if not available_content:
            available_content = self.content_pool
        
        selected = random.choice(available_content)
        
        return selected
    
    def record_interaction(self, nnnc_decision: Dict[str, Any], 
                          encountered_object: InformationObject):
        """Record NNNC's interaction with environment"""
        
        action_magnitude = float(np.linalg.norm(nnnc_decision.get('action', np.zeros(10))))
        
        interaction_record = {
            'time': self.time,
            'object_id': encountered_object.id,
            'object_category': encountered_object.category,
            'object_credibility': encountered_object.credibility,
            'object_complexity': encountered_object.complexity,
            'nnnc_action_magnitude': action_magnitude,
            'nnnc_confidence': nnnc_decision.get('confidence', 0.0),
            'nnnc_capacity': nnnc_decision.get('capacity', 0.0)
        }
        
        self.interaction_history.append(interaction_record)
        self.time += 1
        
        if len(self.interaction_history) > 200:
            self.interaction_history = self.interaction_history[-200:]
        
        return interaction_record
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get current NES state"""
        
        if self.interaction_history:
            recent_complexity = np.mean([r['object_complexity'] 
                                        for r in self.interaction_history[-10:]])
            recent_credibility = np.mean([r['object_credibility'] 
                                         for r in self.interaction_history[-10:]])
        else:
            recent_complexity = self.complexity_baseline
            recent_credibility = 0.5
        
        return {
            'time': self.time,
            'complexity_baseline': self.complexity_baseline,
            'recent_avg_complexity': recent_complexity,
            'recent_avg_credibility': recent_credibility,
            'total_interactions': len(self.interaction_history),
            'content_pool_size': len(self.content_pool)
        }
    
    def simulate_autonomous_existence(self, nnnc_core: Any, steps: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate NNNC's autonomous existence in NES
        No tasks, no goals - pure emergent interaction
        """
        
        simulation_log = []
        
        for step in range(steps):
            
            encountered = self.generate_interaction_opportunity()
            
            decision = nnnc_core.perceive_and_decide(encountered.content)
            
            interaction = self.record_interaction(decision, encountered)
            
            system_state = nnnc_core.get_system_state()
            capacity_utilization = system_state['total_capacity'] / 10.0
            
            self.adaptive_complexity_modulation(capacity_utilization)
            
            step_log = {
                'step': step,
                'encountered': {
                    'id': encountered.id,
                    'category': encountered.category,
                    'complexity': encountered.complexity,
                    'credibility': encountered.credibility
                },
                'decision': decision,
                'system_state': system_state,
                'interaction': interaction
            }
            
            simulation_log.append(step_log)
        
        return simulation_log


class RealityGenerator:
    """
    Advanced NES feature: Generate self-consistent reality with ontological invariants
    Enforces causality, energy conservation, identity continuity, temporal coherence
    """
    
    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim
        self.ontological_invariants = {
            'causality': True,
            'energy_conservation': True,
            'identity_continuity': True,
            'temporal_coherence': True
        }
        
        self.reality_state = np.random.randn(state_dim) * 0.5
        self.previous_state = self.reality_state.copy()
        self.energy_baseline = float(np.sum(self.reality_state ** 2))
        self.time_step = 0
        self.causal_chain = []
        
    def enforce_invariants(self, proposed_state: np.ndarray) -> np.ndarray:
        """Ensure reality remains self-consistent across all invariants"""
        constrained_state = proposed_state.copy()
        
        if self.ontological_invariants['energy_conservation']:
            energy_proposed = np.sum(constrained_state ** 2)
            max_energy_drift = 0.1 * self.energy_baseline
            
            if abs(energy_proposed - self.energy_baseline) > max_energy_drift:
                scale_factor = np.sqrt(self.energy_baseline / (energy_proposed + 1e-8))
                constrained_state = constrained_state * scale_factor
        
        if self.ontological_invariants['temporal_coherence']:
            max_change_rate = 0.3
            constrained_state = (
                (1 - max_change_rate) * self.reality_state + 
                max_change_rate * constrained_state
            )
        
        if self.ontological_invariants['causality']:
            if len(self.causal_chain) > 0:
                last_cause = self.causal_chain[-1]
                cause_influence = last_cause['effect_vector'][:len(constrained_state)]
                constrained_state += cause_influence * 0.1
        
        if self.ontological_invariants['identity_continuity']:
            identity_signature = self.reality_state[:8]
            constrained_state[:8] = 0.7 * identity_signature + 0.3 * constrained_state[:8]
        
        return constrained_state
    
    def generate_reality_packet(self) -> Dict[str, Any]:
        """Generate consistent sensory data packet with enforced invariants"""
        
        delta = np.random.randn(self.state_dim) * 0.15
        proposed = self.reality_state + delta
        
        self.previous_state = self.reality_state.copy()
        
        self.reality_state = self.enforce_invariants(proposed)
        
        state_change = self.reality_state - self.previous_state
        self.causal_chain.append({
            'time': self.time_step,
            'effect_vector': state_change,
            'magnitude': float(np.linalg.norm(state_change))
        })
        
        if len(self.causal_chain) > 10:
            self.causal_chain = self.causal_chain[-10:]
        
        self.time_step += 1
        
        current_energy = float(np.sum(self.reality_state ** 2))
        coherence = 1.0 / (1.0 + np.std(self.reality_state))
        
        return {
            'state': self.reality_state.copy(),
            'invariants_satisfied': all(self.ontological_invariants.values()),
            'energy_level': current_energy,
            'energy_drift': abs(current_energy - self.energy_baseline),
            'coherence': coherence,
            'causal_depth': len(self.causal_chain),
            'time_step': self.time_step
        }