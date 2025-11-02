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

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.content_pool: List[InformationObject] = []
        self.complexity_baseline = 0.5
        self._populate_initial_pool()

    def _populate_initial_pool(self, n: int = 200):
        for i in range(n):
            vec = np.random.randn(16)
            info = InformationObject(
                id=f"info_{i}",
                content=vec,
                complexity=float(np.clip(np.random.beta(2, 5), 0.0, 1.0)),
                credibility=float(np.clip(np.random.beta(3, 3), 0.0, 1.0)),
                category=random.choice(["text", "image", "sensor", "social"]),
                metadata={"source": random.choice(["web", "sim", "sensor"])} ,
            )
            self.content_pool.append(info)

    def adaptive_complexity_modulation(self, nnnc_state: Dict[str, Any]):
        current_utilization = nnnc_state.get("capacity_utilization", 0.5)
        if current_utilization < 0.3:
            self.complexity_baseline = float(np.clip(self.complexity_baseline + 0.05, 0.2, 0.9))
        elif current_utilization > 0.85:
            self.complexity_baseline = float(np.clip(self.complexity_baseline - 0.05, 0.2, 0.9))

    def generate_interaction_opportunity(self) -> InformationObject:
        complexity_distribution = np.random.normal(self.complexity_baseline, 0.15)
        candidates = [c for c in self.content_pool if abs(c.complexity - complexity_distribution) < 0.2]
        if not candidates:
            return random.choice(self.content_pool)
        return random.choice(candidates)

    def execute_action(self, action_command: Dict[str, Any]):
        """Execute an action in the NES (toy placeholder)"""
        # For now just log or return an effect
        return {"status": "executed", "action": action_command}

    def get_sensory_data(self) -> np.ndarray:
        info = self.generate_interaction_opportunity()
        vec = info.content
        # Add noise proportional to (1 - credibility)
        noise = np.random.randn(*vec.shape) * (1.0 - info.credibility) * 0.1
        return vec + noise
