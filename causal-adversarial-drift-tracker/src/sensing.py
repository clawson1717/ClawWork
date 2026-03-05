import numpy as np
from typing import List, Dict, Any, Optional
from .payload import ReasoningPayload
from .tracker import LiveDriftTracker

class UncertaintySenser:
    """
    Senses semantic ambiguity and uncertainty flow through the Dynamic Interaction Graph (DIG).
    Inspired by DenoiseFlow (Yan et al., 2026) sensing logic.
    """

    def __init__(self, tracker: LiveDriftTracker, alpha: float = 0.5, propagation_decay: float = 0.8):
        """
        Initialize the UncertaintySenser.

        Args:
            tracker: The LiveDriftTracker containing the DIG.
            alpha: Weighting factor between ambiguity and drift (0.0 to 1.0).
            propagation_decay: Factor by which parent uncertainty decays as it propagates to children.
        """
        self.tracker = tracker
        self.alpha = alpha
        self.propagation_decay = propagation_decay
        # Heuristic keywords for uncertainty/ambiguity sensing
        self.uncertainty_triggers = [
            "maybe", "perhaps", "uncertain", "not sure", "likely", 
            "possibly", "could be", "unclear", "ambiguous", "depending on",
            "assumption", "assume", "potentially", "hypothetically"
        ]

    def score_ambiguity(self, payload: ReasoningPayload) -> float:
        """
        Calculates a heuristic semantic ambiguity score based on content analysis.
        
        Sensing logic:
        1. Keyword density of uncertainty triggers.
        2. Content length penalty (shorter reasoning steps often hide ambiguity).
        """
        if not payload.content:
            return 1.0
            
        content_lower = payload.content.lower()
        words = content_lower.split()
        if not words:
            return 1.0
            
        trigger_count = sum(1 for word in self.uncertainty_triggers if word in words)
        
        # Density-based scoring
        density = trigger_count / len(words)
        # Boost score slightly for very short snippets that contains triggers
        score = density * 5.0 
        
        return float(np.clip(score, 0.0, 1.0))

    def calculate_node_uncertainty(self, node_id: str) -> float:
        """
        Calculates the combined uncertainty for a single node.
        Integration of local ambiguity and existing drift score.
        
        Uncertainty = alpha * Ambiguity + (1 - alpha) * Drift
        """
        payload = self.tracker.get_payload(node_id)
        if not payload:
            return 0.0
            
        ambiguity = self.score_ambiguity(payload)
        drift = payload.drift_score
        
        # Combined intrinsic uncertainty
        return (self.alpha * ambiguity) + ((1.0 - self.alpha) * drift)

    def get_uncertainty_flow(self, node_id: str) -> float:
        """
        Detects 'uncertainty propagation' through the DIG.
        Calculates the total effective uncertainty at a node, including
        propagated uncertainty from its causal parents.
        
        U_total = U_intrinsic + sum(U_parent * decay)
        """
        return self._get_uncertainty_flow_recursive(node_id, set())

    def _get_uncertainty_flow_recursive(self, node_id: str, visited: set) -> float:
        if not self.tracker.dig.has_node(node_id) or node_id in visited:
            return 0.0
            
        visited.add(node_id)
        intrinsic_uncertainty = self.calculate_node_uncertainty(node_id)
        
        parents = list(self.tracker.dig.predecessors(node_id))
        if not parents:
            return float(np.clip(intrinsic_uncertainty, 0.0, 1.0))
            
        propagated_uncertainty = 0.0
        for p_id in parents:
            p_flow = self._get_uncertainty_flow_recursive(p_id, visited)
            propagated_uncertainty += p_flow * self.propagation_decay
            
        avg_propagated = propagated_uncertainty / len(parents)
        
        total_uncertainty = intrinsic_uncertainty + avg_propagated
        return float(np.clip(total_uncertainty, 0.0, 1.0))

    def sense_drift_hazards(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identifies nodes in the DIG that are high-risk for causal drift 
        due to high uncertainty flow.
        """
        hazards = []
        for node_id in self.tracker.dig.nodes():
            u_flow = self.get_uncertainty_flow(node_id)
            if u_flow >= threshold:
                payload = self.tracker.get_payload(node_id)
                hazards.append({
                    "node_id": node_id,
                    "uncertainty_flow": u_flow,
                    "content_preview": payload.content[:50] + "..." if payload else "",
                    "drift_score": payload.drift_score if payload else 0.0
                })
        
        # Sort by highest flow first
        return sorted(hazards, key=lambda x: x["uncertainty_flow"], reverse=True)
