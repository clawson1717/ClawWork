import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .payload import ReasoningPayload
from .tracker import LiveDriftTracker
from .sensing import UncertaintySenser
from .drift import DriftCalculator

class TruthRegulator:
    """
    Regulator of Truth-Resilience.
    Pinpoints the exact DIG node where drift exceeded the Resilience Threshold
    and calculates Truth-Resilience scores for reasoning nodes.
    Inspired by DenoiseFlow (Yan et al., 2026) regulating logic.
    """

    def __init__(
        self, 
        tracker: LiveDriftTracker, 
        uncertainty_senser: UncertaintySenser,
        drift_calculator: DriftCalculator,
        resilience_threshold: float = 0.6,
        truth_weight: float = 0.7
    ):
        """
        Initialize the TruthRegulator.

        Args:
            tracker: The LiveDriftTracker containing the DIG.
            uncertainty_senser: The UncertaintySenser for ambiguity flow.
            drift_calculator: The DriftCalculator for semantic drift.
            resilience_threshold: The cumulative threshold at which a node is considered 'drifting'.
            truth_weight: Importance of drift vs uncertainty in resilience (0.0 to 1.0).
        """
        self.tracker = tracker
        self.senser = uncertainty_senser
        self.drift_calc = drift_calculator
        self.resilience_threshold = resilience_threshold
        self.truth_weight = truth_weight

    def calculate_resilience_score(self, node_id: str) -> float:
        """
        Calculates the 'Truth-Resilience' score for a node.
        High score means the node is robust and truthful.
        Low score means the node is drifting or highly uncertain.
        
        Score = 1 - (weight * drift + (1 - weight) * uncertainty_flow)
        """
        payload = self.tracker.get_payload(node_id)
        if not payload:
            return 0.0
            
        drift = payload.drift_score
        u_flow = self.senser.get_uncertainty_flow(node_id)
        
        # Combined 'Drift Impact'
        impact = (self.truth_weight * drift) + ((1.0 - self.truth_weight) * u_flow)
        
        # Resilience is the inverse of impact
        resilience = 1.0 - impact
        return float(np.clip(resilience, 0.0, 1.0))

    def pinpoint_drift_origin(self) -> Optional[str]:
        """
        Traverses the DIG to find the earliest node where the 
        Resilience Score falls below the threshold.
        
        This identifies the 'patient zero' of reasoning drift.
        """
        drifting_nodes = []
        
        # Get all nodes that are below the threshold
        for node_id in self.tracker.dig.nodes():
            resilience = self.calculate_resilience_score(node_id)
            if resilience < (1.0 - self.resilience_threshold):
                drifting_nodes.append(node_id)
        
        if not drifting_nodes:
            return None
            
        # Find the node with the shortest path from a root
        origin_node = None
        min_depth = float('inf')
        
        for node_id in drifting_nodes:
            path = self.tracker.get_causal_path(node_id)
            if path and len(path) < min_depth:
                min_depth = len(path)
                origin_node = node_id
                
        return origin_node

    def get_truth_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive report of the DIG's truth-resilience.
        """
        report = {
            "total_nodes": self.tracker.dig.number_of_nodes(),
            "drifting_nodes": [],
            "resilience_scores": {},
            "drift_origin": self.pinpoint_drift_origin(),
            "status": "healthy"
        }
        
        threshold = 1.0 - self.resilience_threshold
        
        for node_id in self.tracker.dig.nodes():
            resilience = self.calculate_resilience_score(node_id)
            report["resilience_scores"][node_id] = resilience
            
            if resilience < threshold:
                payload = self.tracker.get_payload(node_id)
                report["drifting_nodes"].append({
                    "node_id": node_id,
                    "score": resilience,
                    "content": payload.content[:100] + "..." if payload else ""
                })
        
        if report["drifting_nodes"]:
            report["status"] = "drifting"
            
        return report

    def identify_denoise_targets(self) -> List[str]:
        """
        Returns a list of node IDs that require DenoiseFlow correction.
        Identifies nodes that are either the origin of drift or downstream 
        affected nodes with low resilience.
        """
        origin = self.pinpoint_drift_origin()
        if not origin:
            return []
            
        # All descendants of the drift origin are potential targets
        # plus the origin itself.
        import networkx as nx
        targets = list(nx.descendants(self.tracker.dig, origin))
        targets.append(origin)
        
        # Filter for only those that are actually below threshold
        threshold = 1.0 - self.resilience_threshold
        filtered_targets = [
            t for t in targets 
            if self.calculate_resilience_score(t) < threshold
        ]
        
        return filtered_targets
