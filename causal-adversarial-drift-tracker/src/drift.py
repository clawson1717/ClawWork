import numpy as np
from typing import List, Optional
from .payload import ReasoningPayload
from .tracker import LiveDriftTracker

class DriftCalculator:
    """
    Calculates semantic drift for ReasoningPayload nodes within a LiveDriftTracker's DIG.
    Uses cosine distance between semantic vectors as the primary metric.
    """

    def __init__(self, tracker: LiveDriftTracker):
        self.tracker = tracker

    @staticmethod
    def cosine_distance(v1: List[float], v2: List[float]) -> float:
        """
        Calculates the cosine distance between two vectors.
        Distance = 1 - Cosine Similarity.
        Range: [0, 2] (0 means identical, 1 means orthogonal, 2 means opposite).
        """
        if not v1 or not v2:
            return 0.0
        
        arr1 = np.array(v1)
        arr2 = np.array(v2)
        
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = np.dot(arr1, arr2) / (norm1 * norm2)
        # Clip similarity to avoid floating point errors leading to values slightly outside [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return 1.0 - similarity

    def calculate_node_drift(self, node_id: str, root_id: str) -> float:
        """
        Calculates the semantic drift of a specific node relative to a root (intent) node.
        """
        node_payload = self.tracker.get_payload(node_id)
        root_payload = self.tracker.get_payload(root_id)
        
        if not node_payload or not root_payload:
            return 0.0
            
        return self.cosine_distance(node_payload.semantic_vector, root_payload.semantic_vector)

    def calculate_path_drift(self, node_id: str) -> float:
        """
        Calculates the cumulative drift along the causal path from root to the node.
        Uses the path provided by the tracker.
        Returns the average drift across the path nodes relative to the root.
        """
        path = self.tracker.get_causal_path(node_id)
        if not path:
            return 0.0
            
        root_id = path[0]
        root_payload = self.tracker.get_payload(root_id)
        if not root_payload:
            return 0.0
            
        total_drift = 0.0
        for pid in path:
            payload = self.tracker.get_payload(pid)
            if payload:
                # Cumulative drift is often defined as distance from origin (root intent)
                total_drift += self.cosine_distance(payload.semantic_vector, root_payload.semantic_vector)
                
        return total_drift / len(path)

    def update_tracker_drift_scores(self):
        """
        Iterates through all nodes in the tracker's DIG and updates their drift_score.
        Drift is calculated relative to the nearest root (anchor intent) of each node.
        """
        for node_id in self.tracker.dig.nodes():
            path = self.tracker.get_causal_path(node_id)
            if not path:
                continue
                
            root_id = path[0]
            # We use direct distance from root as the primary drift score for the payload
            drift = self.calculate_node_drift(node_id, root_id)
            
            payload = self.tracker.get_payload(node_id)
            if payload:
                payload.drift_score = drift
