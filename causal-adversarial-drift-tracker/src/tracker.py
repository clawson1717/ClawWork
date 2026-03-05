import networkx as nx
from typing import Dict, List, Optional, Any
import json
import datetime
from .payload import ReasoningPayload

class LiveDriftTracker:
    """
    Manages the Dynamic Interaction Graph (DIG) for tracking reasoning branches and causal drift.
    """

    def __init__(self):
        self.dig = nx.DiGraph()
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "nodes_count": 0,
            "edges_count": 0
        }

    def add_reasoning_node(self, payload: ReasoningPayload, parent_ids: Optional[List[str]] = None) -> str:
        """
        Adds a ReasoningPayload as a node in the DIG.
        
        Args:
            payload: The ReasoningPayload to add.
            parent_ids: Optional list of source_ids representing causal parents of this node.
            
        Returns:
            The source_id of the added node.
        """
        node_id = payload.source_id
        
        # Add node with payload data as attributes
        self.dig.add_node(
            node_id, 
            payload=payload,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # If parent IDs are provided, create causal edges
        if parent_ids:
            for p_id in parent_ids:
                if self.dig.has_node(p_id):
                    self.dig.add_edge(p_id, node_id)
                else:
                    # Log or handle missing parent? For now, we'll just skip or could raise error
                    pass
        
        self.metadata["nodes_count"] = self.dig.number_of_nodes()
        self.metadata["edges_count"] = self.dig.number_of_edges()
        
        return node_id

    def get_payload(self, node_id: str) -> Optional[ReasoningPayload]:
        """Retrieves the ReasoningPayload for a given node ID."""
        if self.dig.has_node(node_id):
            return self.dig.nodes[node_id].get("payload")
        return None

    def get_causal_path(self, node_id: str) -> List[str]:
        """Returns the list of node IDs representing the causal path from root(s) to the specified node."""
        if not self.dig.has_node(node_id):
            return []
        
        # Find all ancestors (those that have a path TO node_id)
        ancestors = nx.ancestors(self.dig, node_id)
        # Sort or trace back? Simple approach: find a shortest path from any root to node_id
        roots = [n for n, d in self.dig.in_degree() if d == 0]
        
        paths = []
        for root in roots:
            if nx.has_path(self.dig, root, node_id):
                paths.append(nx.shortest_path(self.dig, root, node_id))
        
        # Return the longest path found among all possibilities (or first one)
        if not paths:
            return [node_id] # It is a root
        
        return max(paths, key=len)

    def get_drift_metrics(self) -> Dict[str, Any]:
        """Calculates aggregate drift metrics across the entire DIG."""
        if not self.dig.nodes:
            return {"avg_drift": 0.0, "max_drift": 0.0}
            
        drift_scores = [
            data["payload"].drift_score 
            for _, data in self.dig.nodes(data=True) 
            if "payload" in data
        ]
        
        if not drift_scores:
            return {"avg_drift": 0.0, "max_drift": 0.0}
            
        return {
            "avg_drift": sum(drift_scores) / len(drift_scores),
            "max_drift": max(drift_scores),
            "node_count": len(drift_scores)
        }

    def export_to_dot(self, path: Optional[str] = None) -> str:
        """Exports the DIG to DOT format for visualization."""
        # Convert graph to DOT format string
        # node labels use source_id and drift_score
        dot_lines = ["digraph DIG {"]
        for node, data in self.dig.nodes(data=True):
            payload = data.get("payload")
            label = f"{node}\nDrift: {payload.drift_score:.2f}" if payload else node
            dot_lines.append(f'  "{node}" [label="{label}"];')
            
        for u, v in self.dig.edges():
            dot_lines.append(f'  "{u}" -> "{v}";')
            
        dot_lines.append("}")
        dot_str = "\n".join(dot_lines)
        
        if path:
            with open(path, "w") as f:
                f.write(dot_str)
                
        return dot_str

    def to_adjacency_list(self) -> Dict[str, List[str]]:
        """Returns the DIG as an adjacency list (node ID mapped to its children)."""
        adj = {}
        for node in self.dig.nodes():
            adj[node] = list(self.dig.successors(node))
        return adj
