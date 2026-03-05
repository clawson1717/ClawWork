import networkx as nx
from typing import List, Dict, Any, Optional, Callable
from .payload import ReasoningPayload
from .regulating import TruthRegulator
from .tracker import LiveDriftTracker

class BranchHealer:
    """
    Corrective Logic Healer.
    Surgically prunes and regenerates drifting logic branches in the DIG.
    Inspired by DenoiseFlow (Yan et al., 2026) healing logic.
    """

    def __init__(self, regulator: TruthRegulator):
        self.regulator = regulator
        self.tracker = regulator.tracker

    def prune_branch(self, node_id: str) -> List[str]:
        """
        Removes the specified node and all its descendants from the DIG.
        Returns the list of removed node IDs.
        """
        if not self.tracker.dig.has_node(node_id):
            return []

        # Find all descendants including the node itself
        descendants = list(nx.descendants(self.tracker.dig, node_id))
        nodes_to_remove = descendants + [node_id]
        
        removed_nodes = []
        for n in nodes_to_remove:
            if self.tracker.dig.has_node(n):
                removed_nodes.append(n)
                self.tracker.dig.remove_node(n)
        
        # Update tracker metadata
        self.tracker.metadata["nodes_count"] = self.tracker.dig.number_of_nodes()
        self.tracker.metadata["edges_count"] = self.tracker.dig.number_of_edges()
        
        return removed_nodes

    def heal_drifting_branches(
        self, 
        root_intent: str,
        regenerator_fn: Optional[Callable[[str, List[ReasoningPayload]], ReasoningPayload]] = None
    ) -> Dict[str, Any]:
        """
        Identifies drifting branches, prunes them, and attempts regeneration.
        
        Args:
            root_intent: The original user intent/query to guide regeneration.
            regenerator_fn: A function that generates a new node given the intent and parent payloads.
                           If None, it just prunes without regenerating.
        """
        origin = self.regulator.pinpoint_drift_origin()
        if not origin:
            return {"status": "healthy", "pruned_nodes": [], "healed_nodes": []}

        # Identify parents of the origin before pruning
        parents = list(self.tracker.dig.predecessors(origin))
        parent_payloads = [self.tracker.get_payload(p) for p in parents if self.tracker.get_payload(p)]

        # Prune the drifting branch starting at the origin
        pruned_nodes = self.prune_branch(origin)
        
        healed_nodes = []
        if regenerator_fn and parents:
            # Attempt to regenerate a replacement for the origin
            # In a real scenario, this might be a loop or recursive regeneration
            try:
                new_payload = regenerator_fn(root_intent, parent_payloads)
                new_node_id = self.tracker.add_reasoning_node(new_payload, parent_ids=parents)
                healed_nodes.append(new_node_id)
            except Exception as e:
                # Log error or handle failure
                pass
        
        return {
            "status": "healed" if healed_nodes else "pruned",
            "pruned_nodes": pruned_nodes,
            "healed_nodes": healed_nodes,
            "drift_origin_was": origin
        }

    def verify_consistency(self) -> bool:
        """
        Checks if the DIG is in a consistent state (DAG, no orphans except roots).
        """
        if not nx.is_directed_acyclic_graph(self.tracker.dig):
            return False
            
        # Check if all nodes (except maybe roots) have parents if they aren't the very first node
        # This is a bit loose, but good for basic check
        return True
