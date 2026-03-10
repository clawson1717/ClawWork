from typing import List, Dict, Optional, Set
from src.node import TrajectoryNode, NodeStatus

class TrajectoryGraph:
    def __init__(self):
        self.nodes: Dict[str, TrajectoryNode] = {}

    def add_node(self, node: TrajectoryNode):
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes[node.id] = node

    def add_edge(self, parent_id: str, child_id: str):
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found.")
        if child_id not in self.nodes:
            raise ValueError(f"Child node {child_id} not found.")
        
        # We don't prevent cycles here for pruner testing, but it's good practice.
        parent_node = self.nodes[parent_id]
        child_node = self.nodes[child_id]
        
        if child_id not in parent_node.children_ids:
            parent_node.children_ids.append(child_id)
        
        child_node.parent_id = parent_id

    def get_node(self, node_id: str) -> Optional[TrajectoryNode]:
        return self.nodes.get(node_id)

    def prune_branch(self, node_id: str):
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.status = NodeStatus.PRUNED
        for child_id in node.children_ids:
            self.prune_branch(child_id)
            
    def get_all_nodes(self) -> Dict[str, TrajectoryNode]:
        return self.nodes
