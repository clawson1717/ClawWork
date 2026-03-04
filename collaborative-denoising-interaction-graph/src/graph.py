import networkx as nx
from typing import Dict, List, Set
from .node import InteractionNode

class InteractionGraph:
    def __init__(self):
        self.nodes: Dict[str, InteractionNode] = {}
        self.nx_graph = nx.DiGraph()

    def add_node(self, node: InteractionNode):
        """
        Adds an InteractionNode to the graph.
        Validates that all causal parents exist and prevents cycles.
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists.")

        # Validate parents exist
        for parent_id in node.causal_parents:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent ID {parent_id} not found in graph.")

        # Check for cycles before adding (optimistic addition)
        temp_edges = [(parent_id, node.id) for parent_id in node.causal_parents]
        self.nx_graph.add_edges_from(temp_edges)
        
        if not nx.is_directed_acyclic_graph(self.nx_graph):
            # Rollback nx_graph and edges
            self.nx_graph.remove_edges_from(temp_edges)
            # Remove the node node.id from nx_graph if it was newly created
            # This is tricky because nx_graph.add_edges_from might call add_node.
            # However, we only care about nodes that are NOT in self.nodes.
            if node.id not in self.nodes:
                self.nx_graph.remove_node(node.id)

            raise ValueError(f"Adding node {node.id} would create a cycle.")

        # Commit
        self.nodes[node.id] = node
        # Ensure the node is in the nx_graph if it has no parents
        if not node.causal_parents:
            self.nx_graph.add_node(node.id)

    def get_node(self, node_id: str) -> InteractionNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
        return self.nodes[node_id]

    def get_roots(self) -> List[str]:
        """Returns IDs of nodes with no causal parents."""
        return [node_id for node_id, node in self.nodes.items() if not node.causal_parents]

    def get_descendants(self, node_id: str) -> Set[str]:
        """Returns all node IDs that are descendants of the given node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
        return nx.descendants(self.nx_graph, node_id)

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Returns all node IDs that are ancestors of the given node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
        return nx.ancestors(self.nx_graph, node_id)
