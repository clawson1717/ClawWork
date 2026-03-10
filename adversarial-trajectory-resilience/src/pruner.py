from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import enum

@dataclass
class PruningStatistics:
    cycles_removed: int = 0
    dead_ends_pruned: int = 0
    high_failure_pruned: int = 0
    nodes_remaining: int = 0

class PruningAggressiveness(enum.Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"

class TrajectoryPruner:
    """
    Prunes TrajectoryGraph based on failure modes, cycles, and dead ends.
    """
    def __init__(self, detector=None, aggressiveness=PruningAggressiveness.CONSERVATIVE):
        self.detector = detector
        self.aggressiveness = aggressiveness
        self.failure_threshold = 0.7 if aggressiveness == PruningAggressiveness.CONSERVATIVE else 0.4

    def prune(self, graph) -> PruningStatistics:
        stats = PruningStatistics()
        
        # 1. Prune cycles (if graph implementation doesn't prevent them)
        stats.cycles_removed = self._prune_cycles(graph)
        
        # 2. Prune high failure mode branches
        if self.detector:
            stats.high_failure_pruned = self._prune_high_failure(graph)
            
        # 3. Prune dead ends (nodes with no children that aren't terminal/leaves)
        stats.dead_ends_pruned = self._prune_dead_ends(graph)
        
        stats.nodes_remaining = len(graph.nodes)
        return stats

    def _prune_cycles(self, graph) -> int:
        removed = 0
        
        # In a trajectory graph, there might be multiple components.
        # Check from all nodes to be sure.
        for start_id in list(graph.nodes.keys()):
            if start_id not in graph.nodes: continue
            
            stack = [(start_id, [start_id])]
            while stack:
                curr_id, path = stack.pop()
                node = graph.get_node(curr_id)
                if not node: continue
                
                for child_id in list(node.children_ids):
                    if child_id in path:
                        # Cycle found!
                        node.children_ids.remove(child_id)
                        removed += 1
                    else:
                        stack.append((child_id, path + [child_id]))
                        
        return removed

    def _prune_high_failure(self, graph) -> int:
        pruned_count = 0
        nodes_to_prune = []
        
        for node_id, node in graph.nodes.items():
            content = getattr(node, 'content', getattr(node, 'text', ""))
            if not content: continue
            
            results = self.detector.detect_all(content)
            max_score = max((res.score for res in results.values()), default=0.0)
            
            if max_score > self.failure_threshold:
                nodes_to_prune.append(node_id)
        
        for node_id in nodes_to_prune:
            if node_id in graph.nodes:
                graph.prune_branch(node_id)
                pruned_count += 1
                
        return pruned_count

    def _prune_dead_ends(self, graph) -> int:
        pruned_count = 0
        from src.node import NodeStatus
        
        nodes_to_remove = []
        for node_id, node in graph.nodes.items():
            # Only prune nodes that are still PENDING
            if node.status != NodeStatus.PENDING:
                continue
                
            is_terminal = node.status in [NodeStatus.SUCCESS, NodeStatus.VERIFIED]
            has_children = len(getattr(node, 'children_ids', [])) > 0
            
            if not has_children and not is_terminal:
                nodes_to_remove.append(node_id)
                
        for node_id in nodes_to_remove:
            if node_id in graph.nodes:
                node = graph.get_node(node_id)
                node.status = NodeStatus.PRUNED
                pruned_count += 1
                    
        return pruned_count
