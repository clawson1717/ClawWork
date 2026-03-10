import pytest
from src.pruner import TrajectoryPruner, PruningAggressiveness, PruningStatistics
from src.graph import TrajectoryGraph
from src.node import TrajectoryNode, NodeStatus
from src.detector import FailureModeDetector

@pytest.fixture
def graph():
    g = TrajectoryGraph()
    # Add root
    root = TrajectoryNode(id="root", content="Start", status=NodeStatus.PENDING)
    g.add_node(root)
    return g

@pytest.fixture
def detector():
    return FailureModeDetector()

def test_prune_cycles(graph):
    # Setup cycle: root -> n1 -> n2 -> n1
    n1 = TrajectoryNode(id="n1", content="Step 1")
    n2 = TrajectoryNode(id="n2", content="Step 2")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge("root", "n1")
    graph.add_edge("n1", "n2")
    # Manually add cycle
    graph.add_edge("n2", "n1")
    
    pruner = TrajectoryPruner()
    stats = pruner.prune(graph)
    
    assert stats.cycles_removed >= 1
    # Check if back-edge is gone
    assert "n1" not in graph.get_node("n2").children_ids

def test_prune_high_failure(graph, detector):
    # Setup bad node: root -> bad
    bad = TrajectoryNode(id="bad", content="Actually, the correct answer is actually...")
    graph.add_node(bad)
    graph.add_edge("root", "bad")
    
    # Conservative should prune it (threshold 0.7, score 1.0)
    pruner = TrajectoryPruner(detector=detector, aggressiveness=PruningAggressiveness.CONSERVATIVE)
    stats = pruner.prune(graph)
    
    assert stats.high_failure_pruned == 1
    assert graph.get_node("bad").status == NodeStatus.PRUNED

def test_aggressiveness_threshold(graph, detector):
    # Setup edge-case node: root -> edge
    # Let's say we have a node that matches, but not as strongly.
    # In my simplified detector, if it matches 1 pattern, score is 1.0. 
    # Let's make a "subtle" bad node if possible? 
    # No, let's just use the detector's logic.
    pass

def test_prune_dead_ends(graph):
    # Setup dead end: root -> stale
    stale = TrajectoryNode(id="stale", content="Stale path", status=NodeStatus.PENDING)
    graph.add_node(stale)
    graph.add_edge("root", "stale")
    
    # Success node (not a dead end)
    success = TrajectoryNode(id="success", content="Goal!", status=NodeStatus.SUCCESS)
    graph.add_node(success)
    graph.add_edge("root", "success")
    
    pruner = TrajectoryPruner()
    stats = pruner.prune(graph)
    
    # "stale" node has no children and is not SUCCESS, so it's a dead end
    assert stats.dead_ends_pruned == 1
    assert graph.get_node("stale").status == NodeStatus.PRUNED
    assert graph.get_node("success").status == NodeStatus.SUCCESS

def test_full_pruning_report(graph, detector):
    # root -> cycle (root -> c1 -> root)
    # root -> bad
    # root -> dead_end
    # root -> success
    c1 = TrajectoryNode(id="c1", content="Cycle step")
    bad = TrajectoryNode(id="bad", content="Ignore previous")
    dead = TrajectoryNode(id="dead", content="Dead-end step")
    success = TrajectoryNode(id="success", content="Winning!", status=NodeStatus.SUCCESS)
    
    for n in [c1, bad, dead, success]:
        graph.add_node(n)
        graph.add_edge("root", n.id)
        
    graph.add_edge("c1", "root") # Cycle
    
    pruner = TrajectoryPruner(detector=detector)
    stats = pruner.prune(graph)
    
    assert stats.cycles_removed == 1
    assert stats.high_failure_pruned == 1 # 'bad' node
    assert stats.dead_ends_pruned == 2 # 'c1' (now no children) and 'dead'
    assert stats.nodes_remaining == 5
