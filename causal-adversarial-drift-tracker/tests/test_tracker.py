import pytest
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.adversary import AdversarialSenser

def test_tracker_add_node():
    tracker = LiveDriftTracker()
    payload = ReasoningPayload(source_id="node1", content="Init reasoning", drift_score=0.1)
    
    node_id = tracker.add_reasoning_node(payload)
    assert node_id == "node1"
    assert tracker.dig.number_of_nodes() == 1
    assert tracker.get_payload("node1").content == "Init reasoning"

def test_tracker_causal_edges():
    tracker = LiveDriftTracker()
    p1 = ReasoningPayload(source_id="p1", content="Parent 1", drift_score=0.2)
    p2 = ReasoningPayload(source_id="p2", content="Parent 2", drift_score=0.1)
    c1 = ReasoningPayload(source_id="c1", content="Child 1", drift_score=0.3)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2)
    tracker.add_reasoning_node(c1, parent_ids=["p1", "p2"])
    
    assert tracker.dig.number_of_nodes() == 3
    assert tracker.dig.number_of_edges() == 2
    assert "p1" in tracker.dig.predecessors("c1")
    assert "p2" in tracker.dig.predecessors("c1")

def test_tracker_causal_path():
    tracker = LiveDriftTracker()
    p1 = ReasoningPayload(source_id="p1", content="Root", drift_score=0.1)
    p2 = ReasoningPayload(source_id="p2", content="Step 2", drift_score=0.2)
    p3 = ReasoningPayload(source_id="p3", content="Goal", drift_score=0.3)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["p1"])
    tracker.add_reasoning_node(p3, parent_ids=["p2"])
    
    path = tracker.get_causal_path("p3")
    assert path == ["p1", "p2", "p3"]

def test_tracker_metrics():
    tracker = LiveDriftTracker()
    tracker.add_reasoning_node(ReasoningPayload(source_id="n1", content="C1", drift_score=0.2))
    tracker.add_reasoning_node(ReasoningPayload(source_id="n2", content="C2", drift_score=0.4))
    
    metrics = tracker.get_drift_metrics()
    assert metrics["avg_drift"] == pytest.approx(0.3)
    assert metrics["max_drift"] == 0.4
    assert metrics["node_count"] == 2

def test_tracker_export():
    tracker = LiveDriftTracker()
    tracker.add_reasoning_node(ReasoningPayload(source_id="n1", content="A", drift_score=0.1))
    tracker.add_reasoning_node(ReasoningPayload(source_id="n2", content="B", drift_score=0.2), parent_ids=["n1"])
    
    dot = tracker.export_to_dot()
    assert 'digraph DIG {' in dot
    assert '"n1"' in dot
    assert '"n1" -> "n2"' in dot
    
    adj = tracker.to_adjacency_list()
    assert adj["n1"] == ["n2"]
    assert adj["n2"] == []

def test_integration_with_adversary():
    # Integrate Step 3 (Adversary) with Step 4 (Tracker)
    tracker = LiveDriftTracker()
    senser = AdversarialSenser(threshold=0.3)
    
    original = ReasoningPayload(source_id="original", content="Trustworthy premise.", drift_score=0.0)
    tracker.add_reasoning_node(original)
    
    # Stress test the original and add the resulting 'drifted' payload
    stressed = senser.stress_test(original, mode="conflicting_hint")
    tracker.add_reasoning_node(stressed, parent_ids=["original"])
    
    assert tracker.dig.number_of_nodes() == 2
    assert tracker.dig.number_of_edges() == 1
    
    metrics = tracker.get_drift_metrics()
    assert metrics["avg_drift"] > 0.0
    assert tracker.dig.has_edge("original", "adversary_original")
