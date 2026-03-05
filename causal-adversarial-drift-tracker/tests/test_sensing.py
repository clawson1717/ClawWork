import pytest
import numpy as np
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.sensing import UncertaintySenser

def test_uncertainty_senser_init():
    tracker = LiveDriftTracker()
    senser = UncertaintySenser(tracker, alpha=0.4, propagation_decay=0.5)
    assert senser.tracker == tracker
    assert senser.alpha == 0.4
    assert senser.propagation_decay == 0.5

def test_score_ambiguity():
    tracker = LiveDriftTracker()
    senser = UncertaintySenser(tracker)
    
    # Low ambiguity
    p1 = ReasoningPayload(source_id="agent1", content="This is a definitive statement with no uncertainty.")
    score1 = senser.score_ambiguity(p1)
    assert score1 == 0.0
    
    # High ambiguity (trigger words)
    p2 = ReasoningPayload(source_id="agent2", content="Maybe it is because we could be assuming that perhaps the drift is potentially occurring.")
    score2 = senser.score_ambiguity(p2)
    assert score2 > score1
    assert score2 <= 1.0

def test_calculate_node_uncertainty():
    tracker = LiveDriftTracker()
    senser = UncertaintySenser(tracker, alpha=0.5)
    
    p = ReasoningPayload(source_id="node1", content="Maybe so.", drift_score=0.8)
    # Using alpha=0.5, combined uncertainty is 0.5 * ambiguity + 0.5 * drift
    tracker.add_reasoning_node(p)
    ambiguity = senser.score_ambiguity(p)
    expected = (0.5 * ambiguity) + (0.5 * 0.8)
    
    uncertainty = senser.calculate_node_uncertainty("node1")
    assert pytest.approx(uncertainty) == expected

def test_uncertainty_flow_propagation():
    tracker = LiveDriftTracker()
    # High decay to keep flow manageable but visible
    senser = UncertaintySenser(tracker, alpha=1.0, propagation_decay=0.9) # alpha 1.0 ignores drift, only ambiguity
    
    # Root node: Certain
    p_root = ReasoningPayload(source_id="root", content="Final truth established.")
    tracker.add_reasoning_node(p_root)
    
    # Child node: Uncertain
    p_child = ReasoningPayload(source_id="child", content="Maybe this is correct.")
    tracker.add_reasoning_node(p_child, parent_ids=["root"])
    
    # Grandchild node: Certain (but should inherit uncertainty)
    p_gchild = ReasoningPayload(source_id="gchild", content="I agree.")
    tracker.add_reasoning_node(p_gchild, parent_ids=["child"])
    
    # Calculate flows
    root_flow = senser.get_uncertainty_flow("root")
    child_flow = senser.get_uncertainty_flow("child")
    gchild_flow = senser.get_uncertainty_flow("gchild")
    
    # Root flow is just its own ambiguity (0.0)
    assert root_flow == 0.0
    
    # Child flow is its ambiguity (> 0.0)
    assert child_flow > 0.0
    
    # Grandchild flow is its ambiguity (0.0) + child_flow * 0.9
    assert gchild_flow > 0.0
    assert gchild_flow == pytest.approx(child_flow * 0.9)

def test_sense_drift_hazards():
    tracker = LiveDriftTracker()
    senser = UncertaintySenser(tracker)
    
    p1 = ReasoningPayload(source_id="safe", content="This is solid fact.")
    p2 = ReasoningPayload(source_id="hazard", content="Maybe we should assume it is potentially broken.", drift_score=0.9)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2)
    
    hazards = senser.sense_drift_hazards(threshold=0.5)
    assert len(hazards) == 1
    assert hazards[0]["node_id"] == "hazard"
    assert hazards[0]["uncertainty_flow"] > 0.5
