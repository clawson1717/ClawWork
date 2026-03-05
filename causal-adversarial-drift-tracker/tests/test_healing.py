import pytest
import networkx as nx
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.sensing import UncertaintySenser
from src.drift import DriftCalculator
from src.regulating import TruthRegulator
from src.healing import BranchHealer

def setup_healing_suite():
    tracker = LiveDriftTracker()
    drift_calc = DriftCalculator(tracker)
    senser = UncertaintySenser(tracker)
    regulator = TruthRegulator(tracker, senser, drift_calc, resilience_threshold=0.6)
    healer = BranchHealer(regulator)
    return tracker, regulator, healer

def test_prune_branch():
    tracker, regulator, healer = setup_healing_suite()
    
    # Simple chain: n1 -> n2 -> n3
    p1 = ReasoningPayload(source_id="n1", content="Root", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Reasoning", drift_score=0.1)
    p3 = ReasoningPayload(source_id="n3", content="Leaf", drift_score=0.2)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    tracker.add_reasoning_node(p3, parent_ids=["n2"])
    
    # Prune from n2 (should remove n2 and n3)
    removed = healer.prune_branch("n2")
    
    assert "n2" in removed
    assert "n3" in removed
    assert "n1" not in removed
    assert not tracker.dig.has_node("n2")
    assert not tracker.dig.has_node("n3")
    assert tracker.dig.has_node("n1")
    assert tracker.metadata["nodes_count"] == 1

def test_heal_drifting_branches_no_drift():
    tracker, regulator, healer = setup_healing_suite()
    
    p1 = ReasoningPayload(source_id="n1", content="Root", drift_score=0.0)
    tracker.add_reasoning_node(p1)
    
    result = healer.heal_drifting_branches(root_intent="Solve X")
    assert result["status"] == "healthy"
    assert len(result["pruned_nodes"]) == 0
    assert tracker.dig.has_node("n1")

def test_heal_drifting_branches_with_regeneration():
    tracker, regulator, healer = setup_healing_suite()
    
    # Tree: n1 -> n2 (drifting), n1 -> n3 (healthy)
    p1 = ReasoningPayload(source_id="n1", content="Solve Math", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Maybe we should think about art instead, perhaps.", drift_score=0.8)
    p3 = ReasoningPayload(source_id="n3", content="2 + 2 equals 4.", drift_score=0.0)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    tracker.add_reasoning_node(p3, parent_ids=["n1"])
    
    # Verify drift origin is n2
    assert regulator.pinpoint_drift_origin() == "n2"
    
    # Mock regenerator function
    def mock_regenerator(intent, parents):
        return ReasoningPayload(
            source_id="n2_healed", 
            content="Solving the actual problem: " + intent,
            drift_score=0.1
        )
    
    result = healer.heal_drifting_branches(
        root_intent="Solve Math", 
        regenerator_fn=mock_regenerator
    )
    
    assert result["status"] == "healed"
    assert "n2" in result["pruned_nodes"]
    assert "n2_healed" in result["healed_nodes"]
    
    assert not tracker.dig.has_node("n2")
    assert tracker.dig.has_node("n1")
    assert tracker.dig.has_node("n3")
    assert tracker.dig.has_node("n2_healed")
    
    # Verify new node is connected to parent
    assert tracker.dig.has_edge("n1", "n2_healed")

def test_verify_consistency():
    tracker, regulator, healer = setup_healing_suite()
    
    # Create healthy graph
    p1 = ReasoningPayload(source_id="n1", content="Root", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Reasoning", drift_score=0.1)
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    
    assert healer.verify_consistency() is True
    
    # Introduce cycle manually (should not happen via API, but good for test)
    tracker.dig.add_edge("n2", "n1")
    assert healer.verify_consistency() is False
