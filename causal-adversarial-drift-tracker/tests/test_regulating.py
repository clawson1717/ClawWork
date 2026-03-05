import pytest
import numpy as np
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.sensing import UncertaintySenser
from src.drift import DriftCalculator
from src.regulating import TruthRegulator

def setup_regulator_suite():
    tracker = LiveDriftTracker()
    drift_calc = DriftCalculator(tracker)
    senser = UncertaintySenser(tracker)
    regulator = TruthRegulator(tracker, senser, drift_calc, resilience_threshold=0.6)
    return tracker, drift_calc, senser, regulator

def test_truth_regulator_init():
    tracker, drift_calc, senser, regulator = setup_regulator_suite()
    assert regulator.tracker == tracker
    assert regulator.senser == senser
    assert regulator.drift_calc == drift_calc
    assert regulator.resilience_threshold == 0.6

def test_calculate_resilience_score():
    tracker, drift_calc, senser, regulator = setup_regulator_suite()
    
    # Healthy node
    p_healthy = ReasoningPayload(
        source_id="healthy", 
        content="Solid reasoning step.",
        drift_score=0.1,
        semantic_vector=[1.0, 0.0, 0.0]
    )
    tracker.add_reasoning_node(p_healthy)
    
    # Drifting node
    p_drifting = ReasoningPayload(
        source_id="drifting", 
        content="Maybe we should consider something else entirely, perhaps.",
        drift_score=0.8,
        semantic_vector=[0.0, 1.0, 0.0]
    )
    tracker.add_reasoning_node(p_drifting)
    
    score_healthy = regulator.calculate_resilience_score("healthy")
    score_drifting = regulator.calculate_resilience_score("drifting")
    
    assert score_healthy > score_drifting
    assert score_healthy > 0.8
    assert score_drifting < 0.4

def test_pinpoint_drift_origin():
    tracker, drift_calc, senser, regulator = setup_regulator_suite()
    
    # Node 1: Root Intent (Healthy)
    p1 = ReasoningPayload(source_id="n1", content="Target: Solve X.", drift_score=0.0)
    tracker.add_reasoning_node(p1)
    
    # Node 2: Reasoning Step 1 (Healthy)
    p2 = ReasoningPayload(source_id="n2", content="Analyzing X component A.", drift_score=0.1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    
    # Node 3: Reasoning Step 2 (Drifting Origin)
    p3 = ReasoningPayload(source_id="n3", content="Maybe Y is also interesting, perhaps we should look at Y.", drift_score=0.7)
    tracker.add_reasoning_node(p3, parent_ids=["n2"])
    
    # Node 4: Reasoning Step 3 (Affected descendant)
    p4 = ReasoningPayload(source_id="n4", content="Exploring Y deep dive.", drift_score=0.9)
    tracker.add_reasoning_node(p4, parent_ids=["n3"])
    
    origin = regulator.pinpoint_drift_origin()
    assert origin == "n3"

def test_get_truth_report():
    tracker, drift_calc, senser, regulator = setup_regulator_suite()
    
    p1 = ReasoningPayload(source_id="n1", content="Root", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Maybe drifting badly, perhaps.", drift_score=0.9)
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    
    report = regulator.get_truth_report()
    assert report["status"] == "drifting"
    assert report["total_nodes"] == 2
    assert len(report["drifting_nodes"]) == 1
    assert report["drift_origin"] == "n2"
    assert "n1" in report["resilience_scores"]
    assert "n2" in report["resilience_scores"]

def test_identify_denoise_targets():
    tracker, drift_calc, senser, regulator = setup_regulator_suite()
    
    # Healthy -> Drifting Origin -> Drifting Child -> Healthy Leaf (re-stabilized)
    p1 = ReasoningPayload(source_id="n1", content="Root", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Maybe drift.", drift_score=0.8)
    p3 = ReasoningPayload(source_id="n3", content="Drifting further.", drift_score=0.9)
    p4 = ReasoningPayload(source_id="n4", content="Stabilized fact.", drift_score=0.1)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    tracker.add_reasoning_node(p3, parent_ids=["n2"])
    tracker.add_reasoning_node(p4, parent_ids=["n3"])
    
    targets = regulator.identify_denoise_targets()
    assert "n2" in targets
    assert "n3" in targets
    assert "n1" not in targets
    assert "n4" not in targets
