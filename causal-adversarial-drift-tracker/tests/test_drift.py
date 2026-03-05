import pytest
import numpy as np
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.drift import DriftCalculator

@pytest.fixture
def tracker():
    return LiveDriftTracker()

@pytest.fixture
def calculator(tracker):
    return DriftCalculator(tracker)

def test_cosine_distance_identical():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    dist = DriftCalculator.cosine_distance(v1, v2)
    assert pytest.approx(dist) == 0.0

def test_cosine_distance_orthogonal():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    dist = DriftCalculator.cosine_distance(v1, v2)
    assert pytest.approx(dist) == 1.0

def test_cosine_distance_opposite():
    v1 = [1.0, 0.0, 0.0]
    v2 = [-1.0, 0.0, 0.0]
    dist = DriftCalculator.cosine_distance(v1, v2)
    assert pytest.approx(dist) == 2.0

def test_calculate_node_drift(tracker, calculator):
    root_payload = ReasoningPayload(
        source_id="root",
        content="Primary intent: Sell apples.",
        semantic_vector=[1.0, 0.0]
    )
    node_payload = ReasoningPayload(
        source_id="step1",
        content="Step 1: Find apple orchid.",
        semantic_vector=[0.707, 0.707] # 45 degrees
    )
    
    tracker.add_reasoning_node(root_payload)
    tracker.add_reasoning_node(node_payload, parent_ids=["root"])
    
    drift = calculator.calculate_node_drift("step1", "root")
    expected_drift = 1.0 - (0.707 * 1.0 + 0.707 * 0.0)
    assert pytest.approx(drift, rel=1e-3) == expected_drift

def test_calculate_path_drift(tracker, calculator):
    # Setup a chain: root -> n1 -> n2
    p0 = ReasoningPayload(source_id="root", content="root", semantic_vector=[1.0, 0.0])
    p1 = ReasoningPayload(source_id="n1", content="n1", semantic_vector=[0.8, 0.6]) # cos sim 0.8, dist 0.2
    p2 = ReasoningPayload(source_id="n2", content="n2", semantic_vector=[0.0, 1.0]) # cos sim 0.0, dist 1.0
    
    tracker.add_reasoning_node(p0)
    tracker.add_reasoning_node(p1, parent_ids=["root"])
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    
    # Path drift (average of 0, 0.2, 1.0) = 1.2 / 3 = 0.4
    avg_drift = calculator.calculate_path_drift("n2")
    assert pytest.approx(avg_drift) == 0.4

def test_update_tracker_drift_scores(tracker, calculator):
    p0 = ReasoningPayload(source_id="root", content="root", semantic_vector=[1.0, 0.0])
    p1 = ReasoningPayload(source_id="n1", content="n1", semantic_vector=[0.0, 1.0])
    
    tracker.add_reasoning_node(p0)
    tracker.add_reasoning_node(p1, parent_ids=["root"])
    
    calculator.update_tracker_drift_scores()
    
    updated_p0 = tracker.get_payload("root")
    updated_p1 = tracker.get_payload("n1")
    
    assert updated_p0.drift_score == 0.0
    assert pytest.approx(updated_p1.drift_score) == 1.0
