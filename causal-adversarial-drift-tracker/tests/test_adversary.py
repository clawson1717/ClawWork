import pytest
import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from payload import ReasoningPayload
from adversary import AdversarialSenser

def test_adversarial_senser_init():
    """
    Test basic initialization of the AdversarialSenser class.
    """
    senser = AdversarialSenser(threshold=0.7)
    assert senser.threshold == 0.7
    assert "perhaps" in senser.indicators

def test_sense_weak_points_ambiguity():
    """
    Test if the senser can identify ambiguity in reasoning.
    """
    senser = AdversarialSenser()
    payload = ReasoningPayload(
        source_id="test_node",
        content="The user's intent is maybe clear, but perhaps we should assume the opposite."
    )
    weak_points = senser.sense_weak_points(payload)
    
    # Check for specific indicators ("maybe", "perhaps", "assume")
    types = [wp['type'] for wp in weak_points]
    markers = [wp.get('marker') for wp in weak_points]
    
    assert "ambiguity" in types
    assert "maybe" in markers
    assert "perhaps" in markers
    assert "assume" in markers

def test_sense_weak_points_drift_vulnerability():
    """
    Test if high drift scores are flagged as vulnerabilities.
    """
    senser = AdversarialSenser(threshold=0.5)
    payload = ReasoningPayload(
        source_id="drifting_node",
        content="This is standard content with high drift score.",
        drift_score=0.8
    )
    weak_points = senser.sense_weak_points(payload)
    
    types = [wp['type'] for wp in weak_points]
    assert "drift_vulnerability" in types

def test_stress_test_conflicting_hint():
    """
    Test if stress_test can inject a conflicting hint into a payload.
    """
    senser = AdversarialSenser()
    original_payload = ReasoningPayload(
        source_id="original_node",
        content="The market is trending upwards."
    )
    
    stressed_payload = senser.stress_test(original_payload, mode="conflicting_hint")
    
    assert "ADVERSARIAL HINT" in stressed_payload.content
    assert stressed_payload.source_id == "adversary_original_node"
    assert stressed_payload.drift_score > original_payload.drift_score

def test_stress_test_noise():
    """
    Test if the noise mode injects randomized noise.
    """
    senser = AdversarialSenser()
    original_payload = ReasoningPayload(
        source_id="original_node",
        content="The market is trending upwards."
    )
    
    stressed_payload = senser.stress_test(original_payload, mode="noise")
    assert len(stressed_payload.content) > len(original_payload.content)
    assert any(c in "!@#$%^&*" for c in stressed_payload.content)

def test_vulnerability_index():
    """
    Test the aggregate vulnerability index calculation.
    """
    senser = AdversarialSenser()
    
    payloads = [
        ReasoningPayload(source_id="n1", content="Solid point."),
        ReasoningPayload(source_id="n2", content="Maybe it's wrong."),
        ReasoningPayload(source_id="n3", content="Assume it's true.")
    ]
    
    v_index = senser.calculate_vulnerability_index(payloads)
    assert 0.0 <= v_index <= 1.0
    # Both "maybe" and "assume" should add severity
    assert v_index > 0.0
