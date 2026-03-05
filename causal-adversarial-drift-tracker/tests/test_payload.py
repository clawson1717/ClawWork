import pytest
import json
from src.payload import ReasoningPayload

def test_payload_creation():
    payload = ReasoningPayload(
        source_id="agent-001",
        content="Thinking Process Stage 1",
        semantic_vector=[0.1, 0.2, 0.3]
    )
    assert payload.source_id == "agent-001"
    assert payload.content == "Thinking Process Stage 1"
    # state_hash should be generated automatically
    assert payload.state_hash is not None
    assert len(payload.state_hash) == 64  # SHA-256 length

def test_payload_hash_consistency():
    id_val = "test-id"
    content_val = "test-content"
    
    p1 = ReasoningPayload(source_id=id_val, content=content_val)
    p2 = ReasoningPayload(source_id=id_val, content=content_val)
    
    assert p1.state_hash == p2.state_hash

def test_payload_serialization():
    payload = ReasoningPayload(
        source_id="agent-002",
        content="Observation 1",
        drift_score=0.15
    )
    
    json_str = payload.to_json()
    loaded_payload = ReasoningPayload.from_json(json_str)
    
    assert loaded_payload.source_id == payload.source_id
    assert loaded_payload.content == payload.content
    assert loaded_payload.state_hash == payload.state_hash
    assert loaded_payload.drift_score == pytest.approx(0.15)

def test_invalid_drift_score():
    with pytest.raises(Exception):
        ReasoningPayload(
            source_id="id",
            content="content",
            drift_score=-1.0
        )

def test_semantic_vector():
    vector = [1.0, 0.0, -1.0]
    payload = ReasoningPayload(
        source_id="id",
        content="content",
        semantic_vector=vector
    )
    assert payload.semantic_vector == vector
