import pytest
import os
import sys
from unittest.mock import MagicMock
# Add workspace to sys.path to allow correct import of src-packaged modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.payload import ReasoningPayload
from src.agent import CADTraceAgent

def test_agent_init():
    agent = CADTraceAgent()
    assert agent is not None
    assert agent.tracker.dig.number_of_nodes() == 0
    assert agent.auto_heal is True

def test_process_interaction_healthy():
    agent = CADTraceAgent()
    
    # Healthy Payload
    payload = ReasoningPayload(
        source_id="root_node",
        content="Ground truth reasoning and intent for analysis.",
        semantic_vector=[1.0, 0.0, 0.0]
    )
    
    result = agent.process_interaction(payload)
    
    assert result["node_id"] == "root_node"
    assert result["resilience_score"] > 0.9 # Should be very high for identical
    assert result["system_status"] == "healthy"
    assert agent.tracker.dig.number_of_nodes() == 1

def test_process_interaction_with_drift():
    # Set resilience_threshold very high so it triggers easily
    agent = CADTraceAgent(resilience_threshold=0.1, auto_heal=False)
    
    # Healthy Root
    root_payload = ReasoningPayload(
        source_id="root",
        content="Original Intent",
        semantic_vector=[1.0, 0.0, 0.0]
    )
    agent.process_interaction(root_payload)
    
    # Drifting Node
    drift_payload = ReasoningPayload(
        source_id="drifting_node",
        content="Drifting content with maybe some uncertainty.",
        semantic_vector=[-1.0, 0.0, 0.0], # Opposite vector (max drift)
        drift_score=0.9
    )
    
    result = agent.process_interaction(drift_payload, parent_ids=["root"])
    
    assert result["node_id"] == "drifting_node"
    assert result["resilience_score"] < 0.5
    assert result["system_status"] == "drifting"

def test_auto_healing():
    agent = CADTraceAgent(resilience_threshold=0.3, auto_heal=True)
    
    # Root Intent Node
    root_payload = ReasoningPayload(
        source_id="root",
        content="Initial Plan",
        semantic_vector=[1.0, 0.0, 0.0]
    )
    agent.process_interaction(root_payload)
    
    # Drifting Node trigger
    drift_payload = ReasoningPayload(
        source_id="drift",
        content="Maybe it doesn't matter much anyway.",
        semantic_vector=[0.0, 1.0, 0.0], # 90 degrees (1.0 cosine dist)
        drift_score=0.9
    )
    
    # Mock regenerator_fn
    mock_regenerator = MagicMock(return_value=ReasoningPayload(
        source_id="healed_replacement",
        content="Healed path: Returning to original intent.",
        semantic_vector=[1.0, 0.1, 0.0]
    ))
    
    result = agent.process_interaction(
        drift_payload, 
        parent_ids=["root"], 
        root_intent="Initial Plan",
        regenerator_fn=mock_regenerator
    )
    
    assert result["healing_performed"] is True
    assert "healed_node_id" in str(result["healing_results"]) or "healed_replacement" in str(result["healing_results"])
    assert result["system_status"] == "healed"
    assert agent.tracker.dig.has_node("healed_replacement")
    assert not agent.tracker.dig.has_node("drift")

def test_get_system_summary():
    agent = CADTraceAgent()
    payload = ReasoningPayload(source_id="node1", content="content1", semantic_vector=[1,0])
    agent.process_interaction(payload)
    
    summary = agent.get_system_summary()
    assert summary["total_nodes"] == 1
    assert "avg_drift" in summary
    assert "vulnerability_index" in summary
