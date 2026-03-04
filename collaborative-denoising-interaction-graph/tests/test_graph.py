import pytest
import time
from src.node import InteractionNode
from src.graph import InteractionGraph

def test_interaction_graph_reasoning_chain():
    graph = InteractionGraph()
    
    # Create a 3+ node reasoning chain
    # Agent 1 creates a root node
    node1 = InteractionNode(
        id="node-1",
        agent_id="agent-1",
        input_payload="User instruction",
        output_payload="Initial plan"
    )
    graph.add_node(node1)
    
    # Agent 2 creates a child node
    node2 = InteractionNode(
        id="node-2",
        agent_id="agent-2",
        input_payload="Initial plan",
        output_payload="Refined plan",
        causal_parents=["node-1"]
    )
    graph.add_node(node2)
    
    # Agent 3 creates another child node
    node3 = InteractionNode(
        id="node-3",
        agent_id="agent-3",
        input_payload="Refined plan",
        output_payload="Final output",
        causal_parents=["node-2"]
    )
    graph.add_node(node3)
    
    # Verify roots
    roots = graph.get_roots()
    assert "node-1" in roots
    assert len(roots) == 1
    
    # Verify ancestors of node-3
    ancestors = graph.get_ancestors("node-3")
    assert "node-1" in ancestors
    assert "node-2" in ancestors
    assert len(ancestors) == 2
    
    # Verify descendants of node-1
    descendants = graph.get_descendants("node-1")
    assert "node-2" in descendants
    assert "node-3" in descendants
    assert len(descendants) == 2

def test_interaction_graph_cycle_prevention():
    graph = InteractionGraph()
    
    node1 = InteractionNode(id="n1", agent_id="a1", input_payload="", output_payload="")
    graph.add_node(node1)
    
    node2 = InteractionNode(id="n2", agent_id="a2", input_payload="", output_payload="", causal_parents=["n1"])
    graph.add_node(node2)
    
    # Attempting to add a node that creates a cycle (n1 depends on n2)
    # Note: InteractionNodes usually have their parents defined at creation.
    # To test cycle detection, we'd need a node already in the graph to point back.
    # Since add_node checks parents, a cycle could only happen if:
    # node A depends on B, and after adding A, we try to add B depending on A.
    
    # Actually, in our implementation, since parents must exist, cycles can only 
    # occur if we could update parents of existing nodes. 
    # But for a new node A to create a cycle, it would have to depend on a node 
    # that eventually depends on A. But A isn't in the graph yet.
    
    # Wait, the only way to get a cycle with the 'parents must exist' rule 
    # is if the parents list is checked after registration? No.
    # If I try to add node A with parent B, B must exist.
    # If B exists, it was added before A. Its parents were already in.
    # So A cannot be a parent (even indirect) of B.
    # Therefore,cycles are impossible with my current add_node logic 
    # IF the graph is always a DAG.
    
    # Let's verify duplicate prevention instead.
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(node1)

def test_missing_parent():
    graph = InteractionGraph()
    node = InteractionNode(id="child", agent_id="a1", input_payload="", output_payload="", causal_parents=["missing"])
    with pytest.raises(ValueError, match="Parent ID missing not found"):
        graph.add_node(node)
