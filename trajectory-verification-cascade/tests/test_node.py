import pytest
from src.node import NodeStatus, ChecklistItem, TrajectoryNode

def test_node_status_values():
    assert NodeStatus.PENDING.value == "PENDING"
    assert NodeStatus.VERIFIED.value == "VERIFIED"
    assert NodeStatus.FAILED.value == "FAILED"
    assert NodeStatus.PRUNED.value == "PRUNED"

def test_checklist_item_creation():
    item = ChecklistItem(criterion="Check", passed=True, evidence="Evidence")
    assert item.criterion == "Check"
    assert item.passed is True
    assert item.evidence == "Evidence"

def test_trajectory_node_minimal():
    node = TrajectoryNode(id="1", content="hello")
    assert node.id == "1"
    assert node.content == "hello"
    assert node.status == NodeStatus.PENDING
    assert node.checklist_items == []
    assert node.parent_id is None
    assert node.children_ids == []

def test_trajectory_node_full_init():
    items = [ChecklistItem(criterion="C1", passed=False, evidence="E1")]
    node = TrajectoryNode(
        id="2",
        content="world",
        checklist_items=items,
        status=NodeStatus.VERIFIED,
        parent_id="1",
        children_ids=["3", "4"]
    )
    assert node.id == "2"
    assert node.status == NodeStatus.VERIFIED
    assert node.parent_id == "1"
    assert node.children_ids == ["3", "4"]
    assert len(node.checklist_items) == 1

def test_node_status_update():
    node = TrajectoryNode(id="1", content="test")
    node.status = NodeStatus.FAILED
    assert node.status == NodeStatus.FAILED
    node.status = NodeStatus.PRUNED
    assert node.status == NodeStatus.PRUNED

def test_adding_checklist_items():
    node = TrajectoryNode(id="1", content="test")
    node.checklist_items.append(ChecklistItem("C1", True, "E1"))
    assert len(node.checklist_items) == 1
    assert node.checklist_items[0].criterion == "C1"

def test_adding_children_ids():
    node = TrajectoryNode(id="1", content="test")
    node.children_ids.append("child-1")
    assert "child-1" in node.children_ids

def test_node_parent_id_assignment():
    child = TrajectoryNode(id="c1", content="child")
    parent = TrajectoryNode(id="p1", content="parent")
    child.parent_id = parent.id
    assert child.parent_id == "p1"

def test_list_factories_are_isolated():
    node1 = TrajectoryNode(id="1", content="a")
    node2 = TrajectoryNode(id="2", content="b")
    node1.children_ids.append("3")
    assert "3" not in node2.children_ids
    node1.checklist_items.append(ChecklistItem("c", True, "v"))
    assert len(node2.checklist_items) == 0

def test_enum_comparison():
    node = TrajectoryNode(id="1", content="a")
    assert node.status == NodeStatus.PENDING
    assert node.status != NodeStatus.VERIFIED

def test_node_with_multiple_children():
    node = TrajectoryNode(id="1", content="a", children_ids=["2", "3", "4"])
    assert len(node.children_ids) == 3
    assert node.children_ids == ["2", "3", "4"]
