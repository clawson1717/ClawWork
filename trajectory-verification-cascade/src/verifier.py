from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.node import TrajectoryNode, ChecklistItem, NodeStatus

@dataclass
class VerificationResult:
    """
    Result of a verification run on a node.
    """
    node_id: str
    items: List[ChecklistItem] = field(default_factory=list)
    overall_status: NodeStatus = NodeStatus.PENDING

class ChecklistVerifier:
    """
    Handles verification of nodes against a set of checklist criteria.
    Supports custom templates per node type.
    """
    def __init__(self, templates: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the verifier with optional templates.
        
        Args:
            templates: A dictionary mapping node types to lists of criteria strings.
        """
        self.templates = templates or {}

    def add_template(self, node_type: str, criteria: List[str]):
        """
        Adds or updates a template for a specific node type.
        """
        self.templates[node_type] = criteria

    def verify(self, node: TrajectoryNode, node_type: str, 
               evaluator: Any = None) -> VerificationResult:
        """
        Evaluates a node's content against the criteria defined for its node type.
        
        Args:
            node: The TrajectoryNode to verify.
            node_type: The type of the node (to select the appropriate template).
            evaluator: An optional object with an `evaluate(content, criterion)` method 
                      that returns (passed: bool, evidence: str).
                      If None, a default failing evaluation is used.
                      
        Returns:
            VerificationResult containing the checklist items and the overall status.
        """
        criteria = self.templates.get(node_type, [])
        checklist_items = []
        
        # If no criteria are defined for this type, consider it verified by default
        if not criteria:
            node.status = NodeStatus.VERIFIED
            node.checklist_items = []
            return VerificationResult(
                node_id=node.id,
                items=[],
                overall_status=NodeStatus.VERIFIED
            )

        for criterion in criteria:
            if evaluator and hasattr(evaluator, "evaluate"):
                passed, evidence = evaluator.evaluate(node.content, criterion)
            else:
                passed, evidence = False, "No evaluation performed"
            
            item = ChecklistItem(
                criterion=criterion,
                passed=passed,
                evidence=evidence
            )
            checklist_items.append(item)

        # Update node status based on whether all criteria passed
        all_passed = all(item.passed for item in checklist_items)
        status = NodeStatus.VERIFIED if all_passed else NodeStatus.FAILED
        
        # Integrity with existing models
        node.checklist_items = checklist_items
        node.status = status
        
        return VerificationResult(
            node_id=node.id,
            items=checklist_items,
            overall_status=status
        )
