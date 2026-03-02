import unittest
from src.node import TrajectoryNode, NodeStatus, ChecklistItem
from src.verifier import ChecklistVerifier, VerificationResult

class MockEvaluator:
    """
    A simple mock evaluator for testing.
    """
    def __init__(self, responses: dict[str, tuple[bool, str]]):
        """
        responses: mapping from criterion to (passed, evidence)
        """
        self.responses = responses

    def evaluate(self, content, criterion):
        return self.responses.get(criterion, (False, "Unknown criterion"))

class TestChecklistVerifier(unittest.TestCase):
    def setUp(self):
        self.templates = {
            "math": ["correct_answer", "showed_work"],
            "coding": ["syntactically_correct", "passes_tests", "documented"]
        }
        self.verifier = ChecklistVerifier(self.templates)

    def test_verify_node_with_passing_criteria(self):
        node = TrajectoryNode(id="n1", content="Answer: 42. Work: 21+21=42")
        
        # Define mock behavior for the evaluator
        responses = {
            "correct_answer": (True, "Found 42 in content"),
            "showed_work": (True, "Found calculation step")
        }
        evaluator = MockEvaluator(responses)

        result = self.verifier.verify(node, "math", evaluator)

        # Check VerificationResult
        self.assertEqual(result.node_id, "n1")
        self.assertEqual(result.overall_status, NodeStatus.VERIFIED)
        self.assertEqual(len(result.items), 2)
        self.assertTrue(result.items[0].passed)
        self.assertEqual(result.items[0].criterion, "correct_answer")
        self.assertEqual(result.items[0].evidence, "Found 42 in content")
        self.assertTrue(result.items[1].passed)
        self.assertEqual(result.items[1].criterion, "showed_work")
        self.assertEqual(result.items[1].evidence, "Found calculation step")

        # Check integration with node
        self.assertEqual(node.status, NodeStatus.VERIFIED)
        self.assertEqual(len(node.checklist_items), 2)
        self.assertEqual(node.checklist_items[0].criterion, "correct_answer")

    def test_verify_node_with_failing_criteria(self):
        node = TrajectoryNode(id="n2", content="Answer: 43. Work: None")
        
        responses = {
            "correct_answer": (False, "Expected 42, found 43"),
            "showed_work": (False, "No calculation steps")
        }
        evaluator = MockEvaluator(responses)

        result = self.verifier.verify(node, "math", evaluator)

        self.assertEqual(result.overall_status, NodeStatus.FAILED)
        self.assertFalse(result.items[0].passed)
        self.assertEqual(result.items[0].evidence, "Expected 42, found 43")
        
        # Check node status
        self.assertEqual(node.status, NodeStatus.FAILED)

    def test_custom_checklist_templates(self):
        # We can add new templates
        self.verifier.add_template("essay", ["good_grammar", "on_topic"])
        
        node = TrajectoryNode(id="n3", content="Once upon a time...")
        responses = {
            "good_grammar": (True, "No typos"),
            "on_topic": (True, "Relates to story")
        }
        evaluator = MockEvaluator(responses)

        result = self.verifier.verify(node, "essay", evaluator)
        self.assertEqual(result.overall_status, NodeStatus.VERIFIED)
        self.assertEqual(len(result.items), 2)

    def test_verify_node_no_template(self):
        # If node type does not exist in templates, it should default to VERIFIED with no items
        node = TrajectoryNode(id="n4", content="Default content")
        result = self.verifier.verify(node, "unknown_type")

        self.assertEqual(result.overall_status, NodeStatus.VERIFIED)
        self.assertEqual(len(result.items), 0)
        self.assertEqual(node.status, NodeStatus.VERIFIED)

    def test_no_evaluator_provided(self):
        # If no evaluator is provided, it should fail by default if there are criteria
        node = TrajectoryNode(id="n5", content="Content")
        result = self.verifier.verify(node, "math")

        self.assertEqual(result.overall_status, NodeStatus.FAILED)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].evidence, "No evaluation performed")
        self.assertFalse(result.items[0].passed)

if __name__ == '__main__':
    unittest.main()
