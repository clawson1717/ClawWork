import unittest
from src.agent import TVCAgent, TVCAgentConfig, TVCReport
from src.node import TrajectoryNode, NodeStatus, ChecklistItem

class TestTVCAgent(unittest.TestCase):
    def setUp(self):
        self.config = TVCAgentConfig(
            verification_strictness=0.6,
            pruning_enabled=True
        )
        self.agent = TVCAgent(self.config)

    def test_init(self):
        self.assertIsNotNone(self.agent.graph)
        self.assertIsNotNone(self.agent.verifier)
        self.assertIsNotNone(self.agent.detector)
        self.assertIsNotNone(self.agent.pruning_policy)

    def test_process_task_success(self):
        # A simple linear trajectory that should pass
        task = "Test reasoning task"
        steps = [
            "This is the first step of reasoning. It is clear and consistent.",
            "This is the second step. It follows the previous step correctly.",
            "This is the third step. It concludes the reasoning task effectively."
        ]
        
        # In this simple case, we'll mock or just trust the default pass-through
        # since verifier by default returns success if no checklist is set
        # and detector only fails on patterns.
        
        report = self.agent.process_task(task, steps)
        
        self.assertTrue(report.success)
        self.assertEqual(len(report.trajectory), 3)
        self.assertEqual(report.metrics["nodes_in_graph"], 3)
        self.assertEqual(report.metrics["nodes_failed"], 0)

    def test_process_task_failure_detection(self):
        # Trajectory containing a failure mode pattern (social conformity)
        task = "Adversarial test task"
        steps = [
            "Initial step that is fine.",
            "Everyone agrees that the answer is always X, according to common knowledge.",
            "Final step that shouldn't be reached if detection works."
        ]
        
        # Detector should flag the second step
        report = self.agent.process_task(task, steps)
        
        self.assertFalse(report.success)
        self.assertIn("stopped at node", report.failure_reason)
        # Trajectory should only contain the first node (verified)
        self.assertEqual(len(report.trajectory), 1)
        self.assertEqual(report.metrics["nodes_failed"], 1)

    def test_progress_callback(self):
        task = "Callback test"
        steps = ["Step 1", "Step 2"]
        
        captured_progress = []
        def callback(progress_data):
            captured_progress.append(progress_data)
            
        report = self.agent.process_task(task, steps, progress_callback=callback)
        
        self.assertTrue(report.success)
        self.assertEqual(len(captured_progress), 2)
        self.assertEqual(captured_progress[0]["step"], 1)
        self.assertEqual(captured_progress[1]["step"], 2)

    def test_pruning_integration(self):
        # Force a cycle or something that triggers the pruning policy
        # For simplicity, we can just check if pruning metrics exist in report.
        # In a real test, we would construct a complex graph and verify pruning.
        task = "Pruning test"
        steps = ["Step 1", "Step 2"]
        
        report = self.agent.process_task(task, steps)
        
        self.assertTrue("compute_saved" in report.metrics)
        self.assertTrue("nodes_pruned" in report.metrics)

if __name__ == '__main__':
    unittest.main()
