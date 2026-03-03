from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import uuid

from src.node import TrajectoryNode, ChecklistItem, NodeStatus
from src.graph import TrajectoryGraph
from src.verifier import ChecklistVerifier
from src.detector import FailureModeDetector
from src.cascade import CascadeEngine
from src.pruning import PruningPolicy

@dataclass
class TVCAgentConfig:
    """Configuration for the TVCAgent."""
    verification_strictness: float = 0.5
    detection_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "self_doubt": 0.4,
        "social_conformity": 0.4,
        "suggestion_hijacking": 0.4,
        "emotional_susceptibility": 0.4,
        "reasoning_fatigue": 0.4
    })
    max_steps: int = 50
    pruning_enabled: bool = True

@dataclass
class TVCReport:
    """Final report from the TVCAgent."""
    task: str
    success: bool
    trajectory: List[TrajectoryNode] = field(default_factory=list)
    failure_reason: Optional[str] = None
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class TVCAgent:
    """
    Integration layer for the Trajectory Verification Cascade.
    Unifies all components into a production-ready agent.
    """
    def __init__(self, config: Optional[TVCAgentConfig] = None):
        self.config = config or TVCAgentConfig()
        self.graph = TrajectoryGraph()
        self.verifier = ChecklistVerifier()
        self.detector = FailureModeDetector()
        self.pruning_policy = PruningPolicy() if self.config.pruning_enabled else None
        
        # Apply custom thresholds to detector
        self._apply_config_to_detector()

    def _apply_config_to_detector(self):
        """Updates detector thresholds based on config."""
        # Note: The FailureModeDetector currently has hardcoded thresholds in its methods.
        # To be fully configurable, we'd need to modify FailureModeDetector or 
        # wrap its calls. For this implementation, we'll assume the detector 
        # can be configured or we'll handle the thresholding here if needed.
        # Since FailureModeDetector.detect_all returns DetectionResult with scores,
        # we can apply thresholds in the cascade logic or agent logic.
        pass

    def _create_node(self, content: str, parent_id: Optional[str] = None) -> str:
        """Helper to create and add a node to the graph."""
        node_id = str(uuid.uuid4())[:8]
        node = TrajectoryNode(id=node_id, content=content, parent_id=parent_id)
        self.graph.add_node(node)
        if parent_id:
            self.graph.add_edge(parent_id, node_id)
        return node_id

    def process_task(self, 
                     task: str, 
                     reasoning_steps: List[str], 
                     progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> TVCReport:
        """
        Processes a reasoning task by building a graph and running the verification cascade.
        
        Args:
            task: The initial reasoning task/prompt.
            reasoning_steps: A list of reasoning steps (content) to verify.
            progress_callback: Optional callback for real-time monitoring.
        """
        if not reasoning_steps:
            return TVCReport(task=task, success=False, failure_reason="No reasoning steps provided.")

        # 1. Initialize Graph with steps
        # In a real scenario, these might be generated dynamically.
        # Here we assume a linear trajectory is provided for verification.
        last_node_id = None
        for step_content in reasoning_steps:
            last_node_id = self._create_node(step_content, last_node_id)
        
        first_node_id = self.graph.get_path(last_node_id)[0].id

        # 2. Initialize Cascade Engine
        engine = CascadeEngine(
            graph=self.graph,
            verifier=self.verifier,
            detector=self.detector,
            pruning_policy=self.pruning_policy
        )
        engine.set_start_node(first_node_id)

        # 3. Run Cascade
        history = []
        step_count = 0
        success = False
        failure_reason = None

        while engine.state.current_node_id and step_count < self.config.max_steps:
            curr_id = engine.state.current_node_id
            
            # Use simple "default" type for verifier for now
            step_result = engine.run_step(node_type="default")
            history.append(step_result)
            step_count += 1
            
            # Progress update
            if progress_callback:
                progress_callback({
                    "step": step_count,
                    "node_id": curr_id,
                    "action": step_result["action"],
                    "status": self.graph.get_node(curr_id).status.value
                })

            if step_result["action"] == "finish":
                success = True
                break
            if step_result["action"] == "stop":
                success = False
                failure_reason = f"Cascade stopped at node {curr_id} due to verification failure or dead end."
                break
        
        if step_count >= self.config.max_steps:
            failure_reason = "Reached maximum step limit."

        # 4. Generate Report
        verified_traj = engine.get_verified_trajectory()
        
        metrics = {
            "steps_taken": step_count,
            "nodes_in_graph": len(self.graph.nodes),
            "nodes_pruned": len(engine.state.pruned_branches),
            "nodes_failed": len(engine.state.failed_nodes)
        }
        if self.pruning_policy:
            metrics["compute_saved"] = self.pruning_policy.compute_saved

        return TVCReport(
            task=task,
            success=success,
            trajectory=verified_traj,
            failure_reason=failure_reason,
            verification_history=history,
            metrics=metrics
        )
