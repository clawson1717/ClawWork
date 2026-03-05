import datetime
from typing import List, Dict, Any, Optional, Callable
from .payload import ReasoningPayload
from .tracker import LiveDriftTracker
from .adversary import AdversarialSenser
from .drift import DriftCalculator
from .sensing import UncertaintySenser
from .regulating import TruthRegulator
from .healing import BranchHealer

class CADTraceAgent:
    """
    The CAD-TRACE Monitor Agent.
    Unified interface for tracking, sensing, and healing causal reasoning drift.
    """

    def __init__(
        self,
        resilience_threshold: float = 0.6,
        adversarial_threshold: float = 0.5,
        alpha: float = 0.5,
        truth_weight: float = 0.7,
        auto_heal: bool = True
    ):
        self.tracker = LiveDriftTracker()
        self.adversary = AdversarialSenser(threshold=adversarial_threshold)
        self.drift_calc = DriftCalculator(self.tracker)
        self.senser = UncertaintySenser(self.tracker, alpha=alpha)
        self.regulator = TruthRegulator(
            self.tracker, 
            self.senser, 
            self.drift_calc, 
            resilience_threshold=resilience_threshold, 
            truth_weight=truth_weight
        )
        self.healer = BranchHealer(self.regulator)
        self.auto_heal = auto_heal
        self.history: List[Dict[str, Any]] = []

    def process_interaction(
        self, 
        payload: ReasoningPayload, 
        parent_ids: Optional[List[str]] = None,
        root_intent: Optional[str] = None,
        regenerator_fn: Optional[Callable[[str, List[ReasoningPayload]], ReasoningPayload]] = None
    ) -> Dict[str, Any]:
        """
        Processes a new reasoning step: updates the DIG, senses drift, and triggers healing if necessary.
        """
        # 1. Add to Tracker
        node_id = self.tracker.add_reasoning_node(payload, parent_ids=parent_ids)
        
        # 2. Update Drift Scores
        self.drift_calc.update_tracker_drift_scores()
        
        # 3. Adversarial Assessment (Optional stress test or vulnerability check)
        vulnerabilities = self.adversary.sense_weak_points(payload)
        
        # 4. Check Resilience
        resilience = self.regulator.calculate_resilience_score(node_id)
        report = self.regulator.get_truth_report()
        
        status = {
            "node_id": node_id,
            "resilience_score": resilience,
            "vulnerabilities": vulnerabilities,
            "system_status": report["status"],
            "timestamp": datetime.datetime.now().isoformat(),
            "healing_performed": False
        }

        # 5. Automatic Healing if breached
        if self.auto_heal and report["status"] == "drifting":
            if root_intent:
                healing_results = self.healer.heal_drifting_branches(
                    root_intent=root_intent,
                    regenerator_fn=regenerator_fn
                )
                status["healing_performed"] = True
                status["healing_results"] = healing_results
                # If healed nodes were added, the tracker state has changed
                # We update the status status for the caller
                status["system_status"] = "healed" if healing_results.get("healed_nodes") else "pruned"

        self.history.append(status)
        return status

    def get_system_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the current system state.
        """
        metrics = self.tracker.get_drift_metrics()
        report = self.regulator.get_truth_report()
        
        return {
            "total_nodes": self.tracker.dig.number_of_nodes(),
            "avg_drift": metrics.get("avg_drift", 0.0),
            "max_drift": metrics.get("max_drift", 0.0),
            "vulnerability_index": self.adversary.calculate_vulnerability_index(
                [self.tracker.get_payload(n) for n in self.tracker.dig.nodes() if self.tracker.get_payload(n)]
            ),
            "status": report["status"],
            "drift_origin": report.get("drift_origin"),
            "drifting_nodes_count": len(report.get("drifting_nodes", []))
        }

    def stress_test_node(self, node_id: str, mode: str = "conflicting_hint") -> ReasoningPayload:
        """
        Applies adversarial stress to a specific node to test its resilience.
        """
        payload = self.tracker.get_payload(node_id)
        if not payload:
            raise ValueError(f"Node {node_id} not found.")
            
        return self.adversary.stress_test(payload, mode=mode)
