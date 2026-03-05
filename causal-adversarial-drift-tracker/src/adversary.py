import random
from typing import List, Dict, Any
try:
    from .payload import ReasoningPayload
except ImportError:
    from payload import ReasoningPayload

class AdversarialSenser:
    """
    Senses vulnerabilities and injects adversarial noise into ReasoningPayloads.
    Inspired by TraderBench's methodology for evaluating reasoning robustness.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        # Weak point indicators (simulated for now, would be LLM-driven in production)
        self.indicators = ["perhaps", "maybe", "likely", "assume", "suppose", "if so"]

    def sense_weak_points(self, payload: ReasoningPayload) -> List[Dict[str, Any]]:
        """
        Identifies potential weak points in the reasoning content such as
        ambiguity, logical leaps, or unverified assumptions.
        """
        weak_points = []
        content_lower = payload.content.lower()
        
        # 1. Check for linguistic markers of uncertainty
        for indicator in self.indicators:
            if indicator in content_lower:
                weak_points.append({
                    "type": "ambiguity",
                    "marker": indicator,
                    "severity": 0.4
                })

        # 2. Check for "logical leaps" (long sentences without connecting particles)
        # Simplified: Check length of reasoning vs punctuation density
        if len(payload.content) > 100 and "." not in payload.content:
             weak_points.append({
                "type": "logical_leap",
                "reason": "Long reasoning block without intermediate validation/punctuation.",
                "severity": 0.6
            })

        # 3. Check for high drift score
        if payload.drift_score > self.threshold:
            weak_points.append({
                "type": "drift_vulnerability",
                "score": payload.drift_score,
                "severity": payload.drift_score
            })

        return weak_points

    def stress_test(self, payload: ReasoningPayload, mode: str = "conflicting_hint") -> ReasoningPayload:
        """
        Creates a 'stressed' version of the ReasoningPayload by injecting noise
         or suggesting conflicting information.
        """
        new_content = payload.content
        if mode == "conflicting_hint":
            new_content += " [ADVERSARIAL HINT: However, historical data suggests the opposite trend might occur.]"
        elif mode == "noise":
            noise = "".join(random.choices("!@#$%^&*", k=10))
            new_content = f"{payload.content} {noise}"
        
        # Create a new payload based on the old one but with noise/stress
        stressed_payload = ReasoningPayload(
            source_id=f"adversary_{payload.source_id}",
            content=new_content,
            drift_score=payload.drift_score + 0.1, # Stressing increases drift
            semantic_vector=payload.semantic_vector
        )
        return stressed_payload

    def calculate_vulnerability_index(self, payloads: List[ReasoningPayload]) -> float:
        """
        Calculates an aggregate vulnerability index for a chain of payloads.
        """
        if not payloads:
            return 0.0
        
        total_severity = 0.0
        for p in payloads:
            wp = self.sense_weak_points(p)
            total_severity += sum(item["severity"] for item in wp)
            
        return min(1.0, total_severity / len(payloads))
