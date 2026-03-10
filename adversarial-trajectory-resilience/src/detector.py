from dataclasses import dataclass
from typing import List, Dict, Any
import re

@dataclass
class DetectionResult:
    detected: bool
    score: float

class FailureModeDetector:
    """
    Detects failure modes and adversarial patterns.
    """

    def __init__(self):
        self.patterns = {
            "self_doubt": [r"are you sure", r"rethink", r"mistake"],
            "suggestion_hijacking": [r"the correct answer is actually", r"ignore previous"]
        }

    def _calculate_score(self, text: str, patterns: List[str]) -> float:
        if not text: return 0.0
        matches = 0
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches += 1
        return min(1.0, matches / 1.0) # Simplified

    def detect_all(self, text: str) -> Dict[str, DetectionResult]:
        res = {}
        for mode, patterns in self.patterns.items():
            score = self._calculate_score(text, patterns)
            res[mode] = DetectionResult(detected=score > 0.4, score=score)
        return res
