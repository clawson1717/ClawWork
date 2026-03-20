"""
Capacity-Weighted Voting for Ensemble Response Coordination.

Implements voting mechanisms that weight agent votes by their estimated
information capacity, allowing higher-capacity agents to have more
influence on the final ensemble decision.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VoteResult:
    """
    Result of a capacity-weighted voting process.
    
    Attributes:
        winning_response: The response that received the highest weighted vote
        weights_used: Dictionary mapping agent indices to their softmax weights
        disagreement_detected: Whether significant disagreement was detected
        disagreement_score: Score indicating level of disagreement (0.0-1.0+)
    """
    winning_response: str
    weights_used: Dict[int, float]
    disagreement_detected: bool
    disagreement_score: float


@dataclass
class DisagreementResult:
    """
    Result of disagreement detection among agent responses.
    
    Attributes:
        has_disagreement: Whether disagreement exceeds the threshold
        disagreement_score: Quantitative measure of disagreement (entropy-based)
        dissenting_agents: List of agent indices that disagreed with majority
    """
    has_disagreement: bool
    disagreement_score: float
    dissenting_agents: List[int]


class CapacityWeightedVoter:
    """
    Voter that weights agent responses by their information capacity.
    
    Uses softmax over capacities to compute weights, giving higher capacity
    agents more influence in the voting process. Also provides disagreement
    detection to identify when agents strongly disagree.
    
    Example:
        >>> voter = CapacityWeightedVoter()
        >>> responses = ["A", "B", "A", "C"]
        >>> capacities = [1.0, 2.0, 1.5, 0.5]
        >>> result = voter.weighted_vote(responses, capacities)
        >>> print(result.winning_response)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize the capacity-weighted voter.
        
        Args:
            temperature: Softmax temperature. Higher values make weights more
                        uniform, lower values make them more extreme.
        """
        self.temperature = temperature
    
    def weighted_vote(
        self,
        responses: List[str],
        capacities: List[float]
    ) -> VoteResult:
        """
        Conduct a capacity-weighted vote on agent responses.
        
        Uses softmax over capacities to compute weights, then sums weights
        for each unique response. The response with highest total weight wins.
        
        Args:
            responses: List of response strings from agents
            capacities: List of capacity values (one per agent)
            
        Returns:
            VoteResult with winning response and metadata
            
        Raises:
            ValueError: If responses or capacities are empty or mismatched
        """
        if not responses:
            raise ValueError("responses cannot be empty")
        if not capacities:
            raise ValueError("capacities cannot be empty")
        if len(responses) != len(capacities):
            raise ValueError(
                f"responses ({len(responses)}) and capacities ({len(capacities)}) "
                "must have the same length"
            )
        
        # Compute softmax weights over capacities
        weights = self._softmax(capacities)
        
        # Map agent indices to weights
        weights_used = {i: weights[i] for i in range(len(responses))}
        
        # Sum weights for each unique response
        response_weights: Dict[str, float] = {}
        for i, response in enumerate(responses):
            normalized = response.strip()
            response_weights[normalized] = (
                response_weights.get(normalized, 0.0) + weights[i]
            )
        
        # Find winning response
        winning_response = max(response_weights, key=response_weights.get)
        
        # Detect disagreement
        disagreement_result = self.detect_disagreement(
            responses,
            threshold=0.5  # Default threshold
        )
        
        return VoteResult(
            winning_response=winning_response,
            weights_used=weights_used,
            disagreement_detected=disagreement_result.has_disagreement,
            disagreement_score=disagreement_result.disagreement_score
        )
    
    def detect_disagreement(
        self,
        responses: List[str],
        threshold: float
    ) -> DisagreementResult:
        """
        Detect when agents strongly disagree.
        
        Uses entropy of the response distribution as a disagreement measure.
        Higher entropy indicates more disagreement (responses are spread
        across more unique values). Agents whose response differs from the
        plurality winner are considered dissenting.
        
        Args:
            responses: List of response strings from agents
            threshold: Entropy threshold above which disagreement is flagged
            
        Returns:
            DisagreementResult with disagreement detection results
            
        Raises:
            ValueError: If responses is empty or threshold is negative
        """
        if not responses:
            raise ValueError("responses cannot be empty")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        
        # Calculate response distribution entropy
        disagreement_score = self._calculate_entropy(responses)
        
        # Determine if disagreement exceeds threshold
        has_disagreement = disagreement_score > threshold
        
        # Find dissenting agents (those not voting with plurality winner)
        dissenting_agents = self._find_dissenting_agents(responses)
        
        return DisagreementResult(
            has_disagreement=has_disagreement,
            disagreement_score=disagreement_score,
            dissenting_agents=dissenting_agents
        )
    
    def _softmax(self, values: List[float]) -> List[float]:
        """
        Compute softmax weights over a list of values.
        
        weights_i = exp(value_i / temperature) / sum(exp(value_j / temperature))
        
        Uses the maximum value for numerical stability.
        
        Args:
            values: List of values to compute softmax over
            
        Returns:
            List of weights that sum to 1.0
        """
        if not values:
            return []
        
        # Scale by temperature
        scaled = [v / self.temperature for v in values]
        
        # Subtract max for numerical stability
        max_val = max(scaled)
        scaled = [v - max_val for v in scaled]
        
        # Compute exponentials
        exp_values = [math.exp(v) for v in scaled]
        sum_exp = sum(exp_values)
        
        if sum_exp == 0:
            # All values are -inf, return uniform weights
            return [1.0 / len(values)] * len(values)
        
        # Normalize
        return [exp_val / sum_exp for exp_val in exp_values]
    
    def _calculate_entropy(self, responses: List[str]) -> float:
        """
        Calculate the entropy of a response distribution.
        
        Higher entropy means more disagreement (responses are spread
        across more unique values).
        
        Args:
            responses: List of response strings
            
        Returns:
            Entropy in bits
        """
        if not responses:
            return 0.0
        
        # Count unique responses
        response_counts: Dict[str, int] = {}
        for response in responses:
            normalized = response.strip()
            response_counts[normalized] = response_counts.get(normalized, 0) + 1
        
        # Calculate entropy
        n = len(responses)
        entropy = 0.0
        
        for count in response_counts.values():
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _find_dissenting_agents(self, responses: List[str]) -> List[int]:
        """
        Find indices of agents whose responses differ from the plurality winner.
        
        Args:
            responses: List of response strings
            
        Returns:
            List of agent indices that disagreed with the majority
        """
        if not responses:
            return []
        
        # Find plurality winner
        response_counts: Dict[str, int] = {}
        for response in responses:
            normalized = response.strip()
            response_counts[normalized] = response_counts.get(normalized, 0) + 1
        
        winner = max(response_counts, key=response_counts.get)
        
        # Find dissenting agents
        dissenting = []
        for i, response in enumerate(responses):
            if response.strip() != winner:
                dissenting.append(i)
        
        return dissenting
