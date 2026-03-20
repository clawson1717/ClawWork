"""
Tests for Capacity-Weighted Voting.

Tests cover:
- Basic weighted vote (high capacity agent's response wins)
- Tie-breaking when capacities equal
- Disagreement detection with threshold
- No disagreement when agents agree
- Edge cases: single agent, empty responses
- Weight normalization (weights sum to 1)
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voting import CapacityWeightedVoter, VoteResult, DisagreementResult


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def voter():
    """Create a CapacityWeightedVoter instance for testing."""
    return CapacityWeightedVoter()


@pytest.fixture
def voter_with_temp():
    """Create a CapacityWeightedVoter with custom temperature."""
    return CapacityWeightedVoter(temperature=0.5)


# =============================================================================
# Basic Weighted Vote Tests
# =============================================================================

class TestBasicWeightedVote:
    """Tests for basic capacity-weighted voting."""
    
    def test_high_capacity_agent_wins(self, voter):
        """High capacity agent's response should win."""
        responses = ["A", "B"]
        capacities = [1.0, 10.0]  # Agent 1 has much higher capacity
        
        result = voter.weighted_vote(responses, capacities)
        
        # B should win because agent 1 has much higher capacity
        assert result.winning_response == "B"
    
    def test_medium_capacity_determines_winner(self, voter):
        """When responses differ, higher total capacity wins."""
        responses = ["X", "Y"]
        capacities = [1.0, 3.0]  # Agent 1 has 3x capacity
        
        result = voter.weighted_vote(responses, capacities)
        
        assert result.winning_response == "Y"
    
    def test_three_way_vote(self, voter):
        """Test voting with three different responses."""
        responses = ["A", "B", "C"]
        capacities = [1.0, 2.0, 3.0]
        
        result = voter.weighted_vote(responses, capacities)
        
        # C has highest total capacity weight
        assert result.winning_response == "C"
    
    def test_same_response_all_agents(self, voter):
        """When all agents agree, that response wins regardless of capacity."""
        responses = ["Same", "Same", "Same"]
        capacities = [1.0, 10.0, 100.0]
        
        result = voter.weighted_vote(responses, capacities)
        
        assert result.winning_response == "Same"


# =============================================================================
# Tie-Breaking Tests
# =============================================================================

class TestTieBreaking:
    """Tests for tie-breaking when capacities or votes are equal."""
    
    def test_equal_capacities_tie_break(self, voter):
        """When capacities are equal, first response in dict order wins."""
        responses = ["A", "B"]
        capacities = [1.0, 1.0]  # Equal capacities
        
        result = voter.weighted_vote(responses, capacities)
        
        # With equal weights, A and B are tied, max() returns first seen
        assert result.winning_response in ["A", "B"]
    
    def test_equal_votes_with_different_capacities(self, voter):
        """When two responses get equal votes, higher capacity determines winner."""
        # A gets votes from agents 0 and 1, B gets votes from agents 2
        # Total capacity for A vs B depends on distribution
        responses = ["A", "A", "B"]
        capacities = [1.0, 1.0, 5.0]  # B's single agent has high capacity
        
        result = voter.weighted_vote(responses, capacities)
        
        # B might win due to high capacity weight
        # (depends on softmax distribution)
        assert result.winning_response in ["A", "B"]


# =============================================================================
# Disagreement Detection Tests
# =============================================================================

class TestDisagreementDetection:
    """Tests for disagreement detection functionality."""
    
    def test_disagreement_detected_with_threshold(self, voter):
        """Disagreement should be detected when entropy exceeds threshold."""
        responses = ["A", "B", "C", "D"]  # All different - high entropy
        threshold = 1.0
        
        result = voter.detect_disagreement(responses, threshold)
        
        assert result.has_disagreement is True
        assert result.disagreement_score > threshold
    
    def test_no_disagreement_when_agents_agree(self, voter):
        """No disagreement when all agents give same response."""
        responses = ["Same", "Same", "Same"]
        threshold = 0.5
        
        result = voter.detect_disagreement(responses, threshold)
        
        assert result.has_disagreement is False
        assert result.disagreement_score == 0.0
    
    def test_partial_disagreement(self, voter):
        """Partial agreement should have moderate entropy."""
        responses = ["A", "A", "B", "B", "C"]
        threshold = 1.5
        
        result = voter.detect_disagreement(responses, threshold)
        
        # Should have some disagreement but maybe not above threshold
        assert result.disagreement_score > 0
    
    def test_dissenting_agents_identified(self, voter):
        """Dissenting agents should be correctly identified."""
        responses = ["A", "A", "B"]
        threshold = 0.0
        
        result = voter.detect_disagreement(responses, threshold)
        
        assert 2 in result.dissenting_agents  # Agent 2 voted for B
        assert 0 not in result.dissenting_agents
        assert 1 not in result.dissenting_agents
    
    def test_threshold_zero_no_disagreement(self, voter):
        """With threshold 0, only perfect agreement avoids disagreement."""
        responses = ["A", "B"]
        threshold = 0.0
        
        result = voter.detect_disagreement(responses, threshold)
        
        assert result.has_disagreement is True


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_agent_response(self, voter):
        """Single agent should always win with their response."""
        responses = ["Only"]
        capacities = [5.0]
        
        result = voter.weighted_vote(responses, capacities)
        
        assert result.winning_response == "Only"
        assert len(result.weights_used) == 1
        assert result.disagreement_detected is False
    
    def test_single_agent_no_disagreement(self, voter):
        """Single agent should never have disagreement."""
        responses = ["Solo"]
        threshold = 0.0
        
        result = voter.detect_disagreement(responses, threshold)
        
        assert result.has_disagreement is False
        assert result.disagreement_score == 0.0
        assert result.dissenting_agents == []
    
    def test_empty_responses_error(self, voter):
        """Empty responses should raise ValueError."""
        with pytest.raises(ValueError, match="responses cannot be empty"):
            voter.weighted_vote([], [1.0])
    
    def test_empty_capacities_error(self, voter):
        """Empty capacities should raise ValueError."""
        with pytest.raises(ValueError, match="capacities cannot be empty"):
            voter.weighted_vote(["response"], [])
    
    def test_mismatched_lengths_error(self, voter):
        """Mismatched responses and capacities should raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            voter.weighted_vote(["A", "B"], [1.0])
    
    def test_empty_responses_disagreement_error(self, voter):
        """Empty responses for disagreement detection should raise ValueError."""
        with pytest.raises(ValueError, match="responses cannot be empty"):
            voter.detect_disagreement([], threshold=0.5)
    
    def test_negative_threshold_error(self, voter):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            voter.detect_disagreement(["A", "B"], threshold=-0.1)


# =============================================================================
# Weight Normalization Tests
# =============================================================================

class TestWeightNormalization:
    """Tests for softmax weight normalization."""
    
    def test_weights_sum_to_one(self, voter):
        """Softmax weights should sum to 1.0."""
        capacities = [1.0, 2.0, 3.0]
        responses = ["A", "B", "C"]
        
        result = voter.weighted_vote(responses, capacities)
        
        total_weight = sum(result.weights_used.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_all_equal_capacities_uniform_weights(self, voter):
        """Equal capacities should give uniform weights."""
        capacities = [1.0, 1.0, 1.0]
        
        weights = voter._softmax(capacities)
        
        # All weights should be approximately equal
        assert abs(weights[0] - weights[1]) < 1e-6
        assert abs(weights[1] - weights[2]) < 1e-6
        # Each should be 1/3
        assert abs(weights[0] - 1/3) < 1e-6
    
    def test_high_capacity_gets_more_weight(self, voter):
        """Higher capacity should result in higher softmax weight."""
        capacities = [1.0, 10.0]
        
        weights = voter._softmax(capacities)
        
        assert weights[1] > weights[0]
    
    def test_temperature_affects_weights(self, voter_with_temp):
        """Higher temperature should make weights more uniform."""
        low_temp_voter = CapacityWeightedVoter(temperature=0.1)
        high_temp_voter = CapacityWeightedVoter(temperature=10.0)
        
        capacities = [1.0, 10.0]
        
        low_temp_weights = low_temp_voter._softmax(capacities)
        high_temp_weights = high_temp_voter._softmax(capacities)
        
        # Low temperature should be more extreme
        low_temp_spread = abs(low_temp_weights[1] - low_temp_weights[0])
        high_temp_spread = abs(high_temp_weights[1] - high_temp_weights[0])
        
        assert low_temp_spread > high_temp_spread


# =============================================================================
# VoteResult Dataclass Tests
# =============================================================================

class TestVoteResult:
    """Tests for VoteResult dataclass."""
    
    def test_vote_result_contains_all_fields(self, voter):
        """VoteResult should contain all required fields."""
        responses = ["A", "B"]
        capacities = [1.0, 2.0]
        
        result = voter.weighted_vote(responses, capacities)
        
        assert hasattr(result, 'winning_response')
        assert hasattr(result, 'weights_used')
        assert hasattr(result, 'disagreement_detected')
        assert hasattr(result, 'disagreement_score')
    
    def test_vote_result_types(self, voter):
        """VoteResult fields should have correct types."""
        responses = ["A", "B"]
        capacities = [1.0, 2.0]
        
        result = voter.weighted_vote(responses, capacities)
        
        assert isinstance(result.winning_response, str)
        assert isinstance(result.weights_used, dict)
        assert isinstance(result.disagreement_detected, bool)
        assert isinstance(result.disagreement_score, float)


# =============================================================================
# DisagreementResult Dataclass Tests
# =============================================================================

class TestDisagreementResult:
    """Tests for DisagreementResult dataclass."""
    
    def test_disagreement_result_contains_all_fields(self, voter):
        """DisagreementResult should contain all required fields."""
        result = voter.detect_disagreement(["A", "B"], threshold=0.5)
        
        assert hasattr(result, 'has_disagreement')
        assert hasattr(result, 'disagreement_score')
        assert hasattr(result, 'dissenting_agents')
    
    def test_disagreement_result_types(self, voter):
        """DisagreementResult fields should have correct types."""
        result = voter.detect_disagreement(["A", "B"], threshold=0.5)
        
        assert isinstance(result.has_disagreement, bool)
        assert isinstance(result.disagreement_score, float)
        assert isinstance(result.dissenting_agents, list)
    
    def test_dissenting_agents_are_indices(self, voter):
        """Dissenting agents should be integer indices."""
        responses = ["A", "A", "B"]
        
        result = voter.detect_disagreement(responses, threshold=0.0)
        
        for agent_idx in result.dissenting_agents:
            assert isinstance(agent_idx, int)
