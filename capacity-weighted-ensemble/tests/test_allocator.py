"""
Tests for Test-Time Compute Allocator.
"""

import pytest
from src.allocator import AllocationResult, ComputeAllocator


class TestAllocationResult:
    """Tests for AllocationResult dataclass."""
    
    def test_valid_result_creation(self):
        """Test creating a valid AllocationResult."""
        result = AllocationResult(
            agent_id="agent_1",
            allocated=100,
            used=20,
            remaining=80,
            score=0.75
        )
        assert result.agent_id == "agent_1"
        assert result.allocated == 100
        assert result.used == 20
        assert result.remaining == 80
        assert result.score == 0.75
    
    def test_invalid_allocated_negative(self):
        """Test that negative allocated raises ValueError."""
        with pytest.raises(ValueError, match="allocated must be non-negative"):
            AllocationResult(agent_id="agent_1", allocated=-10, used=0, remaining=0, score=0.0)
    
    def test_invalid_used_negative(self):
        """Test that negative used raises ValueError."""
        with pytest.raises(ValueError, match="used must be non-negative"):
            AllocationResult(agent_id="agent_1", allocated=100, used=-5, remaining=100, score=0.0)
    
    def test_invalid_remaining_negative(self):
        """Test that negative remaining raises ValueError."""
        with pytest.raises(ValueError, match="remaining must be non-negative"):
            AllocationResult(agent_id="agent_1", allocated=100, used=0, remaining=-10, score=0.0)
    
    def test_invalid_score_out_of_range(self):
        """Test that score outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
            AllocationResult(agent_id="agent_1", allocated=100, used=0, remaining=100, score=1.5)
        
        with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
            AllocationResult(agent_id="agent_1", allocated=100, used=0, remaining=100, score=-0.1)
    
    def test_used_exceeds_allocated(self):
        """Test that used > allocated raises ValueError."""
        with pytest.raises(ValueError, match="used .* cannot exceed allocated"):
            AllocationResult(agent_id="agent_1", allocated=50, used=60, remaining=0, score=0.5)
    
    def test_remaining_computed_from_allocated_minus_used(self):
        """Test that remaining is computed if not set."""
        result = AllocationResult(
            agent_id="agent_1",
            allocated=100,
            used=30,
            remaining=0,  # Will be computed
            score=0.5
        )
        assert result.remaining == 70
    
    def test_use_method(self):
        """Test using budget via use() method."""
        result = AllocationResult(
            agent_id="agent_1",
            allocated=100,
            used=0,
            remaining=100,
            score=0.5
        )
        result.use(30)
        assert result.used == 30
        assert result.remaining == 70
    
    def test_use_exceeds_remaining(self):
        """Test that using more than remaining raises ValueError."""
        result = AllocationResult(
            agent_id="agent_1",
            allocated=100,
            used=0,
            remaining=100,
            score=0.5
        )
        with pytest.raises(ValueError, match="Cannot use 150"):
            result.use(150)
    
    def test_reset_usage(self):
        """Test resetting usage."""
        result = AllocationResult(
            agent_id="agent_1",
            allocated=100,
            used=50,
            remaining=50,
            score=0.5
        )
        result.reset_usage()
        assert result.used == 0
        assert result.remaining == 100


class TestComputeAllocatorInit:
    """Tests for ComputeAllocator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        allocator = ComputeAllocator(total_budget=1000)
        assert allocator.total_budget == 1000
        assert allocator.capacity_weight == 0.7
        assert allocator.uncertainty_weight == 0.3
    
    def test_custom_initialization(self):
        """Test custom weights."""
        allocator = ComputeAllocator(
            total_budget=500,
            capacity_weight=0.6,
            uncertainty_weight=0.4
        )
        assert allocator.total_budget == 500
        assert allocator.capacity_weight == 0.6
        assert allocator.uncertainty_weight == 0.4
    
    def test_invalid_budget_negative(self):
        """Test that negative budget raises ValueError."""
        with pytest.raises(ValueError, match="total_budget must be non-negative"):
            ComputeAllocator(total_budget=-100)
    
    def test_invalid_capacity_weight_out_of_range(self):
        """Test that capacity_weight outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="capacity_weight must be between 0 and 1"):
            ComputeAllocator(total_budget=100, capacity_weight=1.5)
    
    def test_invalid_uncertainty_weight_out_of_range(self):
        """Test that uncertainty_weight outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="uncertainty_weight must be between 0 and 1"):
            ComputeAllocator(total_budget=100, uncertainty_weight=-0.1)
    
    def test_weights_dont_sum_to_one(self):
        """Test that weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="capacity_weight \\+ uncertainty_weight must equal 1.0"):
            ComputeAllocator(total_budget=100, capacity_weight=0.5, uncertainty_weight=0.3)


class TestBasicAllocation:
    """Tests for basic allocation functionality."""
    
    def test_single_agent_allocation(self):
        """Test allocating to a single agent."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        assert result["agent_1"] == 100
    
    def test_two_agents_equal_allocation(self):
        """Test allocation when agents have equal capacity and uncertainty."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.2}
        )
        # Equal scores should split evenly
        assert result["agent_1"] == 50
        assert result["agent_2"] == 50
    
    def test_empty_agents_list(self):
        """Test allocation with empty agents list."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(agents=[], capacities={}, uncertainties={})
        assert result == {}
    
    def test_zero_budget(self):
        """Test allocation with zero budget."""
        allocator = ComputeAllocator(total_budget=0)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 5.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.3}
        )
        assert result["agent_1"] == 0
        assert result["agent_2"] == 0


class TestCapacityWeightedAllocation:
    """Tests for capacity-weighted allocation."""
    
    def test_higher_capacity_gets_more(self):
        """Test that higher capacity agents get more allocation."""
        allocator = ComputeAllocator(total_budget=100, capacity_weight=1.0, uncertainty_weight=0.0)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 20.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.5, "agent_2": 0.5}  # Equal uncertainty
        )
        # agent_1 has 2x capacity, should get more
        assert result["agent_1"] > result["agent_2"]
    
    def test_capacity_only_weighting(self):
        """Test allocation with capacity as sole factor."""
        allocator = ComputeAllocator(total_budget=100, capacity_weight=1.0, uncertainty_weight=0.0)
        result = allocator.allocate(
            agents=["agent_1", "agent_2", "agent_3"],
            capacities={"agent_1": 30.0, "agent_2": 20.0, "agent_3": 10.0},
            uncertainties={"agent_1": 0.5, "agent_2": 0.5, "agent_3": 0.5}
        )
        # Ratios should be 3:2:1
        assert result["agent_1"] > result["agent_2"] > result["agent_3"]


class TestUncertaintyWeightedAllocation:
    """Tests for uncertainty-weighted allocation."""
    
    def test_lower_uncertainty_gets_more(self):
        """Test that lower uncertainty agents get more allocation."""
        allocator = ComputeAllocator(total_budget=100, capacity_weight=0.0, uncertainty_weight=1.0)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},  # Equal capacity
            uncertainties={"agent_1": 0.2, "agent_2": 0.8}
        )
        # agent_1 has lower uncertainty, should get more
        assert result["agent_1"] > result["agent_2"]
    
    def test_uncertainty_only_weighting(self):
        """Test allocation with uncertainty as sole factor."""
        allocator = ComputeAllocator(total_budget=100, capacity_weight=0.0, uncertainty_weight=1.0)
        result = allocator.allocate(
            agents=["agent_1", "agent_2", "agent_3"],
            capacities={"agent_1": 10.0, "agent_2": 10.0, "agent_3": 10.0},
            uncertainties={"agent_1": 0.0, "agent_2": 0.5, "agent_3": 1.0}
        )
        # Lower uncertainty = higher allocation (inverted)
        assert result["agent_1"] > result["agent_2"] > result["agent_3"]


class TestCombinedWeighting:
    """Tests for combined capacity and uncertainty weighting."""
    
    def test_combined_weights_default(self):
        """Test allocation with default weights (0.7 capacity, 0.3 uncertainty)."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 20.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.4}
        )
        # agent_1: higher capacity AND lower uncertainty
        assert result["agent_1"] > result["agent_2"]
    
    def test_capacity_dominates_uncertainty(self):
        """Test that capacity is the primary factor."""
        allocator = ComputeAllocator(total_budget=100, capacity_weight=0.9, uncertainty_weight=0.1)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 50.0, "agent_2": 10.0},  # agent_1 higher capacity
            uncertainties={"agent_1": 0.8, "agent_2": 0.2}  # agent_2 lower uncertainty
        )
        # Capacity dominates, so agent_1 should still get more
        assert result["agent_1"] > result["agent_2"]


class TestBudgetTracking:
    """Tests for budget tracking."""
    
    def test_get_budget_after_allocation(self):
        """Test getting budget after allocation."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        assert allocator.get_budget("agent_1") == 100
    
    def test_get_budget_nonexistent_agent(self):
        """Test getting budget for unallocated agent raises KeyError."""
        allocator = ComputeAllocator(total_budget=100)
        with pytest.raises(KeyError, match="Agent .* has not been allocated"):
            allocator.get_budget("unknown_agent")
    
    def test_total_allocated(self):
        """Test total allocated computation."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.2}
        )
        assert allocator.total_allocated() == 100
    
    def test_total_remaining(self):
        """Test total remaining computation."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.2}
        )
        assert allocator.total_remaining() == 100


class TestBudgetUpdates:
    """Tests for budget update operations."""
    
    def test_update_budget_use(self):
        """Test using budget via update_budget."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        allocator.update_budget("agent_1", -30)
        assert allocator.get_budget("agent_1") == 70
    
    def test_update_budget_add(self):
        """Test adding to budget via update_budget."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        allocator.update_budget("agent_1", 50)
        assert allocator.get_budget("agent_1") == 150
    
    def test_update_budget_exceeds_remaining(self):
        """Test that using more than remaining raises ValueError."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        with pytest.raises(ValueError, match="Cannot use 150"):
            allocator.update_budget("agent_1", -150)
    
    def test_use_budget_convenience_method(self):
        """Test use_budget convenience method."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1"],
            capacities={"agent_1": 10.0},
            uncertainties={"agent_1": 0.2}
        )
        allocator.use_budget("agent_1", 40)
        assert allocator.get_budget("agent_1") == 60
    
    def test_reset(self):
        """Test resetting allocator."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.2}
        )
        allocator.use_budget("agent_1", 50)
        allocator.reset()
        
        # After reset, no allocations exist
        with pytest.raises(KeyError):
            allocator.get_budget("agent_1")


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_all_zero_capacities(self):
        """Test allocation when all capacities are zero."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 0.0, "agent_2": 0.0},
            uncertainties={"agent_1": 0.5, "agent_2": 0.5}
        )
        # Should distribute equally (all scores equal at 0.5)
        assert result["agent_1"] == 50
        assert result["agent_2"] == 50
    
    def test_all_zero_uncertainties(self):
        """Test allocation when all uncertainties are zero."""
        allocator = ComputeAllocator(total_budget=100)
        result = allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.0, "agent_2": 0.0}
        )
        # Should distribute equally
        assert result["agent_1"] == 50
        assert result["agent_2"] == 50
    
    def test_missing_capacity(self):
        """Test that missing capacity raises ValueError."""
        allocator = ComputeAllocator(total_budget=100)
        with pytest.raises(ValueError, match="Missing capacity for agent"):
            allocator.allocate(
                agents=["agent_1"],
                capacities={},
                uncertainties={"agent_1": 0.5}
            )
    
    def test_missing_uncertainty(self):
        """Test that missing uncertainty raises ValueError."""
        allocator = ComputeAllocator(total_budget=100)
        with pytest.raises(ValueError, match="Missing uncertainty for agent"):
            allocator.allocate(
                agents=["agent_1"],
                capacities={"agent_1": 10.0},
                uncertainties={}
            )
    
    def test_negative_capacity(self):
        """Test that negative capacity raises ValueError."""
        allocator = ComputeAllocator(total_budget=100)
        with pytest.raises(ValueError, match="Capacity for .* must be non-negative"):
            allocator.allocate(
                agents=["agent_1"],
                capacities={"agent_1": -5.0},
                uncertainties={"agent_1": 0.5}
            )
    
    def test_negative_uncertainty(self):
        """Test that negative uncertainty raises ValueError."""
        allocator = ComputeAllocator(total_budget=100)
        with pytest.raises(ValueError, match="Uncertainty for .* must be non-negative"):
            allocator.allocate(
                agents=["agent_1"],
                capacities={"agent_1": 10.0},
                uncertainties={"agent_1": -0.2}
            )
    
    def test_large_number_of_agents(self):
        """Test allocation with many agents."""
        allocator = ComputeAllocator(total_budget=1000)
        agents = [f"agent_{i}" for i in range(100)]
        capacities = {f"agent_{i}": float(i + 1) for i in range(100)}
        uncertainties = {f"agent_{i}": 0.5 for i in range(100)}
        
        result = allocator.allocate(agents=agents, capacities=capacities, uncertainties=uncertainties)
        
        # Total should equal budget
        assert sum(result.values()) == 1000
        # Higher capacity agents should get more
        assert result["agent_99"] > result["agent_0"]


class TestTotalBudgetEnforcement:
    """Tests for total budget enforcement."""
    
    def test_total_allocation_equals_budget(self):
        """Test that total allocation equals budget."""
        allocator = ComputeAllocator(total_budget=137)
        result = allocator.allocate(
            agents=["agent_1", "agent_2", "agent_3"],
            capacities={"agent_1": 30.0, "agent_2": 20.0, "agent_3": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.3, "agent_3": 0.4}
        )
        assert sum(result.values()) == 137
    
    def test_odd_budget_distribution(self):
        """Test that odd budget is fully distributed."""
        allocator = ComputeAllocator(total_budget=101)
        result = allocator.allocate(
            agents=["agent_1", "agent_2", "agent_3"],
            capacities={"agent_1": 10.0, "agent_2": 10.0, "agent_3": 10.0},
            uncertainties={"agent_1": 0.5, "agent_2": 0.5, "agent_3": 0.5}
        )
        # Should distribute 101 across 3 agents
        assert sum(result.values()) == 101
    
    def test_get_all_allocations(self):
        """Test getting all allocations."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 20.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.4}
        )
        all_allocations = allocator.get_all_allocations()
        assert len(all_allocations) == 2
        assert "agent_1" in all_allocations
        assert "agent_2" in all_allocations
        assert isinstance(all_allocations["agent_1"], AllocationResult)
    
    def test_total_used(self):
        """Test total used computation."""
        allocator = ComputeAllocator(total_budget=100)
        allocator.allocate(
            agents=["agent_1", "agent_2"],
            capacities={"agent_1": 10.0, "agent_2": 10.0},
            uncertainties={"agent_1": 0.2, "agent_2": 0.2}
        )
        allocator.use_budget("agent_1", 30)
        allocator.use_budget("agent_2", 20)
        assert allocator.total_used() == 50
