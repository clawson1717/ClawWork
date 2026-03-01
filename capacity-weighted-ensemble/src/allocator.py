"""
Test-Time Compute Allocator for capacity-weighted ensemble.

Allocates compute budget to agents based on their capacity (primary factor)
and uncertainty (secondary factor).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AllocationResult:
    """
    Result of compute allocation for a single agent.
    
    Attributes:
        agent_id: Unique identifier for the agent
        allocated: Total compute units allocated
        used: Compute units already used
        remaining: Compute units remaining
        score: Allocation score (normalized combination of capacity and uncertainty)
    """
    agent_id: str
    allocated: int
    used: int = 0
    remaining: int = 0
    score: float = 0.0
    
    def __post_init__(self):
        """Validate and compute derived fields after initialization."""
        if self.allocated < 0:
            raise ValueError(f"allocated must be non-negative, got {self.allocated}")
        if self.used < 0:
            raise ValueError(f"used must be non-negative, got {self.used}")
        if self.remaining < 0:
            raise ValueError(f"remaining must be non-negative, got {self.remaining}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if self.used > self.allocated:
            raise ValueError(f"used ({self.used}) cannot exceed allocated ({self.allocated})")
        
        # If remaining wasn't explicitly set, compute it
        if self.remaining == 0 and self.allocated > 0:
            object.__setattr__(self, 'remaining', self.allocated - self.used)
    
    def use(self, amount: int) -> None:
        """
        Mark some compute units as used.
        
        Args:
            amount: Number of compute units to mark as used
            
        Raises:
            ValueError: If amount exceeds remaining budget
        """
        if amount < 0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        if amount > self.remaining:
            raise ValueError(f"Cannot use {amount}, only {self.remaining} remaining")
        
        object.__setattr__(self, 'used', self.used + amount)
        object.__setattr__(self, 'remaining', self.remaining - amount)
    
    def reset_usage(self) -> None:
        """Reset used and remaining to initial allocated state."""
        object.__setattr__(self, 'used', 0)
        object.__setattr__(self, 'remaining', self.allocated)


class ComputeAllocator:
    """
    Allocates compute budget to agents based on capacity and uncertainty.
    
    Capacity is the primary factor (determines how much useful information an agent
    can provide), while uncertainty is secondary (agents with lower uncertainty
    get more allocation).
    
    Example:
        >>> allocator = ComputeAllocator(total_budget=1000)
        >>> allocation = allocator.allocate(
        ...     agents=["agent_a", "agent_b"],
        ...     capacities={"agent_a": 10.0, "agent_b": 5.0},
        ...     uncertainties={"agent_a": 0.2, "agent_b": 0.4}
        ... )
        >>> print(allocation)
        {'agent_a': 667, 'agent_b': 333}
    """
    
    def __init__(
        self,
        total_budget: int,
        capacity_weight: float = 0.7,
        uncertainty_weight: float = 0.3
    ):
        """
        Initialize the compute allocator.
        
        Args:
            total_budget: Total compute budget to allocate
            capacity_weight: Weight for capacity in allocation (default 0.7)
            uncertainty_weight: Weight for uncertainty in allocation (default 0.3)
            
        Raises:
            ValueError: If budget is negative or weights don't sum to 1.0
        """
        if total_budget < 0:
            raise ValueError(f"total_budget must be non-negative, got {total_budget}")
        
        if capacity_weight < 0 or capacity_weight > 1:
            raise ValueError(f"capacity_weight must be between 0 and 1, got {capacity_weight}")
        
        if uncertainty_weight < 0 or uncertainty_weight > 1:
            raise ValueError(f"uncertainty_weight must be between 0 and 1, got {uncertainty_weight}")
        
        weight_sum = capacity_weight + uncertainty_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"capacity_weight + uncertainty_weight must equal 1.0, got {weight_sum}"
            )
        
        self.total_budget = total_budget
        self.capacity_weight = capacity_weight
        self.uncertainty_weight = uncertainty_weight
        
        # Internal state for budget tracking
        self._allocations: Dict[str, AllocationResult] = {}
        self._remaining_total = total_budget
    
    def allocate(
        self,
        agents: List[str],
        capacities: Dict[str, float],
        uncertainties: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Allocate compute budget to agents based on capacity and uncertainty.
        
        The allocation algorithm:
        1. Normalize capacities to 0-1 range
        2. Normalize uncertainties to 0-1 range (inverted: high uncertainty = lower allocation)
        3. Compute weighted score for each agent
        4. Allocate proportionally to scores
        
        Args:
            agents: List of agent IDs to allocate budget to
            capacities: Dictionary mapping agent_id to capacity value
            uncertainties: Dictionary mapping agent_id to uncertainty value
            
        Returns:
            Dictionary mapping agent_id to allocated compute units
            
        Raises:
            ValueError: If agents list is empty or data is invalid
        """
        # Handle empty agents
        if not agents:
            return {}
        
        # Validate that all agents have capacity and uncertainty values
        for agent_id in agents:
            if agent_id not in capacities:
                raise ValueError(f"Missing capacity for agent: {agent_id}")
            if agent_id not in uncertainties:
                raise ValueError(f"Missing uncertainty for agent: {agent_id}")
        
        # Validate non-negative values
        for agent_id, cap in capacities.items():
            if cap < 0:
                raise ValueError(f"Capacity for {agent_id} must be non-negative, got {cap}")
        
        for agent_id, unc in uncertainties.items():
            if unc < 0:
                raise ValueError(f"Uncertainty for {agent_id} must be non-negative, got {unc}")
        
        # Handle zero budget case
        if self.total_budget == 0:
            result = {agent_id: 0 for agent_id in agents}
            for agent_id in agents:
                self._allocations[agent_id] = AllocationResult(
                    agent_id=agent_id,
                    allocated=0,
                    used=0,
                    remaining=0,
                    score=0.0
                )
            return result
        
        # Normalize capacities to 0-1 range
        norm_capacities = self._normalize_values(capacities, agents)
        
        # Normalize uncertainties to 0-1 range and invert
        # High uncertainty = lower allocation, so we use (1 - norm_uncertainty)
        norm_uncertainties = self._normalize_values(uncertainties, agents)
        
        # Compute weighted scores for each agent
        scores: Dict[str, float] = {}
        for agent_id in agents:
            norm_cap = norm_capacities[agent_id]
            norm_unc = norm_uncertainties[agent_id]
            
            # Score = capacity_weight * norm_capacity + uncertainty_weight * (1 - norm_uncertainty)
            # Higher score = more allocation
            score = (
                self.capacity_weight * norm_cap +
                self.uncertainty_weight * (1 - norm_unc)
            )
            scores[agent_id] = score
        
        # Handle case where all scores are 0
        total_score = sum(scores.values())
        if total_score == 0:
            # Distribute equally
            allocation_per_agent = self.total_budget // len(agents)
            remainder = self.total_budget % len(agents)
            
            result = {}
            for i, agent_id in enumerate(agents):
                # Distribute remainder to first agents
                result[agent_id] = allocation_per_agent + (1 if i < remainder else 0)
                self._allocations[agent_id] = AllocationResult(
                    agent_id=agent_id,
                    allocated=result[agent_id],
                    used=0,
                    remaining=result[agent_id],
                    score=0.0
                )
            return result
        
        # Allocate proportionally to scores
        result: Dict[str, int] = {}
        allocated_total = 0
        
        # Calculate raw allocations (may not sum to budget due to rounding)
        raw_allocations = {}
        for agent_id in agents:
            raw_alloc = int((scores[agent_id] / total_score) * self.total_budget)
            raw_allocations[agent_id] = raw_alloc
        
        # Round to integers and handle remainder
        # Sort by score descending to distribute remainder to highest-scoring agents
        sorted_agents = sorted(agents, key=lambda a: scores[a], reverse=True)
        
        for agent_id in sorted_agents:
            result[agent_id] = raw_allocations[agent_id]
            allocated_total += result[agent_id]
        
        # Distribute any remainder (due to rounding) to top agents
        remainder = self.total_budget - allocated_total
        if remainder > 0:
            for agent_id in sorted_agents[:remainder]:
                result[agent_id] += 1
        
        # Store allocation results
        for agent_id in agents:
            self._allocations[agent_id] = AllocationResult(
                agent_id=agent_id,
                allocated=result[agent_id],
                used=0,
                remaining=result[agent_id],
                score=scores[agent_id]
            )
        
        return result
    
    def _normalize_values(
        self,
        values: Dict[str, float],
        agents: List[str]
    ) -> Dict[str, float]:
        """
        Normalize values to 0-1 range.
        
        If all values are equal (including all zeros), returns 0.5 for all.
        
        Args:
            values: Dictionary mapping agent_id to value
            agents: List of agent IDs to normalize
            
        Returns:
            Dictionary mapping agent_id to normalized value (0-1)
        """
        agent_values = [values[agent_id] for agent_id in agents]
        min_val = min(agent_values)
        max_val = max(agent_values)
        
        # Handle case where all values are equal
        if max_val == min_val:
            return {agent_id: 0.5 for agent_id in agents}
        
        # Normalize to 0-1
        range_val = max_val - min_val
        return {
            agent_id: (values[agent_id] - min_val) / range_val
            for agent_id in agents
        }
    
    def get_budget(self, agent_id: str) -> int:
        """
        Get the remaining budget for an agent.
        
        Args:
            agent_id: Agent ID to query
            
        Returns:
            Remaining compute units for the agent, or 0 if not allocated
            
        Raises:
            KeyError: If agent_id has not been allocated
        """
        if agent_id not in self._allocations:
            raise KeyError(f"Agent {agent_id} has not been allocated")
        
        return self._allocations[agent_id].remaining
    
    def get_allocation_result(self, agent_id: str) -> AllocationResult:
        """
        Get the full allocation result for an agent.
        
        Args:
            agent_id: Agent ID to query
            
        Returns:
            AllocationResult for the agent
            
        Raises:
            KeyError: If agent_id has not been allocated
        """
        if agent_id not in self._allocations:
            raise KeyError(f"Agent {agent_id} has not been allocated")
        
        return self._allocations[agent_id]
    
    def update_budget(self, agent_id: str, delta: int) -> None:
        """
        Update the budget for an agent by a delta amount.
        
        Positive delta increases remaining budget (adds to allocated).
        Negative delta decreases remaining budget (marks as used).
        
        Args:
            agent_id: Agent ID to update
            delta: Amount to change the budget (positive or negative)
            
        Raises:
            KeyError: If agent_id has not been allocated
            ValueError: If update would result in negative budget
        """
        if agent_id not in self._allocations:
            raise KeyError(f"Agent {agent_id} has not been allocated")
        
        allocation = self._allocations[agent_id]
        
        if delta >= 0:
            # Adding to budget
            new_allocated = allocation.allocated + delta
            new_remaining = allocation.remaining + delta
            new_used = allocation.used
            
            self._allocations[agent_id] = AllocationResult(
                agent_id=agent_id,
                allocated=new_allocated,
                used=new_used,
                remaining=new_remaining,
                score=allocation.score
            )
        else:
            # Using budget (negative delta)
            use_amount = -delta
            if use_amount > allocation.remaining:
                raise ValueError(
                    f"Cannot use {use_amount}, only {allocation.remaining} remaining for {agent_id}"
                )
            
            new_used = allocation.used + use_amount
            new_remaining = allocation.remaining - use_amount
            
            self._allocations[agent_id] = AllocationResult(
                agent_id=agent_id,
                allocated=allocation.allocated,
                used=new_used,
                remaining=new_remaining,
                score=allocation.score
            )
    
    def use_budget(self, agent_id: str, amount: int) -> None:
        """
        Mark compute units as used for an agent.
        
        This is a convenience method that calls update_budget with a negative delta.
        
        Args:
            agent_id: Agent ID to update
            amount: Number of compute units to mark as used
            
        Raises:
            KeyError: If agent_id has not been allocated
            ValueError: If amount exceeds remaining budget
        """
        self.update_budget(agent_id, -amount)
    
    def reset(self) -> None:
        """
        Reset the allocator to initial state.
        
        Clears all allocations and resets remaining total to the full budget.
        """
        self._allocations.clear()
        self._remaining_total = self.total_budget
    
    def get_all_allocations(self) -> Dict[str, AllocationResult]:
        """
        Get all allocation results.
        
        Returns:
            Dictionary mapping agent_id to AllocationResult
        """
        return dict(self._allocations)
    
    def total_allocated(self) -> int:
        """
        Get the total allocated compute units.
        
        Returns:
            Sum of all allocated compute units
        """
        return sum(a.allocated for a in self._allocations.values())
    
    def total_remaining(self) -> int:
        """
        Get the total remaining compute units across all agents.
        
        Returns:
            Sum of all remaining compute units
        """
        return sum(a.remaining for a in self._allocations.values())
    
    def total_used(self) -> int:
        """
        Get the total used compute units across all agents.
        
        Returns:
            Sum of all used compute units
        """
        return sum(a.used for a in self._allocations.values())
