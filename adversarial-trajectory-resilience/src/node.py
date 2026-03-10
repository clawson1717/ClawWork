from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class NodeStatus(Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    PRUNED = "PRUNED"
    SUCCESS = "SUCCESS" # Added for terminal leaf detection

@dataclass
class TrajectoryNode:
    id: str
    content: str
    status: NodeStatus = NodeStatus.PENDING
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
