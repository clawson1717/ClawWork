import hashlib
import json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class ReasoningPayload(BaseModel):
    """
    Represents a single node of reasoning within the Dynamic Interaction Graph (DIG).
    """
    source_id: str = Field(..., description="Unique identifier for the source agent or node.")
    content: str = Field(..., description="The actual reasoning text or data content.")
    state_hash: Optional[str] = Field(None, description="SHA-256 hash representing the state of this node.")
    semantic_vector: List[float] = Field(default_factory=list, description="Embedding vector for semantic drift analysis.")
    drift_score: float = Field(default=0.0, description="Calculated drift from the ground-truth intent.")

    @model_validator(mode="after")
    def ensure_hash(self) -> "ReasoningPayload":
        if not self.state_hash:
            self.state_hash = self.generate_hash()
        return self

    def generate_hash(self) -> str:
        """
        Generates a SHA-256 hash based on the content and source_id.
        """
        payload_data = {
            "source_id": self.source_id,
            "content": self.content
        }
        # Ensure stable serialization for hashing
        encoded = json.dumps(payload_data, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> str:
        """
        Serializes the payload to a JSON string.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningPayload":
        """
        Deserializes a JSON string into a ReasoningPayload instance.
        """
        return cls.model_validate_json(json_str)

    @field_validator("drift_score")
    @classmethod
    def validate_drift_score(cls, v: float) -> float:
        if v < 0:
            raise ValueError("drift_score must be non-negative")
        return v
