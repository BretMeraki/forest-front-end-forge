from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime

class DesireBase(BaseModel):
    """Desire Base model."""
    id: str
    strength: float
    category: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('strength')
    def strength_must_be_0_to_1(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('strength must be between 0 and 1')
        return v

class FinancialMetricsBase(BaseModel):
    """Financial Metrics Base model."""
    user_id: str
    score: float
    last_updated: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('user_id')
    def user_id_must_not_be_empty(cls, v):
        if v is None or v.strip() == "":
            raise ValueError("user_id must not be empty")
        return v

class HTANodeBase(BaseModel):
    id: str
    title: str
    parent_id: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if v is None or v.strip() == "":
            raise ValueError("title must not be empty")
        return v

class PatternBase(BaseModel):
    """Pattern Base model."""
    id: str
    pattern_type: str
    confidence: float
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('pattern_type')
    def pattern_type_must_not_be_empty(cls, v):
        if v is None or v.strip() == "":
            raise ValueError("pattern_type must not be empty")
        return v

    @validator('confidence')
    def confidence_must_be_nonnegative(cls, v):
        if v < 0:
            raise ValueError("confidence must be non-negative")
        return v
