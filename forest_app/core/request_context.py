"""RequestContext for in-process context propagation."""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """
    Context object for request-scoped information propagation.
    
    Contains user_id, trace_id, timestamp and feature flags to be
    passed through service layers and included in logs.
    """
    user_id: Optional[UUID] = None
    trace_id: UUID = Field(default_factory=uuid4)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    
    model_config = {
        "frozen": True,
        "extra": "forbid",
    }
    
    @lru_cache(maxsize=128)
    def has_feature(self, feature_name: str) -> bool:
        """
        Check if a feature flag is enabled.
        
        Cached for performance as this may be called frequently.
        """
        return self.feature_flags.get(feature_name, False)
