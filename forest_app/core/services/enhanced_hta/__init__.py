"""Enhanced HTA Service - A modular implementation.

This package contains well-structured components that together form the
Enhanced HTA Service, organized in a clean, maintainable architecture.

Modules:
- core: The main service class and core HTA operations
- memory: Semantic and episodic memory integration
- reinforcement: Positive feedback and motivation generation
- events: Event publishing and notification handling
- background: Asynchronous task processing
- utils: Common utility functions
"""

# Core service
from forest_app.core.services.enhanced_hta.core import EnhancedHTAService

# Component managers
from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
from forest_app.core.services.enhanced_hta.reinforcement import ReinforcementManager
from forest_app.core.services.enhanced_hta.events import EventManager
from forest_app.core.services.enhanced_hta.background import BackgroundTaskManager

# Utility functions
from forest_app.core.services.enhanced_hta.utils import (
    format_uuid,
    safe_serialize,
    get_now,
    Result
)

__all__ = [
    # Core service
    'EnhancedHTAService',
    
    # Component managers
    'HTAMemoryManager',
    'ReinforcementManager',
    'EventManager',
    'BackgroundTaskManager',
    
    # Utility functions
    'format_uuid',
    'safe_serialize',
    'get_now',
    'Result'
]
