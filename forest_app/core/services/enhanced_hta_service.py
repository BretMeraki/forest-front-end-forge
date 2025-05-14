"""
Enhanced HTAService with Dynamic HTA Generation Framework

This module implements the dynamic backend framework for HTA tree generation that ensures
a perfect balance between performance, personalization, and alignment with the PRD's vision.

Key features:
- Schema contract approach (not templates) that defines structure without dictating content
- Context-infused node generation that creates unique, personalized content
- Performance optimizations like bulk operations and denormalized fields
- Transaction management to ensure data integrity
- Cache management to reduce latency
- Positive reinforcement system integrated with task completion

This implementation aligns with the PRD's core vision: "Remind the user why being alive
is a beautiful and precious experience" by creating a truly personal and engaging experience.
"""

# Import the modularized EnhancedHTAService
from forest_app.core.services.enhanced_hta.core import EnhancedHTAService
