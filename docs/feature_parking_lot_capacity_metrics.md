# Feature Parking Lot: Capacity Metrics System

## Overview

This document captures feature ideas for a progressive capacity development system inspired by fitness applications like Fitbod. The system aims to create psychological safety for users by providing metrics that decay gracefully during breaks while offering targeted activities to build specific capabilities.

## Core Framework

### Primary Metrics

1. **Overall: Productivity Capacity (0-100)**
   - Composite score of all dimensions
   - Decay rate: ~3-4% monthly (slow)
   - Primary indicator of a user's readiness for challenging tasks

2. **Planning Capacity (0-100)**
   - Measures organizational and structural skills
   - Decay rate: ~4% monthly
   - Data sources: Task breakdown quality, timeline creation, roadmap development

3. **Focusing Capacity (0-100)**
   - Measures concentrated attention abilities
   - Decay rate: ~5% monthly
   - Data sources: Session duration, task completion without switching, depth metrics

4. **Learning Capacity (0-100)**
   - Measures information absorption and integration
   - Decay rate: ~3% monthly
   - Data sources: Knowledge connections, reflection quality, conceptual integration

5. **Executing Capacity (0-100)**
   - Measures ability to complete and deliver
   - Decay rate: ~6% monthly
   - Data sources: Completion rates, implementation quality, delivery consistency

### Benchmark Achievements

- Personal best in each category
- Overall top accomplishment
- Never decays or disappears

## Dual Activity System

### 1. Goal Activities (HTA Tree)
- Actual project objectives and deliverables
- Generated based on project needs and roadmap
- Affect metrics as a byproduct of completion
- Examples: "Implement CompletionProcessor", "Design database schema"

### 2. Conditioning Activities
- Scientifically designed capability-building exercises
- Generated based on behavioral psychology research
- Specifically optimized to build particular capacities
- Examples: "30-minute focus sprint", "Decision matrix exercise"

## Technical Implementation

### 1. Metrics Calculation Engine

```python
class CapacityMetricsEngine:
    def calculate_metrics(self, user_id):
        # Calculate each capacity dimension
        planning = self._calculate_planning_capacity(user_id)
        focusing = self._calculate_focusing_capacity(user_id)
        learning = self._calculate_learning_capacity(user_id)
        executing = self._calculate_executing_capacity(user_id)
        
        # Calculate overall capacity
        overall = (planning * 0.25 + focusing * 0.25 + 
                   learning * 0.25 + executing * 0.25)
        
        return {
            "overall": overall,
            "planning": planning,
            "focusing": focusing,
            "learning": learning,
            "executing": executing
        }
        
    def apply_decay(self, user_id, days_inactive):
        # Apply appropriate decay to each metric
        # Ensure decay is noticeable but not punishing
```

### 2. Conditioning Activity Generator

```python
class ConditioningActivityGenerator:
    def generate_activities(self, user_id):
        # Get current metrics
        metrics = self.metrics_service.get_metrics(user_id)
        
        # Identify dimensions to prioritize
        priority_dimensions = self._identify_priority_dimensions(metrics)
        
        # Generate appropriate conditioning activities
        # Select from library of research-backed exercises
```

### 3. Task Suggestion Engine

```python
class TaskSuggestionEngine:
    def get_suggestions(self, user_id):
        # Get current metrics and preferences
        metrics = self.metrics_service.get_metrics(user_id)
        preferences = self.user_service.get_preferences(user_id)
        
        # Calculate appropriate challenge level
        challenge_level = self._calculate_challenge_level(metrics, preferences)
        
        # Generate balanced set of suggestions
        # Mix goal tasks and conditioning activities
```

## Data Collection Requirements

1. **Session Timing Data**
   - Task start/end timestamps
   - App usage durations
   - Time between activities

2. **Task Quality Metrics**
   - Completion status
   - Complexity indicators
   - Depth of detail
   - Dependency structures

3. **User Interaction Patterns**
   - Navigation flow
   - Feature usage
   - Task switching frequency
   - Revisit patterns

## Conditioning Activities Library

### Planning Conditioning Activities

- **Decision Matrix Exercise**
  - Create a structured decision framework
  - Builds analytical planning skills
  - Estimated gain: +2-4 Planning points

- **Project Breakdown Challenge**
  - Break complex concept into actionable steps
  - Builds hierarchical thinking
  - Estimated gain: +3-5 Planning points

- **Timeline Construction**
  - Create realistic timeline with dependencies
  - Builds sequential planning skills
  - Estimated gain: +2-4 Planning points

### Focusing Conditioning Activities

- **Deep Work Sprint**
  - Timed, focused implementation session
  - Builds concentration endurance
  - Estimated gain: +3-6 Focusing points

- **Distraction Defense**
  - Complete task while managing controlled interruptions
  - Builds attention management
  - Estimated gain: +2-4 Focusing points

- **Single-Task Challenge**
  - Complete task without switching contexts
  - Builds sustained attention
  - Estimated gain: +2-5 Focusing points

### Learning Conditioning Activities

- **Concept Mapping**
  - Create visualization of related concepts
  - Builds information integration skills
  - Estimated gain: +2-4 Learning points

- **Reflection Protocol**
  - Structured review of completed work
  - Builds metacognition skills
  - Estimated gain: +3-5 Learning points

- **Knowledge Application**
  - Apply learning from one context to another
  - Builds transfer skills
  - Estimated gain: +2-4 Learning points

### Executing Conditioning Activities

- **Implementation Sprint**
  - Rapid execution of defined task
  - Builds completion momentum
  - Estimated gain: +3-6 Executing points

- **Obstacle Navigation**
  - Complete task despite controlled friction
  - Builds resilience
  - Estimated gain: +2-4 Executing points

- **Output Challenge**
  - Produce specific deliverable under constraints
  - Builds production quality
  - Estimated gain: +3-5 Executing points

## UI Concepts

### Dashboard View

```
PRODUCTIVITY CAPACITY: 76%
Your overall capability remains strong.

PLANNING: 82% ↑
Your ability to organize work is excellent.

FOCUSING: 70% ↑
Your ability to concentrate is building steadily.

LEARNING: 75% ↔
Your ability to integrate knowledge is stable.

EXECUTING: 74% ↑
Your ability to complete tasks is strengthening.

PERSONAL BESTS:
Most complex project: "Task Completion System"
Longest focus session: 85 minutes (Apr 15)
```

### Activity Selection View

```
TODAY'S RECOMMENDED ACTIVITIES

CONDITIONING:
• "30-minute deep focus sprint" (Builds Focus +5)
  A timed, distraction-free implementation session

• "Decision matrix creation" (Builds Planning +3)
  Apply structured decision framework to an issue

GOALS:
• "Design notification system database schema"
  Part of your Database Architecture project

• "Implement user preference storage"
  Part of your User Settings feature
```

### Return After Break Experience

```
Welcome back! Your productivity foundation remains strong.

Your Productivity Capacity is 72% (was 78%)
Your planning capacity remains particularly strong (80%)

We've prepared an ideal re-entry activity:

"Review project status and next steps" (20 min)
→ Reconnects you with your work
→ Builds: Planning and Focus capabilities
```

## Future Expansion Ideas

1. **Adaptive Decay Rates**
   - Personalize decay rates based on historical patterns
   - Learn how quickly each user's capabilities actually decay

2. **Capacity Specialization**
   - Allow users to develop "specialist" profiles
   - Support different work styles (planner, executor, etc.)

3. **Team Capacity Metrics**
   - Aggregate metrics across team members
   - Identify team-level strengths and growth areas

4. **Capacity Milestone Celebrations**
   - Recognize significant milestones in capability development
   - Create achievement system for capacity growth

5. **AI-Generated Custom Activities**
   - Use AI to create personalized conditioning activities
   - Tailored to specific user preferences and patterns

6. **Integration with Time Tracking**
   - Connect capacity development to time investment
   - Show ROI of conditioning activities

7. **Environmental Factors**
   - Account for external factors affecting capacity
   - Time of day, week, month impacts on performance

## Implementation Prioritization

### Phase 1: Core Metrics
- Implement the four capacity metrics
- Develop data collection framework
- Create basic visualization

### Phase 2: Conditioning Activities
- Implement initial library of activities
- Create activity suggestion engine
- Connect activities to metric growth

### Phase 3: Intelligent Adaptation
- Implement personalized decay rates
- Develop challenge level calibration
- Create full return experience

### Phase 4: Advanced Features
- Implement capacity specialization
- Develop team metrics
- Create AI-generated custom activities
