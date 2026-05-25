# Travel Concierge Scenario Spec

## Purpose

This scenario is intended as a rich demo for the interaction layer, memory
architecture, deterministic planning, HITL gates, semantic recommendations, and
fake external APIs.

Working title:

```text
Kortex Travel Concierge
```

Core thesis:

```text
A deterministic travel-planning assistant can hold a friendly conversation,
stay inside its travel domain, use remembered preferences and destination
knowledge for recommendations, clarify missing trip details, plan with HTN
methods, call fake booking/search APIs, and require human approval before
booking or placing holds.
```

## Why This Use Case

Trip organization is a good Kortex demo because it naturally requires:

- pleasant conversation before task execution
- clear domain boundaries
- semi-grounded user requests
- clarification for missing slots
- memory-backed recommendations
- semantic destination knowledge
- fake but realistic API plugins
- HITL approval for booking or financial actions
- working-memory updates across multiple turns
- validated trace memory after successful execution

## Domain Boundary

The agent can help with:

- trip planning
- itinerary construction
- flight search
- hotel search
- local experience search
- restaurant or activity recommendations
- calendar holds
- refundable reservation holds
- approval-gated final booking

The agent should reject or redirect out-of-domain requests, such as:

- churn modeling
- legal advice
- medical advice
- unrelated company analytics
- unsafe automation
- actions outside the fake travel API/domain manifest

Example response:

```text
I can't handle that in the travel planning domain. I can help with trip
planning, itinerary options, reservations, and travel logistics.
```

## Conversation Flow

### Turn 1: Greeting

User:

```text
Hey, good morning.
```

Expected behavior:

- interaction layer classifies as conversation
- no planner call
- user turn is persisted as `ConversationMemoryPayload`
- assistant response is friendly but does not claim task execution

Assistant:

```text
Good morning. What trip are we working on today?
```

### Turn 2: Out-of-Scope Request

User:

```text
Can you analyze my company's Q4 churn risk?
```

Expected behavior:

- interaction policy or interpreter marks this outside travel domain
- no `KortexAgent` task execution
- assistant explains boundary
- turn is persisted as conversation memory

Assistant:

```text
I can't handle that in the travel planning domain. I can help with trip
planning, itinerary options, reservations, and travel logistics.
```

### Turn 3: Semi-Grounded Travel Request

User:

```text
I need to plan a three-day trip to Rome next month. Keep it relaxed, not too
touristy.
```

Interaction interpretation candidate:

```json
{
  "turn_type": "task",
  "task_prompt": "Plan a relaxed three-day trip to Rome next month with non-touristy activities.",
  "candidate_entities": ["Rome", "next month", "relaxed", "non-touristy"],
  "candidate_directives": [],
  "memory_notes": []
}
```

Extractor output candidate:

```json
{
  "root_task_name": "plan_trip",
  "task_parameters": {
    "destination": "Rome",
    "duration_days": 3,
    "travel_window": "next_month",
    "style": "relaxed",
    "avoid": ["overly_touristy"]
  }
}
```

Expected behavior:

- `InteractionSession` treats this as a task
- `KortexAgent` extracts named parameters
- domain `intent_bindings` maps parameters into an HTN task
- planner detects missing required information if the domain requires origin
  city and budget

### Turn 4: Clarification

Assistant:

```text
What city are you departing from, and what budget range should I stay within?
```

User:

```text
Departing from Boston. Around $2500 excluding food.
```

Expected behavior:

- `InteractionSession` resumes the pending prompt
- resumed prompt includes original request plus clarification answer
- `KortexAgent` re-runs extraction/planning
- extracted parameters now include origin and budget

Resumed prompt shape:

```text
I need to plan a three-day trip to Rome next month. Keep it relaxed, not too touristy.
Clarification answer: Departing from Boston. Around $2500 excluding food.
```

### Turn 5: Memory-Backed Recommendation

Relevant remembered preferences:

```text
User prefers boutique hotels.
User dislikes packed schedules.
User previously enjoyed food markets and architecture walks.
```

Relevant semantic destination knowledge:

```text
Trastevere is good for evening dining.
Monti is central and quieter than the Spanish Steps area.
Testaccio is strong for food-market oriented experiences.
```

Expected assistant behavior:

```text
Based on your preference for relaxed trips and boutique hotels, I'll bias the
hotel search toward Monti or Trastevere and keep the itinerary lighter than a
standard first-time tourist route.
```

Important constraint:

```text
Retrieved semantic/conversation memory can inform recommendations, but it does
not become planner truth unless explicitly validated/promoted.
```

## Candidate Domain Manifest Shape

Sketch:

```yaml
domain_name: travel_concierge
types:
  - City
  - Airport
  - Hotel
  - Neighborhood
  - Activity
  - Budget
  - TravelWindow
  - Itinerary

fluents:
  trip_profile_complete:
    signature: { destination: City }
  flight_options_found:
    signature: { origin: City, destination: City }
  hotel_options_found:
    signature: { destination: City }
  itinerary_built:
    signature: { destination: City }
  flight_hold_created:
    signature: { destination: City }
  hotel_hold_created:
    signature: { destination: City }
  booking_finalized:
    signature: { destination: City }

intent_bindings:
  plan_trip:
    type: task
    task: build_trip_plan
    required_parameters:
      - origin
      - destination
      - duration_days
      - travel_window
      - budget
      - style
    args:
      - origin
      - destination
      - duration_days
      - travel_window
      - budget
      - style

htn_methods:
  - name: m_build_trip_plan
    target_task: build_trip_plan
    parameters:
      origin: City
      destination: City
      duration_days: Budget
      travel_window: TravelWindow
      budget: Budget
      style: Budget
    ordered_subtasks:
      - ["search_flights", "origin", "destination", "travel_window", "budget"]
      - ["search_hotels", "destination", "budget", "style"]
      - ["search_local_experiences", "destination", "style"]
      - ["build_itinerary", "destination", "duration_days", "style"]
```

Note: the type sketch above needs refinement. `duration_days` and `style` should
probably become first-class typed values or structured task parameters rather
than overloading `Budget`.

## Candidate Fake Plugins

Safe/non-approval plugins:

```text
search_flights(origin, destination, travel_window, budget)
search_hotels(destination, budget, style)
search_local_experiences(destination, style)
build_itinerary(destination, duration_days, style)
create_calendar_hold(destination, travel_window)
```

Approval-gated plugins:

```text
reserve_flight_hold(origin, destination, selected_flight)
reserve_hotel_hold(destination, selected_hotel)
book_flight(origin, destination, selected_flight)
book_hotel(destination, selected_hotel)
```

Booking and payment-like actions must use `requires_approval=True`.

## HITL Approval Moment

Assistant:

```text
I found a Boston to Rome flight for $780 and a boutique hotel in Monti for
$310/night. I can place refundable holds for both. Do you approve?
```

Expected behavior:

- `reserve_flight_hold` and `reserve_hotel_hold` require approval
- denial stops execution and records HITL denial
- approval allows execution and records validated trace

## Expected Final Response

If approved:

```text
I placed refundable holds for the flight and hotel. I also created a relaxed
three-day itinerary centered around Monti, Trastevere, food markets, and one
architecture walk.
```

If not approved:

```text
I did not place any holds. The proposed itinerary and options are saved for
review.
```

## Cognitive Features Exercised

- conversation memory
- semantic memory retrieval
- working-memory continuity
- named intent bindings
- clarification and resumption
- deterministic HTN execution
- fake external API plugins
- HITL approval
- pre-response guard
- validated trace memory
- domain boundary enforcement

## Implementation Tasks For Tomorrow

1. Create a travel scenario domain manifest.
2. Add fake travel plugins with a local registry.
3. Add an interaction-level scenario runner.
4. Add tests for:
   - greeting/conversation-only turn
   - out-of-domain rejection
   - missing origin/budget clarification
   - clarification resumption
   - memory-backed recommendation note
   - HITL approval for holds/bookings
   - final working-memory facts
5. Decide how semantic memory is mocked:
   - seeded `MemoryRecord` entries
   - fake Graphiti retrieval adapter
   - simple in-memory semantic preference provider
6. Refine the travel type system so `duration_days`, `style`, and `budget`
   are not awkwardly represented as domain object types unless needed.

