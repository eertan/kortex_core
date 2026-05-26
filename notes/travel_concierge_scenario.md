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
- competing HTN methods are filtered by hard preconditions and ranked by
  extracted preferences such as `style:relaxed` and `overly_touristy`
- planner detects missing required information if the domain requires origin
  city and budget

Representative method-selection candidates:

```yaml
htn_methods:
  - name: m_relaxed_local_trip
    target_task: plan_trip
    preference_matches: [relaxed, style:relaxed, overly_touristy]
    subtasks:
      - [search_boutique_hotels, destination, budget]
      - [search_low_density_activities, destination, duration_days]
      - [assemble_relaxed_itinerary, destination, duration_days]

  - name: m_standard_first_time_trip
    target_task: plan_trip
    preference_matches: [first_time, landmarks]
    subtasks:
      - [search_hotels, destination, budget]
      - [search_major_attractions, destination, duration_days]
      - [assemble_standard_itinerary, destination, duration_days]
```

If both methods remain equally preferred after precondition filtering and
preference scoring, Kortex should produce a tie impasse instead of silently
using manifest order.

For this demo, prefer `subtasks` over `ordered_subtasks` when the order can be
derived from primitive preconditions and effects. That lets the HTN layer choose
the strategy while the PDDL/classical planner determines the legal execution
order inside that strategy. Keep `ordered_subtasks` for learned chunks or
procedures where sequence is part of the skill itself.

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

## Implementation Status

Completed:

1. Created a multi-file package at `scenarios/domains/travel_concierge/`:
   - `domain.yaml`
   - `intents.yaml`
   - `decisions.yaml`
   - `responses.yaml`
2. Added fake travel plugins in `scenarios/travel_concierge.py`.
3. Added reusable scenario harness helpers in `scenarios/harness.py`.
4. Added a travel smoke test for planning, HITL approval, working-memory facts,
   selected method notes, and structured JSON logs.
5. Added generic optimization through `KortexOptimizer`:
   - mock flight and hotel inventories include names, prices, locations,
     schedule metadata, tags, and preference-fit scores
   - search plugins record those inventories as `ExternalKnowledgePayload`
     memory records and working memory references them without promoting them
     to planner truth
   - flight and hotel candidate bundles are enumerated from the inventories
   - hard constraints reject non-refundable or over-budget bundles
   - weighted objectives score total cost, preference fit, and boutique match
   - the selected bundle is recorded as an `optimization_decision` memory record
   - planner truth only sees the promoted symbolic effect
     `reservation_group_selected(destination)`
6. Added guarded response rendering:
   - optimizer summaries can use natural narration through `ResponseRenderer`
   - forbidden claims such as confirmed booking/payment are blocked
   - deterministic templates remain available as fallback
   - travel demo logs `response.optimizer_summary` before HITL approval

Run:

```bash
.venv/bin/python -m scenarios.travel_concierge --approval y
```

Remaining:

1. Build a config-aware interaction session, likely in
   `kortex/configured_interaction.py`, before adding UI.
2. Add interaction-level travel turns for:
   - greeting/conversation-only turn
   - out-of-domain rejection
   - missing origin/budget clarification
   - clarification resumption
3. Replace the current logged memory note with a real semantic-memory retrieval
   adapter or seeded typed memory records.
4. Use `intents.yaml` to drive a real interaction-level travel flow rather than
   the current scripted scenario prompt.
5. Refine the travel type system so `duration_days`, `style`, and `budget`
   are not represented as domain object types unless that remains intentional
   for planner grounding.

## Config-Aware Interaction Session Spec

The travel demo should not get a travel-specific chat loop. It should exercise
a reusable config-aware interaction layer that can serve any domain package.

Proposed module:

```text
kortex/configured_interaction.py
```

Proposed runtime object:

```text
ConfiguredInteractionSession
```

Inputs:

- loaded `DomainPackage`
- `IntentFrameBuilder`
- `ResponseRenderer`
- existing `KortexAgent` or lower-level planner/execution bridge
- optional structured LLM interpreter for mapping user language into intent
  names and raw slots
- memory sink for conversation, optimizer decisions, traces, and external
  option records

Turn flow:

```text
user turn
  -> persist conversation memory
  -> deterministic unsafe-directive check
  -> domain scope check from intents.yaml
  -> classify conversation/task/clarification
  -> build or resume IntentFrame
  -> if missing slots: return configured clarification
  -> if complete: create planner goal through domain.yaml intent_bindings
  -> run planner/executor/optimizer
  -> render response through responses.yaml and ResponseRenderer
  -> persist trace and memory records
```

Required behavior for travel:

1. User greeting:
   - no planner call
   - response is friendly
   - conversation memory is persisted
2. Out-of-domain request:
   - no planner call
   - response uses configured `out_of_domain` response
3. In-domain incomplete request:
   - `IntentFrameBuilder` returns `IntentClarification`
   - pending slots are stored
   - configured clarification asks for origin and budget
4. Clarification answer:
   - merges new slot values into pending state
   - rebuilds complete `IntentFrame`
   - normalizes values into planner objects
5. Execution:
   - planner uses HTN method selection and Pyperplan ordering
   - optimizer summary is rendered before HITL
   - HITL approval/denial is reflected in final response

Important policy:

- LLM/interpreter may only produce intent names and raw slots.
- Planner truth may only come from validated planner facts and declared action
  effects.
- Response narration may only phrase validated facts from `ResponseFrame`.
- Static templates remain mandatory for refusal, clarification, HITL, and
  failure cases.
