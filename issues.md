Issues

  1. Grounding is too late
      - The agent accepts Japan as a destination candidate, continues asking other missing slots, and only later says it cannot ground Japan.
      - Better behavior: detect ungrounded critical entities immediately and ask:

        I can help with Japan, but I need a specific city. Which city do you want to visit?

  2. Grounding is too shallow
      - Current grounding is mostly string normalization plus checking whether the normalized value exists in planner objects.
      - This is not enough for country/city, relative date, airport/city, budget, duration, etc.
      - We need a real grounding layer between extraction and intent-frame completion.
  3. Critical fields need constrained schemas
      - Your observation is right: for critical fields, Pydantic models with Literal / enum-like allowed values usually work better.
      - For the travel demo, fields like destination_city, origin_city, travel_window, and maybe trip_duration_days should be constrained or validated against configured domain
        values.
      - The LLM should not freely return Japan as a city if the slot requires a supported city.
  4. Country-level destination needs a different slot or clarification
      - Japan is valid user intent, but not valid planner grounding for destination: City.
      - Options:
          - Add separate raw extraction fields like destination_region and destination_city.
          - Or require destination_city: Literal[...] and let the LLM return missing/unknown if only a country was supplied.
      - Then deterministic logic asks for city immediately.
  5. Relative date grounding is incomplete
      - two days from now, 2 days later, leave in 2 days, next Friday, etc. are not normalized consistently.
      - Need a date/window grounding component:

        raw: "two days from now"
        grounded: "in_2_days"
        confidence: high

      - If unsupported, clarify.

  6. The agent asks all missing questions before resolving bad provided values
      - Current priority is mostly: collect required slots, then validate grounding.
      - Better priority:
          1. Validate already-provided critical slots.
          2. If any are ungrounded, ask about those first.
          3. Only then ask for other missing slots.
  7. Budget constraint was violated
      - User said 1500$.
      - Optimizer selected a bundle with total 1820.
      - That should either:
          - reject the bundle and report no feasible option,
          - ask whether the user wants to increase budget,
          - or find a cheaper alternative.
      - It must not say “within your budget” unless the constraint actually passed.
  8. Optimizer summary overclaims
      - The response said “stays within your budget” when it did not.
      - Response rendering must be tied to actual optimizer feasibility and selected attributes, not a fixed narrator phrase.
  9. Approval denial recovery is weak
      - no currently stops execution, but the assistant only says it stopped.
      - Better:

        I stopped before placing holds. What would you like to change: budget, dates, duration, hotel preference, or destination?

  10. “No, this is above my budget” should be interpreted as denial plus correction

  - This is not just approval denial.
  - It also contains the reason: selected bundle violates budget expectations.
  - The session should capture this as a correction and either replan or ask a follow-up.

  11. Change requests during HITL need a recovery path

  - “I changed my mind, want to stay 5 days” should cancel the pending approval and reopen the intent with updated duration_days.
  - Current behavior treats it as denial only or repeats approval.
  - Desired behavior:

    I canceled the pending holds. I’ll update the trip to 5 days and re-check options.

  12. No explicit correction model

  - We need a turn type beyond conversation, task, clarification_answer.
  - Likely additions:
      - approval_response
      - correction
      - cancel
      - change_request
  - These should be structured outputs from the interaction interpreter.

  13. No grounded-value confidence or status

  - A slot should not just be present/missing.
  - It should have status:

    missing | extracted | grounded | ambiguous | unsupported

  - That would make the control flow much cleaner.

  14. Planner object inventory is too implicit

  - The LLM/interpreter needs to know supported values for critical slots.
  - These can come from config:

    destination:
      slot_type: City
      grounding:
        allowed_values: [tokyo]
        require_explicit_value: true

  - Or from a dynamic grounding provider later.

  15. Need typed grounding layer before IntentFrameBuilder

  - Proposed pipeline:

    user text
      -> LLM extracts raw candidates
      -> grounding layer validates/maps candidates against config/domain inventory
      -> unresolved values produce clarification
      -> complete grounded frame enters planner

  Key Design Note
  Your instinct about Pydantic/Literal lists is correct. For demo-critical fields, we should make the LLM output stricter:

  destination_city: Literal["tokyo"] | None
  origin_city: Literal["boston", "new_york"] | None
  travel_window: Literal["in_2_days", "next_week", "next_month"] | None

  Then separately allow a raw field:

  destination_mention: str | None

  So if the user says “Japan”, the model can preserve it as destination_mention="Japan" but leave destination_city=None, causing immediate grounding clarification.
