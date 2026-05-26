"""CLI entrypoint for the config-aware travel interaction demo."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel
from unified_planning.shortcuts import get_environment

from kortex.configured_interaction import (
    ConfiguredInteractionSession,
    ConfiguredInteractionTurnResult,
    ConfiguredTurnInterpretation,
    ConfiguredTurnInterpreter,
)
from kortex.configured_interpreter import GeminiConfiguredTurnInterpreter
from kortex.domain_package import DomainPackage, DomainPackageLoader
from kortex.intent_runtime import IntentClarification
from kortex.memory.records import MemoryRecord
from kortex.memory.working import WorkingMemoryState
from kortex.responses import ResponseRenderer
from scenarios.travel_concierge import (
    TRAVEL_DECISIONS,
    TRAVEL_PACKAGE_PATH,
    TravelDemoNarrator,
    build_registry,
)


TRAVEL_OBJECTS = {
    "boston": "City",
    "new_york": "City",
    "tokyo": "City",
    "in_2_days": "TravelWindow",
    "next_week": "TravelWindow",
    "next_month": "TravelWindow",
    "budget_2000": "Budget",
    "budget_2500": "Budget",
    "duration_3_days": "TripDuration",
    "duration_4_days": "TripDuration",
    "duration_5_days": "TripDuration",
    "relaxed": "TravelStyle",
}


class TranscriptMemorySink:
    """Collect memory records emitted during the demo conversation."""

    def __init__(self) -> None:
        """Initialize an empty memory record buffer."""
        self.records: list[MemoryRecord] = []

    def hook_memory_record(self, record: MemoryRecord) -> None:
        """Capture one emitted memory record."""
        self.records.append(record)


class TravelDemoInterpreter:
    """Deterministic interpreter for the travel interaction demo."""

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
        package: DomainPackage,
        pending_clarification: IntentClarification | None,
    ) -> ConfiguredTurnInterpretation:
        """Map a small travel-demo vocabulary into configured intent slots."""
        del working_memory, package
        normalized = user_text.lower()
        if pending_clarification is not None:
            return ConfiguredTurnInterpretation(
                turn_type="clarification_answer",
                intent_name=pending_clarification.intent_name,
                raw_slots=self._travel_slots(normalized),
            )
        if self._is_conversation(normalized):
            return ConfiguredTurnInterpretation(
                turn_type="conversation",
                response_text="Hello. I can help with travel planning.",
            )
        if self._is_travel_request(normalized):
            return ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots=self._travel_slots(normalized),
            )
        return ConfiguredTurnInterpretation(
            turn_type="task",
            intent_name="plan_trip",
            raw_slots={},
        )

    def _is_conversation(self, normalized: str) -> bool:
        """Return whether a turn is a non-task greeting or pleasantry."""
        return normalized.strip() in {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "thanks",
            "thank you",
        }

    def _is_travel_request(self, normalized: str) -> bool:
        """Return whether text is likely an in-domain travel request."""
        return any(
            token in normalized
            for token in [
                "trip",
                "travel",
                "flight",
                "hotel",
                "visit",
                "japan",
                "tokyo",
            ]
        )

    def _travel_slots(self, normalized: str) -> dict[str, Any]:
        """Extract deterministic demo slots from a travel request."""
        slots: dict[str, Any] = {
            "destination": self._destination(normalized),
            "duration_days": self._duration_days(normalized),
            "travel_window": self._travel_window(normalized),
            "style": "relaxed" if "relaxed" in normalized else "relaxed",
        }
        if "boston" in normalized:
            slots["origin"] = "boston"
        elif "new york" in normalized or "nyc" in normalized:
            slots["origin"] = "new_york"
        budget = self._budget(normalized)
        if budget is not None:
            slots["budget"] = budget
        return {
            slot_name: value
            for slot_name, value in slots.items()
            if value is not None
        }

    def _destination(self, normalized: str) -> str | None:
        """Extract a demo destination."""
        if "tokyo" in normalized or "japan" in normalized:
            return "tokyo" if "tokyo" in normalized else "japan"
        return None

    def _duration_days(self, normalized: str) -> int | None:
        """Extract a demo trip duration."""
        if "three" in normalized:
            return 3
        match = re.search(r"\b(\d+)\s*day\w*\b", normalized)
        if match is None:
            return None
        return int(match.group(1))

    def _travel_window(self, normalized: str) -> str | None:
        """Extract a demo travel window."""
        if "next week" in normalized:
            return "next_week"
        if "next month" in normalized:
            return "next_month"
        match = re.search(r"\b(\d+)\s*day\w*\s+from\s+now\b", normalized)
        if match is not None:
            return f"in_{match.group(1)}_days"
        match = re.search(r"\b(?:leave|depart|travel)\s+in\s+(\d+)\s*day\w*\b", normalized)
        if match is not None:
            return f"in_{match.group(1)}_days"
        return None

    def _budget(self, normalized: str) -> int | None:
        """Extract a whole-dollar budget from common demo phrasings."""
        match = re.search(r"(?:\$|\b)(\d{3,6})(?:\s*(?:\$|dollars|usd))?", normalized)
        if match is None:
            return None
        return int(match.group(1))


async def run_scripted_travel_conversation(
    log_path: Path | None = None,
    approval: str = "approve",
    interpreter_mode: str = "deterministic",
) -> list[dict[str, Any]]:
    """Run the deterministic scripted travel conversation."""
    get_environment().credits_stream = None
    TRAVEL_DECISIONS.clear()
    approval_turns = ["yes please", "approve it"] if approval == "approve" else ["no"]
    turns = [
        "hello",
        "I need trip planning for Tokyo next month for three days",
        "Boston, and keep it under 2500 dollars",
        *approval_turns,
    ]
    transcript = await run_turns(turns, interpreter_mode=interpreter_mode)
    if log_path is not None:
        write_transcript(log_path, transcript)
    return transcript


async def run_turns(
    turns: list[str],
    interpreter_mode: str = "deterministic",
) -> list[dict[str, Any]]:
    """Run configured travel interaction turns and return a JSON-safe transcript."""
    get_environment().credits_stream = None
    package = DomainPackageLoader().load(TRAVEL_PACKAGE_PATH)
    memory_sink = TranscriptMemorySink()
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=build_registry(package),
        interpreter=_build_interpreter(interpreter_mode),
        renderer=ResponseRenderer(narrator=TravelDemoNarrator()),
        memory_sink=memory_sink,
        interactive_execution=False,
        session_id="travel_interaction_demo",
        user_id="demo_user",
    )
    transcript: list[dict[str, Any]] = []
    for turn in turns:
        result = await session.handle_turn(turn)
        transcript.append(_turn_entry(turn, result))
    transcript.append(
        {
            "type": "memory_summary",
            "records": [_jsonable(record) for record in memory_sink.records],
        }
    )
    return transcript


async def run_interactive_travel_conversation(
    log_path: Path | None = None,
    interpreter_mode: str = "gemini",
) -> None:
    """Run an interactive terminal travel conversation."""
    get_environment().credits_stream = None
    package = DomainPackageLoader().load(TRAVEL_PACKAGE_PATH)
    memory_sink = TranscriptMemorySink()
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=build_registry(package),
        interpreter=_build_interpreter(interpreter_mode),
        renderer=ResponseRenderer(narrator=TravelDemoNarrator()),
        memory_sink=memory_sink,
        interactive_execution=False,
        session_id="travel_interaction_interactive",
        user_id="demo_user",
    )
    transcript: list[dict[str, Any]] = []
    print("Kortex travel demo. Type 'quit' to exit.")
    while True:
        user_text = input("user> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        result = await session.handle_turn(user_text)
        transcript.append(_turn_entry(user_text, result))
        print(f"kortex> {result.response_text}")
    transcript.append(
        {
            "type": "memory_summary",
            "records": [_jsonable(record) for record in memory_sink.records],
        }
    )
    if log_path is not None:
        write_transcript(log_path, transcript)


def _build_interpreter(interpreter_mode: str) -> ConfiguredTurnInterpreter:
    """Build the configured turn interpreter for the CLI."""
    if interpreter_mode == "deterministic":
        return TravelDemoInterpreter()
    if interpreter_mode == "gemini":
        return GeminiConfiguredTurnInterpreter()
    raise ValueError(f"Unsupported interpreter mode: {interpreter_mode}")


def write_transcript(log_path: Path, transcript: list[dict[str, Any]]) -> None:
    """Write a JSON transcript to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")


def _turn_entry(
    user_text: str,
    result: ConfiguredInteractionTurnResult,
) -> dict[str, Any]:
    """Convert one turn result into transcript data."""
    return {
        "type": "turn",
        "user": user_text,
        "status": result.status,
        "assistant": result.response_text,
        "intent_frame": _jsonable(result.intent_frame),
        "clarification": _jsonable(result.clarification),
        "approval_request": (
            result.execution_result.approval_request
            if result.execution_result is not None
            else None
        ),
        "plan_actions": (
            result.execution_result.plan_actions
            if result.execution_result is not None
            else []
        ),
        "execution": (
            _jsonable(result.execution_result.execution)
            if result.execution_result is not None
            else []
        ),
        "rendered_responses": (
            result.execution_result.rendered_responses
            if result.execution_result is not None
            else []
        ),
        "trace": (
            _jsonable(result.execution_result.trace)
            if result.execution_result is not None
            else []
        ),
    }


def _jsonable(value: Any) -> Any:
    """Convert local runtime objects into JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the config-aware Kortex travel interaction demo."
    )
    parser.add_argument(
        "--mode",
        choices=["scripted", "interactive"],
        default="scripted",
        help="Run a fixed transcript or read turns from stdin.",
    )
    parser.add_argument(
        "--approval",
        choices=["approve", "deny"],
        default="approve",
        help="Scripted HITL branch to exercise.",
    )
    parser.add_argument(
        "--interpreter",
        choices=["deterministic", "gemini"],
        default=None,
        help=(
            "Turn interpreter to use. Defaults to deterministic for scripted "
            "mode and gemini for interactive mode."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("demo_logs/travel_interaction_latest.json"),
        help="Path for the JSON interaction transcript.",
    )
    return parser.parse_args()


async def async_main() -> None:
    """Run the requested demo mode."""
    args = parse_args()
    interpreter_mode = args.interpreter or (
        "gemini" if args.mode == "interactive" else "deterministic"
    )
    if args.mode == "interactive":
        await run_interactive_travel_conversation(
            args.log_path,
            interpreter_mode=interpreter_mode,
        )
        return
    transcript = await run_scripted_travel_conversation(
        log_path=args.log_path,
        approval=args.approval,
        interpreter_mode=interpreter_mode,
    )
    for entry in transcript:
        if entry["type"] != "turn":
            continue
        print(f"user> {entry['user']}")
        print(f"kortex> {entry['assistant']}")
        print(f"status> {entry['status']}")
    print(f"Wrote transcript: {args.log_path}")


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
