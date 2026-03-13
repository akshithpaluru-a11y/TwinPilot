"""OpenAI Responses-based planner for the CyberWave simulation demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - depends on local install
    from openai import OpenAI
except Exception:  # pragma: no cover - depends on local install
    OpenAI = None  # type: ignore[assignment]

ALLOWED_ACTION_TYPES = {
    "move_above_object",
    "grasp_object",
    "lift_object",
    "move_to_zone",
    "place_object",
    "return_home",
}
VALID_ZONES = {"left", "center", "right", "home"}
OBJECT_TARGET_ACTIONS = {
    "move_above_object",
    "grasp_object",
    "lift_object",
    "place_object",
}
PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "goal",
        "target_object_id",
        "target_display_name",
        "destination_zone",
        "needs_clarification",
        "clarification_question",
        "actions",
    ],
    "properties": {
        "goal": {"type": "string"},
        "target_object_id": {"type": ["string", "null"]},
        "target_display_name": {"type": ["string", "null"]},
        "destination_zone": {"type": ["string", "null"]},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"]},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "target_object_id", "target_display_name", "zone"],
                "properties": {
                    "type": {"type": "string"},
                    "target_object_id": {"type": ["string", "null"]},
                    "target_display_name": {"type": ["string", "null"]},
                    "zone": {"type": ["string", "null"]},
                },
            },
        },
    },
}
SYSTEM_PROMPT = """You control a simulated robot claw in a tabletop CyberWave demo.

Rules:
- You may only reference objects from the provided ACTIVE_OBJECTS list.
- You may only reference zones from the provided ALLOWED_ZONES list.
- Do not invent object identifiers, display names, assets, or zones.
- target_object_id must be one of the ACTIVE_OBJECTS object_id values when a target is known.
- Return valid JSON only.
- If the target object is ambiguous or missing, set needs_clarification=true and ask one short question.
- Do not guess between multiple matching objects.
- Only use these action types:
  - move_above_object
  - grasp_object
  - lift_object
  - move_to_zone
  - place_object
  - return_home
- For a pick-and-place request, include a complete sequence.
- Keep actions limited to the requested task.
"""


def _normalize(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(cleaned.split())


def _normalize_object_id(text: str) -> str:
    return _normalize(text).replace(" ", "_")


def _scene_objects(scene_context: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene_context.get("active_objects")
    if isinstance(objects, list):
        return [dict(object_) for object_ in objects if isinstance(object_, dict)]
    objects = scene_context.get("objects", [])
    return [dict(object_) for object_ in objects if isinstance(object_, dict)]


def _format_object_option(object_: dict[str, Any]) -> str:
    return (
        f"{object_.get('display_name', object_.get('name', 'object'))} "
        f"at {object_.get('zone', 'unknown')}"
    )


@dataclass
class PlannerAction:
    """Validated planner action."""

    action_type: str
    target_object_id: str | None = None
    target_display_name: str | None = None
    zone: str | None = None

    @property
    def target(self) -> str | None:
        return self.target_display_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.action_type,
            "target_object_id": self.target_object_id,
            "target_display_name": self.target_display_name,
            "zone": self.zone,
        }


@dataclass
class PlannerResponse:
    """Validated planner response."""

    success: bool
    goal: str = ""
    target_object_id: str | None = None
    target_display_name: str | None = None
    destination_zone: str | None = None
    actions: list[PlannerAction] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    message: str = ""
    error: str = ""
    error_type: str = ""
    model: str = ""
    raw_text: str = ""

    @property
    def target_name(self) -> str | None:
        return self.target_display_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "target_object_id": self.target_object_id,
            "target_display_name": self.target_display_name,
            "destination_zone": self.destination_zone,
            "actions": [action.to_dict() for action in self.actions],
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "message": self.message,
            "error": self.error,
            "error_type": self.error_type,
            "model": self.model,
            "raw_text": self.raw_text,
        }


class OpenAISimulationPlanner:
    """Small wrapper around the OpenAI Responses API for simulation plans."""

    def __init__(self, model: str, *, enabled: bool = True) -> None:
        self.model = model
        self.enabled = enabled
        self.last_error = ""
        self.last_message = ""

    def status(self) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        return {
            "enabled": self.enabled,
            "openai_available": OpenAI is not None,
            "api_key_configured": bool(api_key),
            "model": self.model,
            "last_error": self.last_error,
            "last_message": self.last_message,
        }

    def _build_prompt(
        self,
        command_text: str,
        scene_context: dict[str, Any],
        clarification_context: str | None = None,
    ) -> str:
        payload = {
            "USER_COMMAND": command_text,
            "CLARIFICATION_CONTEXT": clarification_context,
            "ACTIVE_OBJECTS": scene_context.get("active_objects", []),
            "ALLOWED_ZONES": scene_context.get("allowed_zones", list(VALID_ZONES)),
            "AVAILABLE_ACTIONS": scene_context.get("available_actions", sorted(ALLOWED_ACTION_TYPES)),
            "SCENE_STATE": {
                "active_condition": scene_context.get("active_condition", ""),
                "notes": scene_context.get("notes", ""),
                "claw_state": scene_context.get("claw_state", {}),
                "active_assets": scene_context.get("active_assets", []),
            },
        }
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Return JSON that matches the required schema exactly.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", "")
        if output_text:
            return str(output_text)
        dump_method = getattr(response, "model_dump", None)
        if not callable(dump_method):
            return ""
        payload = dump_method()
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
        return ""

    def _planner_error(
        self,
        *,
        message: str,
        error_type: str,
        goal: str = "",
        raw_text: str = "",
    ) -> PlannerResponse:
        return PlannerResponse(
            success=False,
            goal=goal,
            error=message,
            error_type=error_type,
            raw_text=raw_text,
        )

    def _clarification_response(
        self,
        *,
        question: str,
        goal: str,
        target_display_name: str | None = None,
        destination_zone: str | None = None,
        message: str = "Planner requested clarification.",
    ) -> PlannerResponse:
        return PlannerResponse(
            success=True,
            goal=goal,
            target_display_name=target_display_name,
            destination_zone=destination_zone,
            actions=[],
            needs_clarification=True,
            clarification_question=question,
            message=message,
        )

    def _get_object_by_id(
        self,
        object_id: str,
        scene_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        normalized_id = _normalize_object_id(object_id)
        if not normalized_id:
            return None
        for object_ in _scene_objects(scene_context):
            candidate = str(object_.get("object_id", object_.get("id", "")))
            if _normalize_object_id(candidate) == normalized_id:
                return object_
        return None

    def _match_objects_by_name(
        self,
        name: str,
        scene_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query = _normalize(name)
        if not query:
            return []

        exact_display: list[dict[str, Any]] = []
        exact_alias: list[dict[str, Any]] = []
        fallback_matches: list[dict[str, Any]] = []
        query_tokens = query.split()

        for object_ in _scene_objects(scene_context):
            display_name = str(object_.get("display_name", object_.get("name", "")))
            normalized_display = _normalize(display_name)
            aliases = [_normalize(str(alias)) for alias in object_.get("aliases", [])]
            if query == normalized_display:
                exact_display.append(object_)
                continue
            if query in aliases:
                exact_alias.append(object_)
                continue
            search_space = [normalized_display, *aliases]
            if any(all(token in candidate for token in query_tokens) for candidate in search_space):
                fallback_matches.append(object_)

        if exact_display:
            return exact_display
        if exact_alias:
            return exact_alias
        return fallback_matches

    def _resolve_target_reference(
        self,
        *,
        target_object_id: str | None,
        target_display_name: str | None,
        scene_context: dict[str, Any],
        goal: str,
        destination_zone: str | None,
    ) -> tuple[dict[str, Any] | None, PlannerResponse | None, str]:
        if target_object_id:
            target_object = self._get_object_by_id(target_object_id, scene_context)
            if target_object is None:
                return None, self._planner_error(
                    message=f"Planner referenced an unknown target_object_id: {target_object_id}.",
                    error_type="unknown_object_id",
                    goal=goal,
                ), ""
            return target_object, None, ""

        if not target_display_name:
            return None, None, ""

        matches = self._match_objects_by_name(target_display_name, scene_context)
        if not matches:
            return None, self._planner_error(
                message=f"Planner referenced an unknown target: {target_display_name}.",
                error_type="unknown_target",
                goal=goal,
            ), ""
        if len(matches) > 1:
            options = " or ".join(_format_object_option(object_) for object_ in matches)
            question = f"Which object do you mean: {options}?"
            return None, self._clarification_response(
                question=question,
                goal=goal,
                target_display_name=target_display_name,
                destination_zone=destination_zone,
                message="Planner target was ambiguous and requires clarification.",
            ), ""
        return matches[0], None, ""

    def _validate_action_payload(
        self,
        payload: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        goal: str,
        destination_zone: str | None,
        default_target: dict[str, Any] | None,
    ) -> tuple[PlannerAction | None, PlannerResponse | None]:
        action_type = str(payload.get("type", "")).strip()
        if action_type not in ALLOWED_ACTION_TYPES:
            return None, self._planner_error(
                message=f"Planner returned unsupported action type: {action_type or 'unknown'}.",
                error_type="invalid_action",
                goal=goal,
            )

        zone = payload.get("zone")
        normalized_zone = None
        if zone is not None:
            normalized_zone = str(zone).strip().lower()
            if normalized_zone not in VALID_ZONES:
                return None, self._planner_error(
                    message=f"Planner returned unsupported zone: {zone}.",
                    error_type="invalid_destination",
                    goal=goal,
                )
        if action_type in {"move_to_zone", "place_object"} and normalized_zone is None:
            return None, self._planner_error(
                message=f"Planner action '{action_type}' is missing a zone.",
                error_type="invalid_destination",
                goal=goal,
            )

        target_object = default_target
        if action_type in OBJECT_TARGET_ACTIONS:
            action_target_id = payload.get("target_object_id")
            action_target_display_name = payload.get("target_display_name")
            if action_target_id or action_target_display_name:
                target_object, clarification_or_error, _ = self._resolve_target_reference(
                    target_object_id=None
                    if action_target_id is None
                    else str(action_target_id).strip() or None,
                    target_display_name=None
                    if action_target_display_name is None
                    else str(action_target_display_name).strip() or None,
                    scene_context=scene_context,
                    goal=goal,
                    destination_zone=destination_zone,
                )
                if clarification_or_error is not None:
                    return None, clarification_or_error
            if target_object is None:
                return None, self._planner_error(
                    message=f"Planner action '{action_type}' is missing a target object.",
                    error_type="missing_target",
                    goal=goal,
                )

        return PlannerAction(
            action_type=action_type,
            target_object_id=None if target_object is None else str(target_object["object_id"]),
            target_display_name=None if target_object is None else str(target_object["display_name"]),
            zone=normalized_zone,
        ), None

    def _validate_response(
        self,
        payload: dict[str, Any],
        scene_context: dict[str, Any],
    ) -> PlannerResponse:
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            return self._planner_error(
                message="Planner response is missing goal.",
                error_type="schema_error",
            )

        target_object_id = payload.get("target_object_id")
        target_display_name = payload.get("target_display_name")
        destination_zone = payload.get("destination_zone")
        destination_value = None if destination_zone is None else str(destination_zone).strip().lower()
        needs_clarification = bool(payload.get("needs_clarification", False))
        clarification_question = payload.get("clarification_question")

        if destination_value is not None and destination_value not in VALID_ZONES:
            return self._planner_error(
                message=f"Planner destination zone is invalid: {destination_zone}.",
                error_type="invalid_destination",
                goal=goal,
            )

        if needs_clarification:
            question = str(clarification_question or "").strip()
            if not question:
                return self._planner_error(
                    message="Planner requested clarification but did not provide a question.",
                    error_type="schema_error",
                    goal=goal,
                )
            return self._clarification_response(
                question=question,
                goal=goal,
                target_display_name=None
                if target_display_name is None
                else str(target_display_name).strip() or None,
                destination_zone=destination_value,
            )

        target_object, clarification_or_error, _ = self._resolve_target_reference(
            target_object_id=None
            if target_object_id is None
            else str(target_object_id).strip() or None,
            target_display_name=None
            if target_display_name is None
            else str(target_display_name).strip() or None,
            scene_context=scene_context,
            goal=goal,
            destination_zone=destination_value,
        )
        if clarification_or_error is not None:
            return clarification_or_error

        validated_actions: list[PlannerAction] = []
        for entry in payload.get("actions", []):
            if not isinstance(entry, dict):
                return self._planner_error(
                    message="Planner action entry was not an object.",
                    error_type="schema_error",
                    goal=goal,
                )
            action, validation_error = self._validate_action_payload(
                entry,
                scene_context,
                goal=goal,
                destination_zone=destination_value,
                default_target=target_object,
            )
            if validation_error is not None:
                return validation_error
            if action is not None:
                validated_actions.append(action)

        if not validated_actions:
            return self._planner_error(
                message="Planner response did not contain any actions.",
                error_type="schema_error",
                goal=goal,
            )

        if destination_value is None:
            for action in reversed(validated_actions):
                if action.zone in VALID_ZONES:
                    destination_value = action.zone
                    break

        if target_object is None:
            action_targets = {
                action.target_object_id
                for action in validated_actions
                if action.target_object_id is not None
            }
            if len(action_targets) == 1:
                target_object = self._get_object_by_id(next(iter(action_targets)), scene_context)
            elif len(action_targets) > 1:
                return self._planner_error(
                    message="Planner returned actions that reference multiple targets.",
                    error_type="ambiguous_target",
                    goal=goal,
                )

        return PlannerResponse(
            success=True,
            goal=goal,
            target_object_id=None if target_object is None else str(target_object["object_id"]),
            target_display_name=None
            if target_object is None
            else str(target_object["display_name"]),
            destination_zone=destination_value,
            actions=validated_actions,
            needs_clarification=False,
            clarification_question=None,
            message="Planner response validated.",
        )

    def plan_command(
        self,
        command_text: str,
        scene_context: dict[str, Any],
        clarification_context: str | None = None,
    ) -> PlannerResponse:
        if not self.enabled:
            return self._planner_error(
                message="OpenAI simulation planner is disabled by config.",
                error_type="planner_disabled",
            )
        if OpenAI is None:
            return self._planner_error(
                message="The openai package is not installed.",
                error_type="missing_dependency",
            )
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return self._planner_error(
                message="OPENAI_API_KEY is not set.",
                error_type="missing_api_key",
            )

        client = OpenAI(api_key=api_key)
        prompt = self._build_prompt(command_text, scene_context, clarification_context)

        try:
            response = client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "simulation_plan",
                        "schema": PLANNER_SCHEMA,
                        "strict": True,
                    }
                },
            )
        except Exception as error:  # pragma: no cover - depends on network and local key
            self.last_error = str(error)
            return self._planner_error(
                message=f"OpenAI planner request failed: {error}",
                error_type="api_error",
            )

        raw_text = self._extract_text(response)
        if not raw_text:
            return self._planner_error(
                message="OpenAI planner returned no text output.",
                error_type="empty_response",
            )

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            return self._planner_error(
                message=f"Planner returned invalid JSON: {error}",
                error_type="invalid_json",
                raw_text=raw_text,
            )

        if not isinstance(payload, dict):
            return self._planner_error(
                message="Planner JSON response was not an object.",
                error_type="schema_error",
                raw_text=raw_text,
            )

        result = self._validate_response(payload, scene_context)
        result.model = self.model
        result.raw_text = raw_text
        self.last_message = result.message or "Planner request finished."
        self.last_error = result.error
        return result


_PLANNER: OpenAISimulationPlanner | None = None


def get_simulation_planner(model: str, *, enabled: bool = True) -> OpenAISimulationPlanner:
    global _PLANNER
    if _PLANNER is None or _PLANNER.model != model or _PLANNER.enabled != enabled:
        _PLANNER = OpenAISimulationPlanner(model=model, enabled=enabled)
    return _PLANNER
