"""CyberWave-first simulation client with optional browser sync.

This module owns the local simulation scene state and is the only place that
should know about CyberWave SDK or REST details.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "config"
ASSETS_PATH = CONFIG_DIR / "assets.json"
CONDITIONS_PATH = CONFIG_DIR / "demo_conditions.json"
CYBERWAVE_CONFIG_PATH = CONFIG_DIR / "cyberwave_config.json"

VALID_SIM_ZONES = ("left", "center", "right")
VALID_CLAW_ZONES = (*VALID_SIM_ZONES, "home")
SUPPORTED_SIM_ACTIONS = (
    "move_above_object",
    "grasp_object",
    "lift_object",
    "move_to_zone",
    "place_object",
    "return_home",
)


def _now() -> float:
    return time.time()


def _normalize(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(cleaned.split())


def _normalize_object_id(text: str) -> str:
    return _normalize(text).replace(" ", "_")


def _dedupe_aliases(values: list[str]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = str(value).strip()
        normalized = _normalize(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(alias)
    return aliases


@dataclass
class AssetDefinition:
    """Static asset definition used by the CyberWave demo scene."""

    asset_id: str
    display_name: str
    asset_type: str
    color: str
    default_size: str
    aliases: list[str] = field(default_factory=list)
    cyberwave_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "display_name": self.display_name,
            "asset_type": self.asset_type,
            "color": self.color,
            "default_size": self.default_size,
            "aliases": list(self.aliases),
            "cyberwave_ref": dict(self.cyberwave_ref),
            "metadata": dict(self.metadata),
        }


@dataclass
class ConditionDefinition:
    """Preset simulation condition for the demo."""

    name: str
    notes: str
    objects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "notes": self.notes,
            "objects": [dict(object_) for object_ in self.objects],
        }


@dataclass
class CyberWaveConfig:
    """Local config for optional CyberWave connectivity."""

    enabled: bool = True
    project_id: str = ""
    environment_id: str = ""
    twin_id: str = ""
    scene_id: str = ""
    api_base: str = ""
    default_condition: str = "demo_showcase_scene"
    use_openai_planner: bool = True
    openai_model: str = "gpt-5-mini"
    notes: str = ""
    config_source: str = "defaults"
    config_path: str = str(CYBERWAVE_CONFIG_PATH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project_id": self.project_id,
            "environment_id": self.environment_id,
            "twin_id": self.twin_id,
            "scene_id": self.scene_id,
            "api_base": self.api_base,
            "default_condition": self.default_condition,
            "use_openai_planner": self.use_openai_planner,
            "openai_model": self.openai_model,
            "notes": self.notes,
            "config_source": self.config_source,
            "config_path": self.config_path,
        }


@dataclass
class SimulationSceneState:
    """Current CyberWave demo scene state."""

    active_condition: str
    active_assets: list[str]
    objects: list[dict[str, Any]]
    claw_state: dict[str, Any]
    current_goal: str = ""
    last_action_sequence: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    scene_source: str = "cyberwave_local"
    last_updated: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_condition": self.active_condition,
            "active_assets": list(self.active_assets),
            "objects": [dict(object_) for object_ in self.objects],
            "claw_state": dict(self.claw_state),
            "current_goal": self.current_goal,
            "last_action_sequence": [dict(action) for action in self.last_action_sequence],
            "notes": self.notes,
            "scene_source": self.scene_source,
            "last_updated": self.last_updated,
        }


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def load_cyberwave_config() -> CyberWaveConfig:
    """Load the CyberWave demo config safely."""

    defaults = CyberWaveConfig().to_dict()
    config_source = "defaults"
    payload = _load_json(CYBERWAVE_CONFIG_PATH, defaults)
    if CYBERWAVE_CONFIG_PATH.exists():
        config_source = str(CYBERWAVE_CONFIG_PATH)
    return CyberWaveConfig(
        enabled=bool(payload.get("enabled", True)),
        project_id=str(payload.get("project_id", "")),
        environment_id=str(payload.get("environment_id", "")),
        twin_id=str(payload.get("twin_id", "")),
        scene_id=str(payload.get("scene_id", "")),
        api_base=str(payload.get("api_base", "")),
        default_condition=str(payload.get("default_condition", "demo_showcase_scene")),
        use_openai_planner=bool(payload.get("use_openai_planner", True)),
        openai_model=str(payload.get("openai_model", "gpt-5-mini")),
        notes=str(payload.get("notes", "")),
        config_source=config_source,
        config_path=str(CYBERWAVE_CONFIG_PATH),
    )


def load_asset_registry() -> dict[str, AssetDefinition]:
    """Load the local asset registry."""

    payload = _load_json(ASSETS_PATH, {"assets": []})
    assets: dict[str, AssetDefinition] = {}
    for entry in payload.get("assets", []):
        if not isinstance(entry, dict):
            continue
        asset_id = str(entry.get("id", "")).strip()
        if not asset_id:
            continue
        assets[asset_id] = AssetDefinition(
            asset_id=asset_id,
            display_name=str(entry.get("display_name", asset_id.replace("_", " "))),
            asset_type=str(entry.get("type", "object")),
            color=str(entry.get("color", "unknown")),
            default_size=str(entry.get("default_size", "medium")),
            aliases=[str(alias) for alias in entry.get("aliases", []) if str(alias).strip()],
            cyberwave_ref=dict(entry.get("cyberwave_ref", {})),
            metadata=dict(entry.get("metadata", {})),
        )
    return assets


def load_demo_conditions() -> dict[str, ConditionDefinition]:
    """Load the preset scene conditions."""

    payload = _load_json(CONDITIONS_PATH, {"conditions": []})
    conditions: dict[str, ConditionDefinition] = {}
    for entry in payload.get("conditions", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        conditions[name] = ConditionDefinition(
            name=name,
            notes=str(entry.get("notes", "")),
            objects=[
                dict(object_)
                for object_ in entry.get("objects", [])
                if isinstance(object_, dict)
            ],
        )
    return conditions


class CyberWaveClient:
    """Local-first CyberWave simulation layer with optional SDK session support."""

    def __init__(self) -> None:
        self.config = load_cyberwave_config()
        self.assets = load_asset_registry()
        self.conditions = load_demo_conditions()
        self.sdk_module: Any | None = None
        self.sdk_available = False
        self.sdk_error = ""
        self._sdk_client: Any | None = None
        self.connected = False
        self.remote_scene_sync_supported = False
        self.asset_sync_supported = False
        self.last_error = ""
        self.last_message = ""
        self.connection_mode = "local-only"
        self.last_remote_event = ""
        self.scene_binding_verified = False
        self._probe_sdk()
        self.scene_state = self._build_scene_state(self.config.default_condition)
        self._initial_scene_state = copy.deepcopy(self.scene_state)

    def _probe_sdk(self) -> None:
        try:
            self.sdk_module = importlib.import_module("cyberwave")
        except Exception as error:
            self.sdk_available = False
            self.sdk_error = str(error)
            self.sdk_module = None
            return
        self.sdk_available = True
        self.sdk_error = ""

    def _default_claw_state(self) -> dict[str, Any]:
        return {
            "zone": "home",
            "is_home": True,
            "holding": None,
            "holding_display_name": None,
            "busy": False,
        }

    def _build_scene_state(self, condition_name: str) -> SimulationSceneState:
        condition = self.conditions.get(condition_name)
        if condition is None and self.conditions:
            fallback_name = next(iter(self.conditions))
            condition = self.conditions[fallback_name]
            condition_name = fallback_name

        objects: list[dict[str, Any]] = []
        active_assets: list[str] = []
        notes = ""
        if condition is not None:
            notes = condition.notes
            for index, object_spec in enumerate(condition.objects, start=1):
                asset_id = str(object_spec.get("asset_id", "")).strip()
                asset = self.assets.get(asset_id)
                if asset is None:
                    continue
                zone = str(object_spec.get("zone", "center")).strip().lower()
                if zone not in VALID_SIM_ZONES:
                    zone = "center"
                display_name = str(object_spec.get("display_name", asset.display_name)).strip()
                if not display_name:
                    display_name = asset.display_name
                object_id = str(
                    object_spec.get("object_id")
                    or object_spec.get("instance_id")
                    or f"obj_{asset.asset_id}_{index}"
                ).strip()
                aliases = [
                    display_name,
                    asset.display_name,
                    asset.asset_type,
                    f"{asset.color} {asset.asset_type}",
                    *asset.aliases,
                ]
                aliases.extend(str(alias) for alias in object_spec.get("aliases", []))
                objects.append(
                    {
                        "object_id": object_id,
                        "id": object_id,
                        "asset_id": asset.asset_id,
                        "display_name": display_name,
                        "name": display_name,
                        "label": asset.asset_type,
                        "color": asset.color,
                        "zone": zone,
                        "position": zone,
                        "confidence": 1.0,
                        "bbox": [0, 0, 0, 0],
                        "area": 0,
                        "source": "cyberwave_local",
                        "aliases": _dedupe_aliases(aliases),
                        "metadata": dict(object_spec.get("metadata", {})),
                    }
                )
                active_assets.append(asset.asset_id)

        return SimulationSceneState(
            active_condition=condition_name,
            active_assets=sorted(dict.fromkeys(active_assets)),
            objects=objects,
            claw_state=self._default_claw_state(),
            current_goal="",
            last_action_sequence=[],
            notes=notes,
            scene_source="cyberwave_local",
            last_updated=_now(),
        )

    def _update_timestamp(self) -> None:
        self.scene_state.last_updated = _now()

    def get_scene_state(self) -> SimulationSceneState:
        return copy.deepcopy(self.scene_state)

    def get_active_scene_objects(self) -> list[dict[str, Any]]:
        return [dict(object_) for object_ in self.scene_state.objects]

    def summarize_active_objects_for_planner(self) -> list[dict[str, Any]]:
        return [
            {
                "object_id": str(object_["object_id"]),
                "display_name": str(object_["display_name"]),
                "label": str(object_["label"]),
                "color": str(object_["color"]),
                "zone": str(object_["zone"]),
                "aliases": list(object_.get("aliases", [])),
            }
            for object_ in self.scene_state.objects
        ]

    def _format_object_option(self, object_: dict[str, Any]) -> str:
        return (
            f"{object_['display_name']} at {object_['zone']} "
            f"({object_['object_id']})"
        )

    def _get_object_ref_by_id(self, object_id: str) -> dict[str, Any] | None:
        query = _normalize_object_id(object_id)
        if not query:
            return None
        for object_ in self.scene_state.objects:
            if _normalize_object_id(str(object_.get("object_id", object_.get("id", "")))) == query:
                return object_
        return None

    def get_object_by_id(self, object_id: str) -> dict[str, Any] | None:
        object_ref = self._get_object_ref_by_id(object_id)
        return None if object_ref is None else dict(object_ref)

    def _match_object_refs_by_display_name(self, name: str) -> list[dict[str, Any]]:
        query = _normalize(name)
        if not query:
            return []
        return [
            object_
            for object_ in self.scene_state.objects
            if query == _normalize(str(object_.get("display_name", object_.get("name", ""))))
        ]

    def _match_object_refs_by_alias(self, name: str) -> list[dict[str, Any]]:
        query = _normalize(name)
        if not query:
            return []
        matches: list[dict[str, Any]] = []
        for object_ in self.scene_state.objects:
            aliases = [_normalize(str(alias)) for alias in object_.get("aliases", [])]
            if query in aliases:
                matches.append(object_)
        return matches

    def _match_object_refs_by_tokens(self, name: str) -> list[dict[str, Any]]:
        query = _normalize(name)
        if not query:
            return []
        query_tokens = query.split()
        matches: list[dict[str, Any]] = []
        for object_ in self.scene_state.objects:
            search_space = [
                _normalize(str(object_.get("display_name", object_.get("name", "")))),
                *[_normalize(str(alias)) for alias in object_.get("aliases", [])],
            ]
            if any(all(token in candidate for token in query_tokens) for candidate in search_space):
                matches.append(object_)
        return matches

    def match_object_by_display_name(self, name: str) -> tuple[dict[str, Any] | None, str]:
        matches = self._match_object_refs_by_display_name(name)
        if not matches:
            return None, f"No active object matches the display name '{name}'."
        if len(matches) > 1:
            options = ", ".join(self._format_object_option(object_) for object_ in matches)
            return None, f"Display name '{name}' is ambiguous: {options}."
        return dict(matches[0]), ""

    def match_object_by_alias(self, name: str) -> tuple[dict[str, Any] | None, str]:
        matches = self._match_object_refs_by_alias(name)
        if not matches:
            return None, f"No active object matches the alias '{name}'."
        if len(matches) > 1:
            options = ", ".join(self._format_object_option(object_) for object_ in matches)
            return None, f"Alias '{name}' is ambiguous: {options}."
        return dict(matches[0]), ""

    def get_scene_context(self) -> dict[str, Any]:
        state = self.get_scene_state()
        return {
            "active_condition": state.active_condition,
            "notes": state.notes,
            "zones": list(VALID_SIM_ZONES),
            "allowed_zones": list(VALID_CLAW_ZONES),
            "available_actions": list(SUPPORTED_SIM_ACTIONS),
            "active_objects": self.summarize_active_objects_for_planner(),
            "objects": [
                {
                    "object_id": object_["object_id"],
                    "id": object_["id"],
                    "asset_id": object_["asset_id"],
                    "display_name": object_["display_name"],
                    "name": object_["name"],
                    "label": object_["label"],
                    "color": object_["color"],
                    "zone": object_["zone"],
                    "position": object_["position"],
                    "aliases": list(object_.get("aliases", [])),
                }
                for object_ in state.objects
            ],
            "claw_state": dict(state.claw_state),
            "active_assets": list(state.active_assets),
        }

    def get_scene_snapshot(self) -> dict[str, Any]:
        state = self.get_scene_state()
        return {
            "source": "cyberwave_simulation",
            "fallback_used": False,
            "roi_source": "simulation",
            "workspace_roi": None,
            "summary": self.summarize_scene(),
            "objects": [dict(object_) for object_ in state.objects],
            "metadata": {
                "scene_stale": False,
                "last_update_age_seconds": 0.0,
                "detection_mode": "cyberwave_simulation",
                "using_cached_scene": False,
                "fallback_reason": "",
                "camera_index": -1,
                "raw_candidates_count": len(state.objects),
                "valid_detections_count": len(state.objects),
                "live_loop_running": False,
            },
            "active_condition": state.active_condition,
            "scene_state": state.to_dict(),
        }

    def list_assets(self) -> list[AssetDefinition]:
        return [self.assets[key] for key in sorted(self.assets)]

    def list_conditions(self) -> list[ConditionDefinition]:
        return [self.conditions[key] for key in sorted(self.conditions)]

    def load_condition(self, condition_name: str) -> tuple[bool, str]:
        requested = condition_name.strip()
        resolved_name = requested
        if resolved_name not in self.conditions:
            normalized_requested = _normalize(requested).replace(" ", "_")
            for candidate in self.conditions:
                if _normalize(candidate).replace(" ", "_") == normalized_requested:
                    resolved_name = candidate
                    break
        if resolved_name not in self.conditions:
            return False, f"Unknown condition: {condition_name}."
        self.scene_state = self._build_scene_state(resolved_name)
        self._initial_scene_state = copy.deepcopy(self.scene_state)
        self.last_error = ""
        self.last_message = f"Loaded condition '{resolved_name}'."
        return True, self.last_message

    def reset_simulation(self) -> tuple[bool, str]:
        self.scene_state = copy.deepcopy(self._initial_scene_state)
        self.scene_state.last_action_sequence = []
        self.scene_state.current_goal = ""
        self._update_timestamp()
        return True, f"Simulation reset to condition '{self.scene_state.active_condition}'."

    def set_current_goal(self, goal: str) -> None:
        self.scene_state.current_goal = goal
        self._update_timestamp()

    def record_last_action_sequence(self, action_sequence: list[dict[str, Any]]) -> None:
        self.scene_state.last_action_sequence = [dict(action) for action in action_sequence]
        self._update_timestamp()

    def resolve_target(
        self,
        target_name: str | None = None,
        *,
        object_id: str | None = None,
        display_name: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        if object_id:
            object_ref = self._get_object_ref_by_id(object_id)
            if object_ref is None:
                return None, f"I could not find object id '{object_id}' in the current simulation scene."
            return object_ref, ""

        query = display_name or target_name or ""
        matches = self._match_object_refs_by_display_name(query)
        if not matches:
            matches = self._match_object_refs_by_alias(query)
        if not matches:
            matches = self._match_object_refs_by_tokens(query)
        if not matches:
            return None, f"I could not find '{query}' in the current simulation scene."
        if len(matches) > 1:
            options = ", ".join(self._format_object_option(object_) for object_ in matches)
            return None, f"'{query}' is ambiguous in the current scene: {options}."
        return matches[0], ""

    def _resolve_action_target(self, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        object_id = str(
            params.get("object_id")
            or params.get("target_object_id")
            or ""
        ).strip()
        object_name = str(params.get("object_name", "")).strip()
        display_name = str(
            params.get("display_name")
            or params.get("target_display_name")
            or ""
        ).strip()

        if object_id:
            return self.resolve_target(object_id=object_id)
        if display_name:
            return self.resolve_target(display_name=display_name)
        if object_name:
            return self.resolve_target(target_name=object_name)
        return None, "No target object was provided for the simulation action."

    def _emit_remote_hint(self, action: str, payload: dict[str, Any]) -> list[str]:
        if not self.connected:
            return ["CyberWave is not connected. Local simulation updated only."]
        self.last_remote_event = f"{action} {json.dumps(payload, sort_keys=True)}"
        return [
            "CyberWave SDK session is connected, but remote scene sync is still local-first for this MVP."
        ]

    def apply_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if action not in SUPPORTED_SIM_ACTIONS:
            message = f"Unsupported simulation action: {action}."
            return {
                "success": False,
                "message": message,
                "supported": False,
                "warnings": [],
                "refusal_reason": message,
            }

        claw = self.scene_state.claw_state
        warnings = self._emit_remote_hint(action, params)

        if action == "move_above_object":
            target_object, error_message = self._resolve_action_target(params)
            if target_object is None:
                return {
                    "success": False,
                    "message": error_message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": error_message,
                }
            claw["zone"] = target_object["zone"]
            claw["is_home"] = False
            self._update_timestamp()
            return {
                "success": True,
                "message": (
                    f"Moved above {target_object['display_name']} in {target_object['zone']} zone."
                ),
                "supported": True,
                "warnings": warnings,
            }

        if action == "grasp_object":
            target_object, error_message = self._resolve_action_target(params)
            if target_object is None:
                return {
                    "success": False,
                    "message": error_message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": error_message,
                }
            if claw["holding"] is not None:
                message = (
                    "The claw is already holding "
                    f"{claw.get('holding_display_name') or claw['holding']}."
                )
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            if claw["zone"] != target_object["zone"]:
                message = (
                    "The claw is in "
                    f"{claw['zone']} but {target_object['display_name']} is in {target_object['zone']}."
                )
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            claw["holding"] = target_object["object_id"]
            claw["holding_display_name"] = target_object["display_name"]
            claw["is_home"] = False
            self._update_timestamp()
            return {
                "success": True,
                "message": f"Grasped {target_object['display_name']}.",
                "supported": True,
                "warnings": warnings,
            }

        if action == "lift_object":
            target_object, error_message = self._resolve_action_target(
                {
                    "object_id": params.get("object_id", claw["holding"] or ""),
                    "display_name": params.get("display_name", ""),
                    "object_name": params.get("object_name", ""),
                }
            )
            if target_object is None:
                return {
                    "success": False,
                    "message": error_message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": error_message,
                }
            if claw["holding"] != target_object["object_id"]:
                message = (
                    "The claw is not holding "
                    f"{target_object['display_name']}."
                )
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            claw["is_home"] = False
            self._update_timestamp()
            return {
                "success": True,
                "message": f"Lifted {target_object['display_name']}.",
                "supported": True,
                "warnings": warnings,
            }

        if action == "move_to_zone":
            zone = str(params.get("zone", "")).lower()
            if zone not in VALID_SIM_ZONES:
                message = f"Unsupported zone: {zone or 'unknown'}."
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            claw["zone"] = zone
            claw["is_home"] = False
            self._update_timestamp()
            return {
                "success": True,
                "message": f"Moved to {zone} zone.",
                "supported": True,
                "warnings": warnings,
            }

        if action == "place_object":
            zone = str(params.get("zone", "")).lower()
            if zone not in VALID_SIM_ZONES:
                message = f"Unsupported zone: {zone or 'unknown'}."
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            target_object, error_message = self._resolve_action_target(
                {
                    "object_id": params.get("object_id", claw["holding"] or ""),
                    "display_name": params.get("display_name", ""),
                    "object_name": params.get("object_name", ""),
                }
            )
            if target_object is None:
                return {
                    "success": False,
                    "message": error_message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": error_message,
                }
            if claw["holding"] != target_object["object_id"]:
                message = (
                    "The claw is not holding "
                    f"{target_object['display_name']}."
                )
                return {
                    "success": False,
                    "message": message,
                    "supported": True,
                    "warnings": warnings,
                    "refusal_reason": message,
                }
            target_object["zone"] = zone
            target_object["position"] = zone
            claw["holding"] = None
            claw["holding_display_name"] = None
            claw["zone"] = zone
            claw["is_home"] = False
            self._update_timestamp()
            return {
                "success": True,
                "message": f"Placed {target_object['display_name']} in {zone} zone.",
                "supported": True,
                "warnings": warnings,
            }

        claw["zone"] = "home"
        claw["is_home"] = True
        claw["holding"] = None
        claw["holding_display_name"] = None
        self._update_timestamp()
        return {
            "success": True,
            "message": "Returned the claw to the home position.",
            "supported": True,
            "warnings": warnings,
        }

    def replay_last_sequence(self) -> tuple[bool, str]:
        if not self.scene_state.last_action_sequence:
            return False, "There is no simulation action sequence to replay."
        replay_actions = [dict(action) for action in self.scene_state.last_action_sequence]
        self.scene_state = copy.deepcopy(self._initial_scene_state)
        self.scene_state.last_action_sequence = replay_actions
        for action in replay_actions:
            response = self.apply_action(str(action.get("type", "")), dict(action.get("params", {})))
            if not response.get("success", False):
                return False, f"Replay stopped: {response.get('message', 'unknown error')}"
        return True, "Replayed the last simulation action sequence."

    def connect(self) -> tuple[bool, str]:
        if not self.config.enabled:
            self.connected = False
            return False, "CyberWave is disabled in config/cyberwave_config.json."
        if not self.sdk_available:
            self.connected = False
            self.last_error = self.sdk_error or "CyberWave SDK is not installed."
            return False, f"CyberWave SDK is unavailable: {self.last_error}"
        token = os.getenv("CYBERWAVE_API_TOKEN", "").strip()
        if not token:
            self.connected = False
            self.last_error = "CYBERWAVE_API_TOKEN is not set."
            return False, self.last_error
        if not all(
            [
                self.config.project_id,
                self.config.environment_id,
                self.config.twin_id,
                self.config.scene_id,
            ]
        ):
            self.connected = False
            self.last_error = (
                "CyberWave config is incomplete. Fill project_id, environment_id, twin_id, and scene_id."
            )
            return False, self.last_error

        client_class = getattr(self.sdk_module, "Cyberwave", None)
        if client_class is None:
            self.connected = False
            self.last_error = "CyberWave SDK is installed, but the Cyberwave client class was not found."
            return False, self.last_error

        try:
            self._sdk_client = client_class(token=token)
        except Exception as error:
            self.connected = False
            self.last_error = f"Failed to initialize CyberWave SDK client: {error}"
            return False, self.last_error

        self.connected = True
        self.connection_mode = "sdk-session"
        self.last_error = ""
        self.last_message = (
            "CyberWave SDK session initialized. Local simulation is active; remote asset sync is still limited."
        )
        return True, self.last_message

    def disconnect(self) -> tuple[bool, str]:
        self._sdk_client = None
        self.connected = False
        self.connection_mode = "local-only"
        self.scene_binding_verified = False
        self.last_message = "CyberWave session closed. Local simulation remains available."
        return True, self.last_message

    def status(self) -> dict[str, Any]:
        token_present = bool(os.getenv("CYBERWAVE_API_TOKEN", "").strip())
        return {
            "enabled": self.config.enabled,
            "sdk_available": self.sdk_available,
            "sdk_error": self.sdk_error,
            "token_configured": token_present,
            "connected": self.connected,
            "connection_mode": self.connection_mode,
            "project_id": self.config.project_id,
            "environment_id": self.config.environment_id,
            "twin_id": self.config.twin_id,
            "scene_id": self.config.scene_id,
            "api_base": self.config.api_base,
            "default_condition": self.config.default_condition,
            "active_condition": self.scene_state.active_condition,
            "local_sim_ready": True,
            "remote_ready": self.connected,
            "scene_binding_verified": self.scene_binding_verified,
            "remote_scene_sync_supported": self.remote_scene_sync_supported,
            "asset_sync_supported": self.asset_sync_supported,
            "backend_name": "cyberwave:sdk" if self.connected else "cyberwave:local-scene",
            "last_remote_event": self.last_remote_event,
            "last_error": self.last_error,
            "last_message": self.last_message,
            "config_source": self.config.config_source,
            "config_path": self.config.config_path,
            "use_openai_planner": self.config.use_openai_planner,
            "openai_model": self.config.openai_model,
            "scene_source": self.scene_state.scene_source,
        }

    def summarize_status(self) -> str:
        status = self.status()
        lines = [
            f"- backend: {status['backend_name']}",
            f"- local sim ready: {'yes' if status['local_sim_ready'] else 'no'}",
            f"- connected: {'yes' if status['connected'] else 'no'}",
            f"- connection mode: {status['connection_mode']}",
            f"- sdk available: {'yes' if status['sdk_available'] else 'no'}",
            f"- token configured: {'yes' if status['token_configured'] else 'no'}",
            f"- project id: {status['project_id'] or 'unset'}",
            f"- environment id: {status['environment_id'] or 'unset'}",
            f"- twin id: {status['twin_id'] or 'unset'}",
            f"- scene id: {status['scene_id'] or 'unset'}",
            f"- active condition: {status['active_condition']}",
            f"- scene sync supported: {'yes' if status['remote_scene_sync_supported'] else 'no'}",
            f"- asset sync supported: {'yes' if status['asset_sync_supported'] else 'no'}",
            f"- openai planner enabled: {'yes' if status['use_openai_planner'] else 'no'}",
            f"- openai model: {status['openai_model']}",
            f"- config source: {status['config_source']}",
            f"- config path: {status['config_path']}",
            f"- last remote event: {status['last_remote_event'] or 'none'}",
            f"- last error: {status['last_error'] or 'none'}",
        ]
        if status["sdk_error"]:
            lines.append(f"- sdk error: {status['sdk_error']}")
        if status["last_message"]:
            lines.append(f"- last message: {status['last_message']}")
        return "\n".join(lines)

    def summarize_scene(self) -> str:
        state = self.scene_state
        lines = [
            f"- active condition: {state.active_condition}",
            f"- scene source: {state.scene_source}",
            f"- current goal: {state.current_goal or 'none'}",
            f"- claw zone: {state.claw_state['zone']}",
            f"- claw home: {'yes' if state.claw_state['is_home'] else 'no'}",
            f"- claw holding: {state.claw_state.get('holding_display_name') or 'nothing'}",
        ]
        if state.objects:
            lines.append("- objects:")
            for object_ in state.objects:
                lines.append(
                    "  - "
                    f"{object_['display_name']} [{object_['object_id']}] "
                    f"({object_['label']}, {object_['color']}) at {object_['zone']}"
                )
        else:
            lines.append("- objects: none")
        if state.notes:
            lines.append(f"- notes: {state.notes}")
        if state.last_action_sequence:
            lines.append(f"- last action sequence: {len(state.last_action_sequence)} steps")
        else:
            lines.append("- last action sequence: none")
        return "\n".join(lines)

    def summarize_assets(self) -> str:
        lines: list[str] = []
        for asset in self.list_assets():
            lines.append(
                f"- {asset.asset_id}: {asset.display_name} | type={asset.asset_type} | "
                f"color={asset.color} | size={asset.default_size}"
            )
        return "\n".join(lines) if lines else "- no assets registered"

    def summarize_conditions(self) -> str:
        lines: list[str] = []
        for condition in self.list_conditions():
            lines.append(
                f"- {condition.name}: {condition.notes or 'no notes'} "
                f"({len(condition.objects)} objects)"
            )
        return "\n".join(lines) if lines else "- no demo conditions registered"


_CYBERWAVE_CLIENT = CyberWaveClient()


def get_cyberwave_client() -> CyberWaveClient:
    return _CYBERWAVE_CLIENT
