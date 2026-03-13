"""Local SO100 bridge service for dry-run and optional LeRobot-backed execution.

This file is the only low-level hardware boundary in the project.
The planner app never imports LeRobot or Feetech directly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from bridge_client import BridgeConfig, load_bridge_config

CONFIG_DIR = Path(__file__).resolve().parent / "config"
SO100_CONFIG_PATH = CONFIG_DIR / "so100_config.json"
SUPPORTED_PRIMITIVES = {
    "home",
    "move_above_object",
    "grasp_object",
    "lift_object",
    "move_to_zone",
    "place_object",
    "return_home",
}
REAL_HOME_PRIMITIVES = {"home", "return_home"}


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class SO100Config:
    """Small real-backend config model for the bridge."""

    backend_type: str = "lerobot_so100"
    robot_type: str = "so100"
    teleop_type: str = ""
    follower_port: str = ""
    leader_port: str = ""
    robot_id: str = "follower_1"
    home_on_connect: bool = False
    default_dry_run: bool = True
    require_connection_before_execute: bool = True
    allow_real_execution: bool = False
    use_lerobot_backend: bool = True
    camera_index: int = -1
    notes: str = ""
    config_source: str = "defaults"
    config_path: str = str(SO100_CONFIG_PATH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_type": self.backend_type,
            "robot_type": self.robot_type,
            "teleop_type": self.teleop_type,
            "follower_port": self.follower_port,
            "leader_port": self.leader_port,
            "robot_id": self.robot_id,
            "home_on_connect": self.home_on_connect,
            "default_dry_run": self.default_dry_run,
            "require_connection_before_execute": self.require_connection_before_execute,
            "allow_real_execution": self.allow_real_execution,
            "use_lerobot_backend": self.use_lerobot_backend,
            "camera_index": self.camera_index,
            "notes": self.notes,
            "config_source": self.config_source,
            "config_path": self.config_path,
        }


@dataclass
class LeRobotDependencyStatus:
    """Optional import readiness for the LeRobot / Feetech stack."""

    lerobot_available: bool = False
    feetech_support_available: bool = False
    factory_available: bool = False
    config_class_available: bool = False
    lerobot_import_error: str = ""
    feetech_import_error: str = ""
    factory_error: str = ""
    config_class_error: str = ""
    selected_factory_path: str = ""
    selected_config_class_path: str = ""
    selected_config_class_name: str = ""
    selected_feetech_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lerobot_available": self.lerobot_available,
            "feetech_support_available": self.feetech_support_available,
            "factory_available": self.factory_available,
            "config_class_available": self.config_class_available,
            "lerobot_import_error": self.lerobot_import_error,
            "feetech_import_error": self.feetech_import_error,
            "factory_error": self.factory_error,
            "config_class_error": self.config_class_error,
            "selected_factory_path": self.selected_factory_path,
            "selected_config_class_path": self.selected_config_class_path,
            "selected_config_class_name": self.selected_config_class_name,
            "selected_feetech_path": self.selected_feetech_path,
        }


def load_so100_config() -> SO100Config:
    """Load the SO100 runtime config safely, preserving sane defaults."""

    defaults = {
        "backend_type": "lerobot_so100",
        "robot_type": "so100",
        "teleop_type": "",
        "follower_port": "",
        "leader_port": "",
        "robot_id": "follower_1",
        "home_on_connect": False,
        "default_dry_run": True,
        "require_connection_before_execute": True,
        "allow_real_execution": False,
        "use_lerobot_backend": True,
        "camera_index": -1,
        "notes": "",
    }
    config_source = "defaults"

    if SO100_CONFIG_PATH.exists():
        try:
            loaded = json.loads(SO100_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            if "serial_port" in loaded and "follower_port" not in loaded:
                loaded["follower_port"] = loaded["serial_port"]
            if "dry_run_default" in loaded and "default_dry_run" not in loaded:
                loaded["default_dry_run"] = loaded["dry_run_default"]
            for key in defaults:
                if key in loaded:
                    defaults[key] = loaded[key]
            config_source = str(SO100_CONFIG_PATH)

    return SO100Config(
        backend_type=str(defaults["backend_type"]),
        robot_type=str(defaults["robot_type"]),
        teleop_type=str(defaults["teleop_type"]),
        follower_port=str(defaults["follower_port"]),
        leader_port=str(defaults["leader_port"]),
        robot_id=str(defaults["robot_id"]),
        home_on_connect=bool(defaults["home_on_connect"]),
        default_dry_run=bool(defaults["default_dry_run"]),
        require_connection_before_execute=bool(defaults["require_connection_before_execute"]),
        allow_real_execution=bool(defaults["allow_real_execution"]),
        use_lerobot_backend=bool(defaults["use_lerobot_backend"]),
        camera_index=int(defaults["camera_index"]),
        notes=str(defaults["notes"]),
        config_source=config_source,
        config_path=str(SO100_CONFIG_PATH),
    )


class BaseBridgeBackend:
    """Common bridge backend behavior for dry-run and future real execution."""

    backend_type = "base"
    backend_name = "bridge:base"

    def __init__(self, runtime: "BridgeRuntime") -> None:
        self.runtime = runtime
        self.last_backend_message = ""
        self.last_raw_backend_message = ""

    def capabilities_report(self) -> dict[str, Any]:
        return {
            "backend_selected": self.backend_name,
            "readiness_stage": "dry_run_only",
            "lerobot_available": False,
            "feetech_support_available": False,
            "backend_ready_for_real_execution": False,
            "real_execution_allowed_by_config": self.runtime.so100_config.allow_real_execution,
            "can_connect": False,
            "can_home": False,
            "can_execute_motion_primitives": False,
            "can_execute_gripper_primitives": False,
            "can_move_above_object": False,
            "can_grasp": False,
            "can_lift": False,
            "can_move_to_zone": False,
            "can_place": False,
            "not_ready_reasons": ["Real backend is not implemented for this bridge backend."],
            "warnings": [],
        }

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_type": self.backend_type,
            "robot_config": self.runtime.so100_config.to_dict(),
            "last_backend_message": self.last_backend_message,
            "raw_backend_message": self.last_raw_backend_message,
        }

    def real_backend_ready(self) -> bool:
        return bool(self.capabilities_report()["backend_ready_for_real_execution"])

    def status_snapshot(self) -> dict[str, Any]:
        capabilities = self.capabilities_report()
        backend_info = self.backend_info()
        return {
            "backend_name": self.backend_name,
            "backend_type": self.backend_type,
            "connected": self.runtime.connected,
            "busy": self.runtime.busy,
            "current_zone": self.runtime.current_zone,
            "held_object": self.runtime.held_object,
            "last_target_handled": self.runtime.last_target_handled,
            "is_home": self.runtime.is_home,
            "last_command": self.runtime.last_command,
            "last_error": self.runtime.last_error,
            "last_connection_attempt": self.runtime.last_connection_attempt,
            "bridge_running": True,
            "dry_run": self.runtime.dry_run,
            "real_backend_ready": self.real_backend_ready(),
            "warnings": list(capabilities.get("warnings", [])),
            "capabilities": capabilities,
            "backend_info": backend_info,
            "lerobot_available": bool(capabilities.get("lerobot_available", False)),
            "feetech_support_available": bool(
                capabilities.get("feetech_support_available", False)
            ),
        }

    def connect(self) -> dict[str, Any]:
        self.runtime.last_connection_attempt = _now()
        self.runtime.abort_requested = False
        if self.runtime.dry_run:
            self.runtime.connected = True
            self.runtime.last_error = ""
            self.last_backend_message = (
                f"Bridge connected in dry-run mode using backend '{self.backend_type}'."
            )
            return self.runtime.response(
                success=True,
                message=self.last_backend_message,
                supported=True,
                warnings=list(self.capabilities_report().get("warnings", [])),
            )
        return self._connect_real()

    def _connect_real(self) -> dict[str, Any]:
        refusal = "Real backend connection is not implemented."
        self.runtime.connected = False
        self.runtime.last_error = refusal
        self.last_backend_message = refusal
        return self.runtime.response(
            success=False,
            message=refusal,
            supported=False,
            refusal_reason=refusal,
            warnings=list(self.capabilities_report().get("warnings", [])),
            raw_backend_message=self.last_raw_backend_message,
        )

    def disconnect(self) -> dict[str, Any]:
        self.runtime.connected = False
        self.runtime.busy = False
        self.runtime.abort_requested = False
        self.last_backend_message = "Bridge disconnected."
        return self.runtime.response(
            success=True,
            message=self.last_backend_message,
            supported=True,
        )

    def abort(self) -> dict[str, Any]:
        if self.runtime.busy:
            self.runtime.abort_requested = True
            self.last_backend_message = "Abort requested."
            return self.runtime.response(
                success=True,
                message=self.last_backend_message,
                supported=True,
            )
        return self.runtime.response(
            success=False,
            message="No active bridge execution to abort.",
            supported=False,
            refusal_reason="No active bridge execution to abort.",
        )

    def execute_primitive(self, payload: dict[str, Any]) -> dict[str, Any]:
        primitive = str(payload.get("primitive", "")).strip()
        command_id = str(payload.get("command_id") or _new_id("command"))

        if primitive not in SUPPORTED_PRIMITIVES:
            refusal = f"Unsupported primitive: {primitive or 'unknown'}."
            return self.runtime.response(
                success=False,
                message=refusal,
                error=refusal,
                supported=False,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
            )

        if self.runtime.so100_config.require_connection_before_execute and not self.runtime.connected:
            refusal = "Bridge is not connected. Call connect first."
            return self.runtime.response(
                success=False,
                message=refusal,
                error=refusal,
                supported=True,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
            )

        if self.runtime.abort_requested:
            self.runtime.abort_requested = False
            refusal = "Bridge execution aborted before primitive start."
            return self.runtime.response(
                success=False,
                message=refusal,
                error=refusal,
                supported=True,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
            )

        self.runtime.last_command = primitive
        if self.runtime.dry_run:
            return self._execute_primitive_dry_run(primitive, payload, command_id)
        return self._execute_primitive_real(primitive, payload, command_id)

    def _execute_primitive_dry_run(
        self,
        primitive: str,
        payload: dict[str, Any],
        command_id: str,
    ) -> dict[str, Any]:
        print(f"Bridge dry-run: {primitive} {json.dumps(payload, sort_keys=True)}")
        time.sleep(0.04)
        self._apply_state(primitive, payload)
        self.runtime.last_error = ""
        self.last_backend_message = f"Dry run only: would execute {primitive}."
        self.last_raw_backend_message = json.dumps(payload, sort_keys=True)
        return self.runtime.response(
            success=True,
            message=self.last_backend_message,
            command_id=command_id,
            primitive=primitive,
            supported=True,
            executed_steps=1,
            warnings=list(self.capabilities_report().get("warnings", [])),
            raw_backend_message=self.last_raw_backend_message,
        )

    def _execute_primitive_real(
        self,
        primitive: str,
        payload: dict[str, Any],
        command_id: str,
    ) -> dict[str, Any]:
        del primitive, payload, command_id
        refusal = "Real bridge backend is not implemented."
        self.runtime.last_error = refusal
        self.last_backend_message = refusal
        return self.runtime.response(
            success=False,
            message=refusal,
            supported=False,
            refusal_reason=refusal,
            raw_backend_message=self.last_raw_backend_message,
        )

    def execute_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(payload.get("plan_id") or _new_id("plan"))
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            refusal = "Plan payload is missing steps."
            return self.runtime.response(
                success=False,
                message=refusal,
                error=refusal,
                supported=False,
                refusal_reason=refusal,
                plan_id=plan_id,
                executed_steps=0,
                failed_step="plan",
            )

        if self.runtime.so100_config.require_connection_before_execute and not self.runtime.connected:
            refusal = "Bridge is not connected. Call connect first."
            return self.runtime.response(
                success=False,
                message=refusal,
                error=refusal,
                supported=True,
                refusal_reason=refusal,
                plan_id=plan_id,
                executed_steps=0,
                failed_step="plan",
            )

        self.runtime.busy = True
        self.runtime.abort_requested = False
        step_results: list[dict[str, Any]] = []
        completed_steps = 0
        final_success = True
        final_message = "Plan completed."
        failed_step = ""

        for index, step in enumerate(steps, start=1):
            command_payload = dict(step)
            primitive_name = str(command_payload.get("primitive", ""))
            command_payload.setdefault("command_id", f"{plan_id}-step-{index}")
            response = self.execute_primitive(command_payload)
            step_results.append(
                {
                    "step_index": index,
                    "primitive": primitive_name,
                    "success": bool(response.get("success", False)),
                    "supported": bool(response.get("supported", True)),
                    "message": response.get("message", ""),
                    "command_id": response.get("command_id"),
                    "refusal_reason": response.get("refusal_reason", ""),
                    "warnings": list(response.get("warnings", [])),
                    "raw_backend_message": response.get("raw_backend_message", ""),
                }
            )
            if response.get("success"):
                completed_steps += 1
                continue
            final_success = False
            final_message = str(response.get("message", "Plan failed."))
            failed_step = primitive_name
            if self.runtime.abort_requested:
                final_message = "Plan aborted."
            break

        self.runtime.busy = False
        warnings = list(self.capabilities_report().get("warnings", []))
        if final_success and self.runtime.dry_run:
            final_message = f"Dry run only. {completed_steps}/{len(steps)} bridge steps simulated."
        elif final_success:
            final_message = f"Executed {completed_steps}/{len(steps)} bridge steps."

        return self.runtime.response(
            success=final_success,
            message=final_message,
            plan_id=plan_id,
            step_results=step_results,
            supported=all(item["supported"] for item in step_results) if step_results else True,
            executed_steps=completed_steps,
            failed_step=failed_step,
            refusal_reason="" if final_success else final_message,
            warnings=warnings,
            raw_backend_message=self.last_raw_backend_message,
        )

    def _apply_state(self, primitive: str, payload: dict[str, Any]) -> None:
        zone = str(payload.get("zone", self.runtime.current_zone or "home"))
        object_name = str(payload.get("object_name", "object"))

        if primitive == "move_above_object":
            self.runtime.current_zone = zone
            self.runtime.is_home = False
        elif primitive == "grasp_object":
            self.runtime.held_object = object_name
            self.runtime.is_home = False
        elif primitive == "lift_object":
            self.runtime.is_home = False
        elif primitive == "move_to_zone":
            self.runtime.current_zone = zone
            self.runtime.is_home = False
        elif primitive == "place_object":
            self.runtime.current_zone = zone
            self.runtime.held_object = None
            self.runtime.last_target_handled = object_name
            self.runtime.is_home = False
        elif primitive in {"home", "return_home"}:
            self.runtime.current_zone = "home"
            self.runtime.held_object = None
            self.runtime.is_home = True


class StubBridgeBackend(BaseBridgeBackend):
    """Dry-run capable bridge backend with honest real-mode refusal."""

    backend_type = "stub"
    backend_name = "bridge:stub"

    def capabilities_report(self) -> dict[str, Any]:
        warnings = []
        if self.runtime.dry_run:
            warnings.append("Stub backend is running in dry-run mode only.")
        return {
            "backend_selected": self.backend_name,
            "readiness_stage": "dry_run_only_stub" if self.runtime.dry_run else "unavailable",
            "lerobot_available": False,
            "feetech_support_available": False,
            "backend_ready_for_real_execution": False,
            "real_execution_allowed_by_config": self.runtime.so100_config.allow_real_execution,
            "can_connect": False,
            "can_home": False,
            "can_execute_motion_primitives": False,
            "can_execute_gripper_primitives": False,
            "can_move_above_object": False,
            "can_grasp": False,
            "can_lift": False,
            "can_move_to_zone": False,
            "can_place": False,
            "not_ready_reasons": ["Stub backend does not support real SO100 execution."],
            "warnings": warnings,
        }

    def _connect_real(self) -> dict[str, Any]:
        refusal = "Stub bridge backend cannot connect for real SO100 motor execution."
        self.runtime.connected = False
        self.runtime.last_error = refusal
        self.last_backend_message = refusal
        return self.runtime.response(
            success=False,
            message=refusal,
            error=refusal,
            supported=False,
            refusal_reason=refusal,
            warnings=list(self.capabilities_report().get("warnings", [])),
        )

    def _execute_primitive_real(
        self,
        primitive: str,
        payload: dict[str, Any],
        command_id: str,
    ) -> dict[str, Any]:
        del primitive, payload, command_id
        refusal = "Stub bridge backend cannot execute real SO100 motor commands."
        self.runtime.last_error = refusal
        self.last_backend_message = refusal
        return self.runtime.response(
            success=False,
            message=refusal,
            error=refusal,
            supported=False,
            refusal_reason=refusal,
            warnings=list(self.capabilities_report().get("warnings", [])),
        )


class LeRobotSO100Backend(BaseBridgeBackend):
    """Optional real backend path for LeRobot + Feetech based SO100 control."""

    backend_type = "lerobot_so100"
    backend_name = "bridge:lerobot_so100"

    def __init__(self, runtime: "BridgeRuntime") -> None:
        super().__init__(runtime)
        self.robot: Any | None = None
        self._factory: Callable[[Any], Any] | None = None
        self._config_class: type[Any] | None = None
        self._home_method_name = ""
        self.dependencies = self._probe_dependencies()

    def _probe_dependencies(self) -> LeRobotDependencyStatus:
        status = LeRobotDependencyStatus()

        try:
            importlib.import_module("lerobot")
            status.lerobot_available = True
        except Exception as error:  # pragma: no cover - depends on local install
            status.lerobot_import_error = str(error)

        factory_candidates = [
            ("lerobot.common.robot_devices.robots.utils", "make_robot_from_config"),
            ("lerobot.common.robots", "make_robot_from_config"),
            ("lerobot.robots", "make_robot_from_config"),
        ]
        for module_path, attr_name in factory_candidates:
            try:
                module = importlib.import_module(module_path)
                candidate = getattr(module, attr_name, None)
            except Exception as error:  # pragma: no cover - depends on local install
                status.factory_error = str(error)
                continue
            if callable(candidate):
                self._factory = candidate
                status.factory_available = True
                status.selected_factory_path = f"{module_path}.{attr_name}"
                break

        config_candidates = [
            ("lerobot.common.robot_devices.robots.configs", "So100RobotConfig"),
            ("lerobot.common.robot_devices.robots.configs", "So101RobotConfig"),
        ]
        for module_path, attr_name in config_candidates:
            try:
                module = importlib.import_module(module_path)
                candidate = getattr(module, attr_name, None)
            except Exception as error:  # pragma: no cover - depends on local install
                status.config_class_error = str(error)
                continue
            if isinstance(candidate, type):
                self._config_class = candidate
                status.config_class_available = True
                status.selected_config_class_path = f"{module_path}.{attr_name}"
                status.selected_config_class_name = attr_name
                break

        feetech_candidates = [
            "lerobot.common.robot_devices.motors.feetech",
            "lerobot.common.motors.feetech",
            "feetech",
        ]
        for module_path in feetech_candidates:
            try:
                importlib.import_module(module_path)
            except Exception as error:  # pragma: no cover - depends on local install
                status.feetech_import_error = str(error)
                continue
            status.feetech_support_available = True
            status.selected_feetech_path = module_path
            break

        return status

    def _not_ready_reasons(self) -> list[str]:
        reasons: list[str] = []
        config = self.runtime.so100_config
        if not config.use_lerobot_backend:
            reasons.append("use_lerobot_backend is disabled in config/so100_config.json")
        if not config.allow_real_execution:
            reasons.append("allow_real_execution is false in config/so100_config.json")
        if not self.dependencies.lerobot_available:
            reasons.append(
                f"LeRobot is not importable: {self.dependencies.lerobot_import_error or 'missing'}"
            )
        if not self.dependencies.feetech_support_available:
            reasons.append(
                "Feetech support is not importable."
                + (
                    f" Import error: {self.dependencies.feetech_import_error}"
                    if self.dependencies.feetech_import_error
                    else ""
                )
            )
        if not self.dependencies.factory_available:
            reasons.append(
                "Could not resolve make_robot_from_config in the installed LeRobot package."
            )
        if not self.dependencies.config_class_available:
            reasons.append(
                "Could not resolve So100RobotConfig or So101RobotConfig in the installed LeRobot package."
            )
        if not self.runtime.so100_config.follower_port:
            reasons.append("follower_port is not configured in config/so100_config.json")
        if self.runtime.so100_config.robot_type not in {"so100", "so101"}:
            reasons.append(
                f"robot_type '{self.runtime.so100_config.robot_type}' is not supported by this backend."
            )
        return reasons

    def _can_connect_real(self) -> bool:
        return not self._not_ready_reasons()

    def _can_home_real(self) -> bool:
        return self.runtime.connected and bool(self._home_method_name)

    def capabilities_report(self) -> dict[str, Any]:
        not_ready = self._not_ready_reasons()
        can_connect = not not_ready
        can_home = self._can_home_real()
        can_motion = False
        can_gripper = False
        stage = "stage_a_dependency_probe"
        if self.runtime.dry_run:
            stage = "dry_run_bridge"
        elif can_connect:
            stage = "stage_b_connect_ready"
        if can_home:
            stage = "stage_b_connect_and_home"

        warnings: list[str] = []
        if self.runtime.dry_run:
            warnings.append("Bridge is in dry-run mode. No real motor commands will be sent.")
        if self.runtime.so100_config.home_on_connect:
            warnings.append(
                "home_on_connect is configured, but automatic homing is disabled in the bridge for safety."
            )

        return {
            "backend_selected": self.backend_name,
            "readiness_stage": stage,
            "lerobot_available": self.dependencies.lerobot_available,
            "feetech_support_available": self.dependencies.feetech_support_available,
            "backend_ready_for_real_execution": can_connect and can_home,
            "real_execution_allowed_by_config": self.runtime.so100_config.allow_real_execution,
            "can_connect": can_connect,
            "can_home": can_home,
            "can_execute_motion_primitives": can_motion,
            "can_execute_gripper_primitives": can_gripper,
            "can_move_above_object": False,
            "can_grasp": False,
            "can_lift": False,
            "can_move_to_zone": False,
            "can_place": False,
            "not_ready_reasons": not_ready,
            "warnings": warnings,
        }

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_type": self.backend_type,
            "robot_config": self.runtime.so100_config.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "resolved_home_method": self._home_method_name or "unresolved",
            "last_backend_message": self.last_backend_message,
            "raw_backend_message": self.last_raw_backend_message,
        }

    def _build_robot_config_instance(self) -> Any:
        if self._config_class is None:
            raise RuntimeError("LeRobot robot config class is not available.")

        config_instance = self._config_class()
        config = self.runtime.so100_config

        for attr_name, value in (
            ("robot_type", config.robot_type),
            ("teleop_type", config.teleop_type),
            ("follower_port", config.follower_port),
            ("leader_port", config.leader_port),
            ("robot_id", config.robot_id),
            ("id", config.robot_id),
            ("port", config.follower_port),
            ("cameras", {}),
        ):
            if hasattr(config_instance, attr_name):
                setattr(config_instance, attr_name, value)

        if hasattr(config_instance, "leader_arms") and not config.leader_port:
            setattr(config_instance, "leader_arms", {})

        return config_instance

    def _resolve_home_method(self, robot: Any) -> str:
        for method_name in ("home", "go_home"):
            if callable(getattr(robot, method_name, None)):
                return method_name
        return ""

    def _connect_real(self) -> dict[str, Any]:
        reasons = self._not_ready_reasons()
        if reasons:
            refusal = reasons[0]
            self.runtime.connected = False
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
                warnings=list(self.capabilities_report().get("warnings", [])),
            )

        try:
            config_instance = self._build_robot_config_instance()
            self.robot = self._factory(config_instance) if self._factory is not None else None
        except Exception as error:  # pragma: no cover - depends on local install
            refusal = f"Failed to build LeRobot SO100 object: {error}"
            self.runtime.connected = False
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            self.last_raw_backend_message = repr(error)
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
                raw_backend_message=self.last_raw_backend_message,
            )

        if self.robot is None:
            refusal = "LeRobot factory returned no robot instance."
            self.runtime.connected = False
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
            )

        connect_method = getattr(self.robot, "connect", None)
        if not callable(connect_method):
            refusal = "Constructed LeRobot object does not expose a connect() method."
            self.runtime.connected = False
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
            )

        try:
            raw_message = connect_method()
        except Exception as error:  # pragma: no cover - depends on local install
            refusal = f"LeRobot connect() failed: {error}"
            self.runtime.connected = False
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            self.last_raw_backend_message = repr(error)
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
                raw_backend_message=self.last_raw_backend_message,
            )

        self.runtime.connected = True
        self.runtime.last_error = ""
        self._home_method_name = self._resolve_home_method(self.robot)
        self.last_backend_message = "LeRobot SO100 backend connected."
        self.last_raw_backend_message = "" if raw_message is None else str(raw_message)
        warnings = list(self.capabilities_report().get("warnings", []))
        if not self._home_method_name:
            warnings.append(
                "Connected to the LeRobot backend, but no home()/go_home() method was detected."
            )
        return self.runtime.response(
            success=True,
            message=self.last_backend_message,
            supported=True,
            warnings=warnings,
            raw_backend_message=self.last_raw_backend_message,
        )

    def disconnect(self) -> dict[str, Any]:
        if not self.runtime.connected:
            return super().disconnect()

        disconnect_method = getattr(self.robot, "disconnect", None)
        if callable(disconnect_method):
            try:
                raw_message = disconnect_method()
            except Exception as error:  # pragma: no cover - depends on local install
                refusal = f"LeRobot disconnect() failed: {error}"
                self.runtime.last_error = refusal
                self.last_backend_message = refusal
                self.last_raw_backend_message = repr(error)
                return self.runtime.response(
                    success=False,
                    message=refusal,
                    supported=False,
                    refusal_reason=refusal,
                    raw_backend_message=self.last_raw_backend_message,
                )
            self.last_raw_backend_message = "" if raw_message is None else str(raw_message)

        self.robot = None
        self._home_method_name = ""
        self.runtime.connected = False
        self.runtime.busy = False
        self.runtime.abort_requested = False
        self.runtime.last_error = ""
        self.last_backend_message = "LeRobot SO100 backend disconnected."
        return self.runtime.response(
            success=True,
            message=self.last_backend_message,
            supported=True,
            raw_backend_message=self.last_raw_backend_message,
        )

    def _execute_home_real(self, primitive: str, command_id: str) -> dict[str, Any]:
        if not self.runtime.connected or self.robot is None:
            refusal = "LeRobot backend is not connected."
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=True,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
            )

        if not self._home_method_name:
            refusal = (
                "Real home is not available because the connected LeRobot object does not expose "
                "home() or go_home()."
            )
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=False,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
            )

        method = getattr(self.robot, self._home_method_name, None)
        try:
            raw_message = method()
        except Exception as error:  # pragma: no cover - depends on local install
            refusal = f"LeRobot {self._home_method_name}() failed: {error}"
            self.runtime.last_error = refusal
            self.last_backend_message = refusal
            self.last_raw_backend_message = repr(error)
            return self.runtime.response(
                success=False,
                message=refusal,
                supported=True,
                refusal_reason=refusal,
                command_id=command_id,
                primitive=primitive,
                executed_steps=0,
                failed_step=primitive,
                raw_backend_message=self.last_raw_backend_message,
            )

        self._apply_state("return_home", {})
        self.runtime.last_error = ""
        self.last_backend_message = "LeRobot home completed."
        self.last_raw_backend_message = "" if raw_message is None else str(raw_message)
        return self.runtime.response(
            success=True,
            message=self.last_backend_message,
            supported=True,
            command_id=command_id,
            primitive=primitive,
            executed_steps=1,
            warnings=list(self.capabilities_report().get("warnings", [])),
            raw_backend_message=self.last_raw_backend_message,
        )

    def _real_primitive_placeholder_reason(self, primitive: str) -> str:
        requirements = {
            "move_above_object": "calibrated zones plus joint targets or IK/motion planning",
            "grasp_object": "a gripper close/open mapping for the SO100 end effector",
            "lift_object": "safe lift joint targets or a motion planner",
            "move_to_zone": "calibrated zone targets plus joint targets or IK/motion planning",
            "place_object": "place joint targets, zone calibration, and gripper release mapping",
        }
        needed = requirements.get(primitive, "additional backend motion mapping")
        return f"Real primitive '{primitive}' is not implemented yet. It requires {needed}."

    def _execute_primitive_real(
        self,
        primitive: str,
        payload: dict[str, Any],
        command_id: str,
    ) -> dict[str, Any]:
        del payload
        if primitive in REAL_HOME_PRIMITIVES:
            return self._execute_home_real(primitive, command_id)

        refusal = self._real_primitive_placeholder_reason(primitive)
        self.runtime.last_error = refusal
        self.last_backend_message = refusal
        return self.runtime.response(
            success=False,
            message=refusal,
            supported=False,
            refusal_reason=refusal,
            command_id=command_id,
            primitive=primitive,
            executed_steps=0,
            failed_step=primitive,
            warnings=list(self.capabilities_report().get("warnings", [])),
        )


class BridgeRuntime:
    """Mutable runtime state shared across bridge requests."""

    def __init__(
        self,
        config: BridgeConfig,
        so100_config: SO100Config,
        backend_type: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.config = config
        self.so100_config = so100_config
        self.backend_type = backend_type or config.backend_type
        default_dry_run = so100_config.default_dry_run
        self.dry_run = default_dry_run if dry_run is None else dry_run
        self.started_at = _now()
        self.connected = False
        self.busy = False
        self.abort_requested = False
        self.last_command = ""
        self.last_error = ""
        self.last_connection_attempt: float | None = None
        self.current_zone: str | None = "home"
        self.held_object: str | None = None
        self.last_target_handled: str | None = None
        self.is_home = True
        self.last_result: dict[str, Any] | None = None
        self.backend = self._build_backend(self.backend_type)

    def _build_backend(self, backend_type: str) -> BaseBridgeBackend:
        if backend_type == "lerobot_so100":
            return LeRobotSO100Backend(self)
        return StubBridgeBackend(self)

    def response(self, *, success: bool, message: str, **extra: Any) -> dict[str, Any]:
        response = {
            "success": success,
            "mode": "bridge",
            "message": message,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "config": self.config.to_dict(),
            "robot_config": self.so100_config.to_dict(),
        }
        response.update(self.backend.status_snapshot())
        response.update(extra)
        if not success and "error" not in response:
            response["error"] = message
        self.last_result = response
        return response

    def status(self) -> dict[str, Any]:
        return self.response(success=True, message="Bridge status ready.")

    def capabilities(self) -> dict[str, Any]:
        return self.response(
            success=True,
            message="Bridge capabilities ready.",
            capabilities=self.backend.capabilities_report(),
        )

    def backend_info(self) -> dict[str, Any]:
        return self.response(
            success=True,
            message="Bridge backend info ready.",
            backend_info=self.backend.backend_info(),
        )

    def hardware_ready(self) -> dict[str, Any]:
        capabilities = self.backend.capabilities_report()
        ready = bool(
            self.connected
            and not self.dry_run
            and capabilities.get("backend_ready_for_real_execution", False)
        )
        reasons = list(capabilities.get("not_ready_reasons", []))
        if self.dry_run:
            reasons.insert(0, "Bridge is in dry-run mode.")
        if not self.connected:
            reasons.insert(0, "Bridge is not connected.")
        message = "Bridge is ready for real SO100 execution." if ready else "Bridge is not ready for real SO100 execution."
        return self.response(
            success=True,
            message=message,
            ready=ready,
            reasons=reasons,
        )

    def set_dry_run(self, enabled: bool) -> dict[str, Any]:
        if self.dry_run == enabled:
            state_text = "on" if enabled else "off"
            return self.response(success=True, message=f"Bridge dry-run is already {state_text}.")
        self.dry_run = enabled
        self.connected = False
        self.abort_requested = False
        self.last_error = ""
        state_text = "on" if enabled else "off"
        return self.response(
            success=True,
            message=f"Bridge dry-run is now {state_text}. Reconnect before executing.",
        )


class BridgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the bridge service."""

    runtime: BridgeRuntime
    server_ref: ThreadingHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(self.runtime.response(success=True, message="Bridge is healthy."))
            return
        if self.path == "/status":
            self._send_json(self.runtime.status())
            return
        if self.path == "/capabilities":
            self._send_json(self.runtime.capabilities())
            return
        if self.path == "/backend-info":
            self._send_json(self.runtime.backend_info())
            return
        if self.path == "/hardware-ready":
            self._send_json(self.runtime.hardware_ready())
            return
        self._send_json(
            self.runtime.response(success=False, message="Unknown bridge endpoint."),
            status_code=404,
        )

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()

        if self.path == "/connect":
            self._send_json(self.runtime.backend.connect())
            return
        if self.path == "/disconnect":
            self._send_json(self.runtime.backend.disconnect())
            return
        if self.path == "/dryrun":
            enabled = bool(payload.get("enabled", True))
            self._send_json(self.runtime.set_dry_run(enabled))
            return
        if self.path == "/primitive":
            self._send_json(self.runtime.backend.execute_primitive(payload))
            return
        if self.path == "/plan":
            self._send_json(self.runtime.backend.execute_plan(payload))
            return
        if self.path == "/abort":
            self._send_json(self.runtime.backend.abort())
            return
        if self.path == "/shutdown":
            self._send_json(
                self.runtime.response(success=True, message="Bridge shutdown requested.")
            )
            threading.Thread(target=self.server_ref.shutdown, daemon=True).start()
            return

        self._send_json(
            self.runtime.response(success=False, message="Unknown bridge endpoint."),
            status_code=404,
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        response_body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def build_runtime_from_args() -> tuple[BridgeRuntime, str, int]:
    parser = argparse.ArgumentParser(description="SO100 bridge service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--dry-run", default=None, choices={"on", "off"})
    args = parser.parse_args()

    config = load_bridge_config()
    so100_config = load_so100_config()
    host = args.host or config.host
    port = args.port or config.port
    backend_type = args.backend or config.backend_type
    dry_run = so100_config.default_dry_run if args.dry_run is None else args.dry_run == "on"
    runtime = BridgeRuntime(
        config=config,
        so100_config=so100_config,
        backend_type=backend_type,
        dry_run=dry_run,
    )
    return runtime, host, port


def main() -> None:
    runtime, host, port = build_runtime_from_args()
    handler = type(
        "ConfiguredBridgeRequestHandler",
        (BridgeRequestHandler,),
        {"runtime": runtime},
    )
    server = ThreadingHTTPServer((host, port), handler)
    handler.server_ref = server
    print(
        f"SO100 bridge listening on {host}:{port} "
        f"(backend={runtime.backend_type}, dry_run={'on' if runtime.dry_run else 'off'})"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
