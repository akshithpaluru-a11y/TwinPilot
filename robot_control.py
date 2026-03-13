"""Execution layer for the local tabletop robot MVP."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from bridge_client import get_bridge_client
from cyberwave_client import (
    SUPPORTED_SIM_ACTIONS,
    VALID_SIM_ZONES,
    get_cyberwave_client,
)
from openai_planner import PlannerAction, PlannerResponse, get_simulation_planner

ALLOWED_ZONES = {"left", "center", "right"}
SUPPORTED_LABELS = {"bottle", "cup", "object", "cube", "block", "tray", "marker"}
SIMULATION_SPEEDS = {
    "fast": 0.02,
    "normal": 0.06,
    "slow": 0.14,
}
SUPPORTED_PRIMITIVES = {
    "home",
    "move_above_object",
    "grasp_object",
    "lift_object",
    "move_to_zone",
    "place_object",
    "return_home",
}
OBJECT_TARGET_STEP_ACTIONS = {
    "move_above_object",
    "grasp_object",
    "lift_object",
    "place_object",
}


def _now() -> float:
    return time.time()


def _timestamp_label(timestamp: float | None) -> str:
    if timestamp is None:
        return "n/a"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class ExecutionStep:
    """Single robot step within a structured execution plan."""

    step_id: str
    action: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: float | None = None
    completed_at: float | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "description": self.description,
            "params": dict(self.params),
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


@dataclass
class PrimitiveCommand:
    """Low-level primitive payload shared with the SO100 bridge."""

    primitive_id: str
    primitive_name: str
    params: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "command_id": self.primitive_id,
            "primitive": self.primitive_name,
        }
        payload.update(self.params)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "primitive_name": self.primitive_name,
            "params": dict(self.params),
            "summary": self.summary,
        }


@dataclass
class ValidationIssue:
    """Pre-execution issue found while validating a plan."""

    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """Validation result for a proposed execution plan."""

    can_execute: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_execute": self.can_execute,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class ExecutionPlan:
    """Structured plan that can be printed or executed by a controller."""

    plan_id: str
    action_type: str
    target_object: dict[str, Any] | None
    target_object_id: str | None
    target_name: str | None
    target_label: str | None
    source_zone: str | None
    destination_zone: str | None
    steps: list[ExecutionStep]
    confirmation_required: bool
    execution_mode: str
    scene_source: str
    fallback_used: bool
    scene_stale: bool
    scene_age_seconds: float | None
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "action_type": self.action_type,
            "target_object": None if self.target_object is None else dict(self.target_object),
            "target_object_id": self.target_object_id,
            "target_name": self.target_name,
            "target_label": self.target_label,
            "source_zone": self.source_zone,
            "destination_zone": self.destination_zone,
            "steps": [step.to_dict() for step in self.steps],
            "confirmation_required": self.confirmation_required,
            "execution_mode": self.execution_mode,
            "scene_source": self.scene_source,
            "fallback_used": self.fallback_used,
            "scene_stale": self.scene_stale,
            "scene_age_seconds": self.scene_age_seconds,
            "created_at": self.created_at,
        }


@dataclass
class StepExecutionResult:
    """Execution outcome for one step."""

    step_id: str
    description: str
    status: str
    message: str
    primitive_name: str = ""
    dry_run: bool = False
    supported: bool = True
    refusal_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_backend_message: str = ""
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "status": self.status,
            "message": self.message,
            "primitive_name": self.primitive_name,
            "dry_run": self.dry_run,
            "supported": self.supported,
            "refusal_reason": self.refusal_reason,
            "warnings": list(self.warnings),
            "raw_backend_message": self.raw_backend_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ExecutionResult:
    """Top-level result of attempting to execute a plan."""

    plan_id: str
    status: str
    message: str
    controller_mode: str
    controller_name: str
    execution_enabled: bool
    completed_steps: int
    total_steps: int
    backend_name: str = ""
    dry_run: bool = False
    aborted: bool = False
    supported: bool = True
    failed_step: str = ""
    refusal_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_backend_message: str = ""
    step_results: list[StepExecutionResult] = field(default_factory=list)
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "message": self.message,
            "controller_mode": self.controller_mode,
            "controller_name": self.controller_name,
            "execution_enabled": self.execution_enabled,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "backend_name": self.backend_name,
            "dry_run": self.dry_run,
            "aborted": self.aborted,
            "supported": self.supported,
            "failed_step": self.failed_step,
            "refusal_reason": self.refusal_reason,
            "warnings": list(self.warnings),
            "raw_backend_message": self.raw_backend_message,
            "step_results": [step.to_dict() for step in self.step_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class RobotSessionState:
    """Session-persistent robot state used by status commands."""

    mode: str = "simulation"
    execution_enabled: bool = False
    speed: str = "normal"
    controller_name: str = "simulation-controller"
    busy: bool = False
    connected: bool = True
    is_home: bool = True
    held_object: str | None = None
    current_zone: str | None = "home"
    last_target_handled: str | None = None
    last_plan_id: str | None = None
    last_result_status: str | None = None
    hardware_backend: str = "bridge:stub"
    hardware_backend_type: str = "stub"
    hardware_dry_run: bool = True
    hardware_ready: bool = False
    hardware_real_ready: bool = False
    last_hardware_error: str = ""
    bridge_running: bool = False
    bridge_connected: bool = False
    bridge_dry_run: bool = True
    bridge_last_command: str = ""
    bridge_last_error: str = ""
    bridge_config_source: str = "defaults"
    bridge_config_path: str = ""
    last_connection_attempt: float | None = None
    abort_requested: bool = False
    lerobot_available: bool = False
    feetech_support_available: bool = False
    backend_ready_for_real_execution: bool = False
    can_connect: bool = False
    can_home: bool = False
    can_move_above_object: bool = False
    can_grasp: bool = False
    can_lift: bool = False
    can_move_to_zone: bool = False
    can_place: bool = False
    cyberwave_connected: bool = False
    cyberwave_backend: str = "cyberwave:local-scene"
    cyberwave_local_ready: bool = True
    cyberwave_remote_ready: bool = False
    cyberwave_active_condition: str = "demo_showcase_scene"
    cyberwave_last_error: str = ""
    cyberwave_openai_enabled: bool = True
    cyberwave_openai_configured: bool = False
    cyberwave_openai_available: bool = False
    cyberwave_openai_model: str = "gpt-5-mini"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "execution_enabled": self.execution_enabled,
            "speed": self.speed,
            "controller_name": self.controller_name,
            "busy": self.busy,
            "connected": self.connected,
            "is_home": self.is_home,
            "held_object": self.held_object,
            "current_zone": self.current_zone,
            "last_target_handled": self.last_target_handled,
            "last_plan_id": self.last_plan_id,
            "last_result_status": self.last_result_status,
            "hardware_backend": self.hardware_backend,
            "hardware_backend_type": self.hardware_backend_type,
            "hardware_dry_run": self.hardware_dry_run,
            "hardware_ready": self.hardware_ready,
            "hardware_real_ready": self.hardware_real_ready,
            "last_hardware_error": self.last_hardware_error,
            "bridge_running": self.bridge_running,
            "bridge_connected": self.bridge_connected,
            "bridge_dry_run": self.bridge_dry_run,
            "bridge_last_command": self.bridge_last_command,
            "bridge_last_error": self.bridge_last_error,
            "bridge_config_source": self.bridge_config_source,
            "bridge_config_path": self.bridge_config_path,
            "last_connection_attempt": self.last_connection_attempt,
            "abort_requested": self.abort_requested,
            "lerobot_available": self.lerobot_available,
            "feetech_support_available": self.feetech_support_available,
            "backend_ready_for_real_execution": self.backend_ready_for_real_execution,
            "can_connect": self.can_connect,
            "can_home": self.can_home,
            "can_move_above_object": self.can_move_above_object,
            "can_grasp": self.can_grasp,
            "can_lift": self.can_lift,
            "can_move_to_zone": self.can_move_to_zone,
            "can_place": self.can_place,
            "cyberwave_connected": self.cyberwave_connected,
            "cyberwave_backend": self.cyberwave_backend,
            "cyberwave_local_ready": self.cyberwave_local_ready,
            "cyberwave_remote_ready": self.cyberwave_remote_ready,
            "cyberwave_active_condition": self.cyberwave_active_condition,
            "cyberwave_last_error": self.cyberwave_last_error,
            "cyberwave_openai_enabled": self.cyberwave_openai_enabled,
            "cyberwave_openai_configured": self.cyberwave_openai_configured,
            "cyberwave_openai_available": self.cyberwave_openai_available,
            "cyberwave_openai_model": self.cyberwave_openai_model,
        }


class RobotController:
    """Common interface for simulation and hardware controllers."""

    name = "controller"
    mode = "unknown"

    def is_available(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return bool(self.get_state().get("connected", False))

    def is_dry_run(self) -> bool:
        return False

    def get_backend_name(self) -> str:
        return self.name

    def set_speed(self, speed: str) -> None:
        del speed

    def connect(self) -> tuple[bool, str]:
        return True, f"{self.name} ready."

    def disconnect(self) -> tuple[bool, str]:
        return True, f"{self.name} disconnected."

    def abort(self) -> tuple[bool, str]:
        return False, "Abort is not supported by this controller."

    def get_readiness(self) -> dict[str, Any]:
        return {
            "ready": True,
            "real_backend_ready": True,
            "warnings": [],
            "errors": [],
            "backend_name": self.get_backend_name(),
            "dry_run": self.is_dry_run(),
        }

    def move_above_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        raise NotImplementedError

    def grasp_object(self, object_name: str) -> tuple[bool, str]:
        raise NotImplementedError

    def lift_object(self, object_name: str) -> tuple[bool, str]:
        raise NotImplementedError

    def move_to_zone(self, zone: str) -> tuple[bool, str]:
        raise NotImplementedError

    def place_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        raise NotImplementedError

    def home(self) -> tuple[bool, str]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        raise NotImplementedError


class SimulationRobotController(RobotController):
    """Dry-run controller that simulates robot motion locally."""

    name = "simulation-controller"
    mode = "simulation"

    def __init__(self) -> None:
        self.connected = True
        self.busy = False
        self.is_home = True
        self.held_object: str | None = None
        self.current_zone: str | None = "home"
        self.last_target_handled: str | None = None
        self.speed = "normal"
        self.abort_requested = False

    def set_speed(self, speed: str) -> None:
        if speed in SIMULATION_SPEEDS:
            self.speed = speed

    def connect(self) -> tuple[bool, str]:
        self.connected = True
        return True, "Simulation controller ready."

    def disconnect(self) -> tuple[bool, str]:
        self.connected = False
        return True, "Simulation controller disconnected."

    def abort(self) -> tuple[bool, str]:
        if self.busy:
            self.abort_requested = True
            return True, "Abort requested for the simulation controller."
        return False, "No active simulation execution to abort."

    def _pause(self) -> None:
        time.sleep(SIMULATION_SPEEDS[self.speed])

    def _log(self, message: str) -> None:
        print(f"Simulation: {message}")

    def move_above_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        self._log(f"moving above {object_name} in {zone} zone")
        self.is_home = False
        self.current_zone = zone
        self._pause()
        return True, f"Moved above {object_name} in {zone} zone."

    def grasp_object(self, object_name: str) -> tuple[bool, str]:
        self._log(f"grasping {object_name}")
        self.held_object = object_name
        self._pause()
        return True, f"Grasped {object_name}."

    def lift_object(self, object_name: str) -> tuple[bool, str]:
        self._log(f"lifting {object_name}")
        self._pause()
        return True, f"Lifted {object_name}."

    def move_to_zone(self, zone: str) -> tuple[bool, str]:
        self._log(f"moving to {zone} zone")
        self.current_zone = zone
        self.is_home = False
        self._pause()
        return True, f"Moved to {zone} zone."

    def place_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        self._log(f"placing {object_name} in {zone} zone")
        self.current_zone = zone
        self.last_target_handled = object_name
        self.held_object = None
        self._pause()
        return True, f"Placed {object_name} in {zone} zone."

    def home(self) -> tuple[bool, str]:
        self._log("returning home")
        self.current_zone = "home"
        self.is_home = True
        self.held_object = None
        self._pause()
        return True, "Returned to home position."

    def get_state(self) -> dict[str, Any]:
        return {
            "controller_name": self.name,
            "controller_mode": self.mode,
            "connected": self.connected,
            "busy": self.busy,
            "is_home": self.is_home,
            "held_object": self.held_object,
            "current_zone": self.current_zone,
            "last_target_handled": self.last_target_handled,
            "speed": self.speed,
            "backend_name": "simulation",
            "dry_run": False,
            "abort_requested": self.abort_requested,
        }

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        started_at = _now()
        step_results: list[StepExecutionResult] = []
        completed_steps = 0
        self.busy = True
        self.abort_requested = False

        dispatch = {
            "move_above_object": lambda step: self.move_above_object(
                str(step.params["object_name"]),
                str(step.params["zone"]),
            ),
            "grasp_object": lambda step: self.grasp_object(str(step.params["object_name"])),
            "lift_object": lambda step: self.lift_object(str(step.params["object_name"])),
            "move_to_zone": lambda step: self.move_to_zone(str(step.params["zone"])),
            "place_object": lambda step: self.place_object(
                str(step.params["object_name"]),
                str(step.params["zone"]),
            ),
            "return_home": lambda step: self.home(),
        }

        final_status = "success"
        final_message = "Plan completed successfully."
        aborted = False

        for step in plan.steps:
            if self.abort_requested:
                final_status = "aborted"
                final_message = "Simulation execution was aborted."
                aborted = True
                break

            step.started_at = _now()
            step.status = "running"
            executor = dispatch.get(step.action)
            if executor is None:
                success = False
                message = f"Unsupported simulation step: {step.action}"
            else:
                success, message = executor(step)
            step.completed_at = _now()
            step.status = "completed" if success else "failed"
            if not success:
                step.error = message

            step_results.append(
                StepExecutionResult(
                    step_id=step.step_id,
                    description=step.description,
                    status=step.status,
                    message=message,
                    primitive_name=step.action,
                    dry_run=False,
                    started_at=step.started_at,
                    completed_at=step.completed_at,
                )
            )

            if success:
                completed_steps += 1
                continue

            final_status = "failed"
            final_message = message
            break

        self.busy = False
        if final_status != "success":
            self.is_home = False

        return ExecutionResult(
            plan_id=plan.plan_id,
            status=final_status,
            message=final_message,
            controller_mode=self.mode,
            controller_name=self.name,
            execution_enabled=True,
            completed_steps=completed_steps,
            total_steps=len(plan.steps),
            backend_name="simulation",
            dry_run=False,
            aborted=aborted,
            step_results=step_results,
            started_at=started_at,
            completed_at=_now(),
        )


class CyberWaveSimulationController(RobotController):
    """Local-first simulation controller with optional CyberWave SDK session support."""

    name = "cyberwave-simulation-controller"
    mode = "cyberwave"

    def __init__(self) -> None:
        self.client = get_cyberwave_client()
        self.planner = get_simulation_planner(
            self.client.config.openai_model,
            enabled=self.client.config.use_openai_planner,
        )
        self.busy = False
        self.abort_requested = False
        self.speed = "normal"

    def is_available(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def is_dry_run(self) -> bool:
        return False

    def set_speed(self, speed: str) -> None:
        if speed in SIMULATION_SPEEDS:
            self.speed = speed

    def connect(self) -> tuple[bool, str]:
        return self.client.connect()

    def disconnect(self) -> tuple[bool, str]:
        return self.client.disconnect()

    def abort(self) -> tuple[bool, str]:
        if self.busy:
            self.abort_requested = True
            return True, "Abort requested for the CyberWave simulation controller."
        return False, "No active CyberWave simulation execution to abort."

    def get_backend_name(self) -> str:
        return str(self.client.status()["backend_name"])

    def planner_status(self) -> dict[str, Any]:
        planner_status = self.planner.status()
        planner_status["configured_model"] = self.client.config.openai_model
        planner_status["planner_enabled"] = self.client.config.use_openai_planner
        return planner_status

    def get_state(self) -> dict[str, Any]:
        status = self.client.status()
        state = self.client.get_scene_state()
        planner_status = self.planner_status()
        return {
            "controller_name": self.name,
            "controller_mode": self.mode,
            "connected": True,
            "busy": self.busy,
            "is_home": bool(state.claw_state["is_home"]),
            "held_object": state.claw_state.get("holding_display_name") or state.claw_state["holding"],
            "current_zone": state.claw_state["zone"],
            "last_target_handled": None,
            "speed": self.speed,
            "backend_name": status["backend_name"],
            "dry_run": False,
            "abort_requested": self.abort_requested,
            "cyberwave_connected": status["connected"],
            "cyberwave_backend": status["backend_name"],
            "cyberwave_local_ready": status["local_sim_ready"],
            "cyberwave_remote_ready": status["remote_ready"],
            "cyberwave_active_condition": status["active_condition"],
            "cyberwave_last_error": status["last_error"],
            "cyberwave_openai_enabled": planner_status["enabled"],
            "cyberwave_openai_configured": planner_status["api_key_configured"],
            "cyberwave_openai_available": planner_status["openai_available"],
            "cyberwave_openai_model": planner_status["model"],
            "cyberwave_scene_state": state.to_dict(),
            "cyberwave_status": status,
        }

    def get_readiness(self) -> dict[str, Any]:
        status = self.client.status()
        warnings: list[str] = []
        errors: list[str] = []
        if not status["local_sim_ready"]:
            errors.append("CyberWave local simulation is not ready.")
        if not status["connected"]:
            warnings.append("CyberWave is not connected. Execution will run in local-only simulation.")
        if not self.planner.status()["api_key_configured"]:
            warnings.append("OPENAI_API_KEY is not configured. Natural-language simulation planning will be unavailable.")
        if status["last_error"]:
            warnings.append(status["last_error"])
        return {
            "ready": status["local_sim_ready"],
            "real_backend_ready": status["connected"],
            "warnings": warnings,
            "errors": errors,
            "backend_name": status["backend_name"],
            "dry_run": False,
        }

    def summarize_status(self) -> str:
        return self.client.summarize_status()

    def summarize_scene(self) -> str:
        return self.client.summarize_scene()

    def summarize_assets(self) -> str:
        return self.client.summarize_assets()

    def summarize_conditions(self) -> str:
        return self.client.summarize_conditions()

    def load_condition(self, name: str) -> tuple[bool, str]:
        return self.client.load_condition(name)

    def reset_simulation(self) -> tuple[bool, str]:
        return self.client.reset_simulation()

    def replay_last_sequence(self) -> tuple[bool, str]:
        return self.client.replay_last_sequence()

    def get_scene_snapshot(self) -> dict[str, Any]:
        return self.client.get_scene_snapshot()

    def set_goal(self, goal: str) -> None:
        self.client.set_current_goal(goal)

    def record_action_sequence(self, actions: list[PlannerAction]) -> None:
        self.client.record_last_action_sequence(
            [
                {
                    "type": action.action_type,
                    "params": {
                        "object_id": action.target_object_id,
                        "object_name": action.target_display_name,
                        "display_name": action.target_display_name,
                        "zone": action.zone,
                    },
                }
                for action in actions
            ]
        )

    def home(self) -> tuple[bool, str]:
        response = self.client.apply_action("return_home", {})
        return bool(response.get("success", False)), str(response.get("message", ""))

    def move_above_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        del zone
        response = self.client.apply_action("move_above_object", {"object_name": object_name})
        return bool(response.get("success", False)), str(response.get("message", ""))

    def grasp_object(self, object_name: str) -> tuple[bool, str]:
        response = self.client.apply_action("grasp_object", {"object_name": object_name})
        return bool(response.get("success", False)), str(response.get("message", ""))

    def lift_object(self, object_name: str) -> tuple[bool, str]:
        response = self.client.apply_action("lift_object", {"object_name": object_name})
        return bool(response.get("success", False)), str(response.get("message", ""))

    def move_to_zone(self, zone: str) -> tuple[bool, str]:
        response = self.client.apply_action("move_to_zone", {"zone": zone})
        return bool(response.get("success", False)), str(response.get("message", ""))

    def place_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        response = self.client.apply_action(
            "place_object",
            {"object_name": object_name, "zone": zone},
        )
        return bool(response.get("success", False)), str(response.get("message", ""))

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        started_at = _now()
        step_results: list[StepExecutionResult] = []
        completed_steps = 0
        self.busy = True
        self.abort_requested = False

        for step in plan.steps:
            if self.abort_requested:
                self.busy = False
                return ExecutionResult(
                    plan_id=plan.plan_id,
                    status="aborted",
                    message="CyberWave simulation execution was aborted.",
                    controller_mode=self.mode,
                    controller_name=self.name,
                    execution_enabled=True,
                    completed_steps=completed_steps,
                    total_steps=len(plan.steps),
                    backend_name=self.get_backend_name(),
                    dry_run=False,
                    aborted=True,
                    step_results=step_results,
                    started_at=started_at,
                    completed_at=_now(),
                )

            step.started_at = _now()
            response = self.client.apply_action(step.action, dict(step.params))
            time.sleep(SIMULATION_SPEEDS[self.speed])
            step.completed_at = _now()
            success = bool(response.get("success", False))
            step.status = "completed" if success else "failed"
            step.error = "" if success else str(response.get("message", ""))
            step_results.append(
                StepExecutionResult(
                    step_id=step.step_id,
                    description=step.description,
                    status=step.status,
                    message=str(response.get("message", "")),
                    primitive_name=step.action,
                    dry_run=False,
                    supported=bool(response.get("supported", True)),
                    refusal_reason=str(response.get("refusal_reason", "")),
                    warnings=list(response.get("warnings", [])),
                    raw_backend_message=str(response.get("raw_backend_message", "")),
                    started_at=step.started_at,
                    completed_at=step.completed_at,
                )
            )
            if success:
                completed_steps += 1
                continue

            self.busy = False
            return ExecutionResult(
                plan_id=plan.plan_id,
                status="failed",
                message=str(response.get("message", "CyberWave simulation execution failed.")),
                controller_mode=self.mode,
                controller_name=self.name,
                execution_enabled=True,
                completed_steps=completed_steps,
                total_steps=len(plan.steps),
                backend_name=self.get_backend_name(),
                dry_run=False,
                supported=bool(response.get("supported", True)),
                failed_step=step.action,
                refusal_reason=str(response.get("refusal_reason", "")),
                warnings=list(response.get("warnings", [])),
                raw_backend_message=str(response.get("raw_backend_message", "")),
                step_results=step_results,
                started_at=started_at,
                completed_at=_now(),
            )

        self.busy = False
        return ExecutionResult(
            plan_id=plan.plan_id,
            status="success",
            message="CyberWave simulation plan completed.",
            controller_mode=self.mode,
            controller_name=self.name,
            execution_enabled=True,
            completed_steps=completed_steps,
            total_steps=len(plan.steps),
            backend_name=self.get_backend_name(),
            dry_run=False,
            step_results=step_results,
            started_at=started_at,
            completed_at=_now(),
        )


class SO100HardwareController(RobotController):
    """Bridge-backed SO100 controller that sends primitives to a local service."""

    name = "so100-hardware-controller"
    mode = "hardware"

    def __init__(self) -> None:
        self.bridge_client = get_bridge_client()
        self._last_bridge_status: dict[str, Any] = self._default_bridge_status()

    def _default_bridge_status(self) -> dict[str, Any]:
        config = self.bridge_client.config
        return {
            "success": False,
            "bridge_running": False,
            "connected": False,
            "dry_run": config.dry_run_default,
            "backend_type": config.backend_type,
            "backend_name": f"bridge:{config.backend_type}",
            "busy": False,
            "last_command": "",
            "last_error": "",
            "current_zone": "home",
            "held_object": None,
            "last_target_handled": None,
            "is_home": True,
            "last_connection_attempt": None,
            "real_backend_ready": False,
            "backend_ready_for_real_execution": False,
            "ready": False,
            "warnings": [],
            "reasons": [],
            "capabilities": {},
            "backend_info": {},
            "lerobot_available": False,
            "feetech_support_available": False,
            "can_connect": False,
            "can_home": False,
            "can_move_above_object": False,
            "can_grasp": False,
            "can_lift": False,
            "can_move_to_zone": False,
            "can_place": False,
            "config_source": config.config_source,
            "config_path": config.config_path,
            "message": "Bridge is not running.",
            "ready_message": "Bridge is not running.",
        }

    def _normalize_bridge_status(self, response: dict[str, Any] | None = None) -> dict[str, Any]:
        config = self.bridge_client.config
        if response is None:
            response = self.bridge_client.status()

        if not response.get("success"):
            normalized = self._default_bridge_status()
            if self._last_bridge_status.get("bridge_running", False):
                normalized["last_error"] = str(response.get("error", ""))
            normalized["message"] = str(response.get("message", "Bridge is not running."))
            return normalized

        capabilities = response.get("capabilities", {})
        backend_info = response.get("backend_info", {})
        normalized = {
            "success": True,
            "bridge_running": bool(response.get("bridge_running", True)),
            "connected": bool(response.get("connected", False)),
            "dry_run": bool(response.get("dry_run", config.dry_run_default)),
            "backend_type": str(response.get("backend_type", config.backend_type)),
            "backend_name": f"bridge:{response.get('backend_type', config.backend_type)}",
            "busy": bool(response.get("busy", False)),
            "last_command": str(response.get("last_command", "")),
            "last_error": str(response.get("last_error", "")),
            "current_zone": response.get("current_zone", "home"),
            "held_object": response.get("held_object"),
            "last_target_handled": response.get("last_target_handled"),
            "is_home": bool(response.get("is_home", False)),
            "last_connection_attempt": response.get("last_connection_attempt"),
            "real_backend_ready": bool(response.get("real_backend_ready", False)),
            "backend_ready_for_real_execution": bool(
                response.get(
                    "backend_ready_for_real_execution",
                    capabilities.get("backend_ready_for_real_execution", False),
                )
            ),
            "ready": bool(response.get("ready", False)),
            "warnings": list(response.get("warnings", capabilities.get("warnings", []))),
            "reasons": list(response.get("reasons", capabilities.get("not_ready_reasons", []))),
            "capabilities": capabilities if isinstance(capabilities, dict) else {},
            "backend_info": backend_info if isinstance(backend_info, dict) else {},
            "lerobot_available": bool(response.get("lerobot_available", False)),
            "feetech_support_available": bool(response.get("feetech_support_available", False)),
            "can_connect": bool(capabilities.get("can_connect", False)),
            "can_home": bool(capabilities.get("can_home", False)),
            "can_move_above_object": bool(capabilities.get("can_move_above_object", False)),
            "can_grasp": bool(capabilities.get("can_grasp", False)),
            "can_lift": bool(capabilities.get("can_lift", False)),
            "can_move_to_zone": bool(capabilities.get("can_move_to_zone", False)),
            "can_place": bool(capabilities.get("can_place", False)),
            "config_source": config.config_source,
            "config_path": config.config_path,
            "message": str(response.get("message", "Bridge status ready.")),
            "ready_message": str(response.get("message", "Bridge status ready.")),
        }
        return normalized

    def bridge_status(self) -> dict[str, Any]:
        self._last_bridge_status = self._normalize_bridge_status()
        return dict(self._last_bridge_status)

    def bridge_capabilities(self) -> dict[str, Any]:
        response = self.bridge_client.capabilities()
        if response.get("success"):
            self._last_bridge_status = self._normalize_bridge_status(response)
        else:
            self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return dict(self._last_bridge_status)

    def bridge_backend_info(self) -> dict[str, Any]:
        response = self.bridge_client.backend_info()
        if response.get("success"):
            self._last_bridge_status = self._normalize_bridge_status(response)
        else:
            self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return dict(self._last_bridge_status)

    def hardware_ready(self) -> dict[str, Any]:
        response = self.bridge_client.hardware_ready()
        if response.get("success"):
            self._last_bridge_status = self._normalize_bridge_status(response)
        else:
            self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return dict(self._last_bridge_status)

    def bridge_start(self) -> tuple[bool, str]:
        response = self.bridge_client.start_bridge()
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def bridge_stop(self) -> tuple[bool, str]:
        response = self.bridge_client.stop_bridge()
        self._last_bridge_status = self._default_bridge_status()
        self._last_bridge_status["message"] = "Bridge is stopped."
        message = str(response.get("message", ""))
        if not response.get("success") and not self._last_bridge_status["bridge_running"]:
            return True, "Bridge is stopped."
        return bool(response.get("success", False)), message

    def bridge_ping(self) -> tuple[bool, str]:
        response = self.bridge_client.ping()
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def set_bridge_dry_run(self, enabled: bool) -> tuple[bool, str]:
        response = self.bridge_client.set_dry_run(enabled)
        self._last_bridge_status = self._normalize_bridge_status(
            self.bridge_client.status()
        )
        return bool(response.get("success", False)), str(response.get("message", ""))

    def is_available(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return bool(self.bridge_status()["connected"])

    def is_dry_run(self) -> bool:
        return bool(self.bridge_status()["dry_run"])

    def get_backend_name(self) -> str:
        return str(self.bridge_status()["backend_name"])

    def get_readiness(self) -> dict[str, Any]:
        bridge_status = self.bridge_status()
        capabilities = bridge_status.get("capabilities", {})
        warnings: list[str] = list(bridge_status.get("warnings", []))
        errors: list[str] = []

        if not bridge_status["bridge_running"]:
            errors.append("Bridge is not running. Start it with 'bridge start'.")
        if bridge_status["dry_run"]:
            warnings.append("Bridge is in dry-run mode. No real motor commands will be sent.")
        if bridge_status["bridge_running"] and not bridge_status["connected"]:
            errors.append("Bridge is not connected. Run 'hardware connect' first.")
        if (
            bridge_status["bridge_running"]
            and bridge_status["connected"]
            and not bridge_status["dry_run"]
            and not bridge_status["backend_ready_for_real_execution"]
        ):
            errors.append("Bridge backend is not ready for real SO100 execution.")
        for reason in capabilities.get("not_ready_reasons", []):
            if reason not in errors and not bridge_status["dry_run"]:
                errors.append(str(reason))
        if bridge_status["last_error"] and bridge_status["last_error"] not in warnings:
            warnings.append(bridge_status["last_error"])

        return {
            "ready": bridge_status["bridge_running"] and bridge_status["connected"],
            "real_backend_ready": bridge_status["backend_ready_for_real_execution"],
            "warnings": warnings,
            "errors": errors,
            "backend_name": bridge_status["backend_name"],
            "dry_run": bridge_status["dry_run"],
        }

    def connect(self) -> tuple[bool, str]:
        response = self.bridge_client.connect()
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def disconnect(self) -> tuple[bool, str]:
        response = self.bridge_client.disconnect()
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def abort(self) -> tuple[bool, str]:
        response = self.bridge_client.abort()
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def _make_primitive(self, primitive_name: str, **params: Any) -> PrimitiveCommand:
        object_name = str(params.get("object_name", "object"))
        zone = str(params.get("zone", "unknown"))

        if primitive_name == "move_above_object":
            summary = f"Move above {object_name} in {zone} zone."
        elif primitive_name == "grasp_object":
            summary = f"Grasp {object_name}."
        elif primitive_name == "lift_object":
            summary = f"Lift {object_name}."
        elif primitive_name == "move_to_zone":
            summary = f"Move to {zone} zone."
        elif primitive_name == "place_object":
            summary = f"Place {object_name} in {zone} zone."
        else:
            summary = "Return robot arm to home position."

        return PrimitiveCommand(
            primitive_id=_new_id("command"),
            primitive_name=primitive_name,
            params=params,
            summary=summary,
        )

    def _step_to_primitive(self, step: ExecutionStep) -> PrimitiveCommand | None:
        if step.action not in SUPPORTED_PRIMITIVES:
            return None
        if step.action == "move_above_object":
            return self._make_primitive(
                "move_above_object",
                object_name=str(step.params["object_name"]),
                zone=str(step.params["zone"]),
            )
        if step.action == "grasp_object":
            return self._make_primitive(
                "grasp_object",
                object_name=str(step.params["object_name"]),
            )
        if step.action == "lift_object":
            return self._make_primitive(
                "lift_object",
                object_name=str(step.params["object_name"]),
            )
        if step.action == "move_to_zone":
            return self._make_primitive("move_to_zone", zone=str(step.params["zone"]))
        if step.action == "place_object":
            return self._make_primitive(
                "place_object",
                object_name=str(step.params["object_name"]),
                zone=str(step.params["zone"]),
            )
        if step.action == "return_home":
            return self._make_primitive("return_home")
        if step.action == "home":
            return self._make_primitive("home")
        return None

    def _send_primitive(self, primitive: PrimitiveCommand) -> tuple[bool, str]:
        response = self.bridge_client.send_primitive(primitive.to_payload())
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        return bool(response.get("success", False)), str(response.get("message", ""))

    def move_above_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        return self._send_primitive(
            self._make_primitive("move_above_object", object_name=object_name, zone=zone)
        )

    def grasp_object(self, object_name: str) -> tuple[bool, str]:
        return self._send_primitive(self._make_primitive("grasp_object", object_name=object_name))

    def lift_object(self, object_name: str) -> tuple[bool, str]:
        return self._send_primitive(self._make_primitive("lift_object", object_name=object_name))

    def move_to_zone(self, zone: str) -> tuple[bool, str]:
        return self._send_primitive(self._make_primitive("move_to_zone", zone=zone))

    def place_object(self, object_name: str, zone: str) -> tuple[bool, str]:
        return self._send_primitive(
            self._make_primitive("place_object", object_name=object_name, zone=zone)
        )

    def home(self) -> tuple[bool, str]:
        return self._send_primitive(self._make_primitive("return_home"))

    def get_state(self) -> dict[str, Any]:
        bridge_status = self.bridge_status()
        return {
            "controller_name": self.name,
            "controller_mode": self.mode,
            "connected": bridge_status["connected"],
            "busy": bridge_status["busy"],
            "is_home": bridge_status["is_home"],
            "held_object": bridge_status["held_object"],
            "current_zone": bridge_status["current_zone"],
            "last_target_handled": bridge_status["last_target_handled"],
            "speed": "n/a",
            "backend_name": bridge_status["backend_name"],
            "dry_run": bridge_status["dry_run"],
            "hardware_backend": bridge_status["backend_name"],
            "hardware_backend_type": bridge_status["backend_type"],
            "hardware_dry_run": bridge_status["dry_run"],
            "hardware_ready": bridge_status["bridge_running"] and bridge_status["connected"],
            "hardware_real_ready": bridge_status["real_backend_ready"],
            "last_hardware_error": bridge_status["last_error"],
            "bridge_running": bridge_status["bridge_running"],
            "bridge_connected": bridge_status["connected"],
            "bridge_dry_run": bridge_status["dry_run"],
            "bridge_last_command": bridge_status["last_command"],
            "bridge_last_error": bridge_status["last_error"],
            "bridge_config_source": bridge_status["config_source"],
            "bridge_config_path": bridge_status["config_path"],
            "last_connection_attempt": bridge_status["last_connection_attempt"],
            "lerobot_available": bridge_status["lerobot_available"],
            "feetech_support_available": bridge_status["feetech_support_available"],
            "backend_ready_for_real_execution": bridge_status["backend_ready_for_real_execution"],
            "can_connect": bridge_status["can_connect"],
            "can_home": bridge_status["can_home"],
            "can_move_above_object": bridge_status["can_move_above_object"],
            "can_grasp": bridge_status["can_grasp"],
            "can_lift": bridge_status["can_lift"],
            "can_move_to_zone": bridge_status["can_move_to_zone"],
            "can_place": bridge_status["can_place"],
            "bridge_warnings": list(bridge_status["warnings"]),
            "bridge_capabilities": bridge_status["capabilities"],
            "bridge_backend_info": bridge_status["backend_info"],
            "bridge_reasons": list(bridge_status["reasons"]),
            "abort_requested": False,
        }

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        started_at = _now()
        bridge_status = self.bridge_status()
        plan_steps: list[dict[str, Any]] = []
        for step in plan.steps:
            primitive = self._step_to_primitive(step)
            if primitive is None:
                return ExecutionResult(
                    plan_id=plan.plan_id,
                    status="failed",
                    message=f"Unsupported hardware step: {step.action}",
                    controller_mode=self.mode,
                    controller_name=self.name,
                    execution_enabled=True,
                    completed_steps=0,
                    total_steps=len(plan.steps),
                    backend_name=bridge_status["backend_name"],
                    dry_run=bridge_status["dry_run"],
                    supported=False,
                    failed_step=step.action,
                    refusal_reason=f"Unsupported hardware step: {step.action}",
                    warnings=list(bridge_status.get("warnings", [])),
                    started_at=started_at,
                    completed_at=_now(),
                )
            plan_steps.append(primitive.to_payload())

        response = self.bridge_client.send_plan({"plan_id": plan.plan_id, "steps": plan_steps})
        self._last_bridge_status = self._normalize_bridge_status(self.bridge_client.status())
        latest_status = self.bridge_status()
        step_results: list[StepExecutionResult] = []

        for index, step_response in enumerate(response.get("step_results", []), start=1):
            original_step = plan.steps[index - 1]
            step_status = "completed" if step_response.get("success") else "failed"
            if not step_response.get("success") and "abort" in str(step_response.get("message", "")).lower():
                step_status = "aborted"
            step_results.append(
                StepExecutionResult(
                    step_id=original_step.step_id,
                    description=original_step.description,
                    status=step_status,
                    message=str(step_response.get("message", "")),
                    primitive_name=str(step_response.get("primitive", "")),
                    dry_run=bool(response.get("dry_run", latest_status["dry_run"])),
                    supported=bool(step_response.get("supported", True)),
                    refusal_reason=str(step_response.get("refusal_reason", "")),
                    warnings=list(step_response.get("warnings", [])),
                    raw_backend_message=str(step_response.get("raw_backend_message", "")),
                )
            )

        status = "success"
        message = str(response.get("message", ""))
        aborted = False
        if not response.get("success", False):
            lowered = message.lower()
            if "abort" in lowered:
                status = "aborted"
                aborted = True
            else:
                status = "failed"

        completed_steps = sum(1 for item in step_results if item.status == "completed")
        return ExecutionResult(
            plan_id=plan.plan_id,
            status=status,
            message=message,
            controller_mode=self.mode,
            controller_name=self.name,
            execution_enabled=True,
            completed_steps=completed_steps,
            total_steps=len(plan.steps),
            backend_name=latest_status["backend_name"],
            dry_run=bool(response.get("dry_run", latest_status["dry_run"])),
            aborted=aborted,
            supported=bool(response.get("supported", True)),
            failed_step=str(response.get("failed_step", "")),
            refusal_reason=str(response.get("refusal_reason", "")),
            warnings=list(response.get("warnings", [])),
            raw_backend_message=str(response.get("raw_backend_message", "")),
            step_results=step_results,
            started_at=started_at,
            completed_at=_now(),
        )


class ExecutionManager:
    """Builds, validates, and optionally executes robot plans."""

    def __init__(self) -> None:
        self._controllers: dict[str, RobotController] = {
            "simulation": SimulationRobotController(),
            "cyberwave": CyberWaveSimulationController(),
            "hardware": SO100HardwareController(),
        }
        self._session = RobotSessionState()
        self._last_plan: ExecutionPlan | None = None
        self._last_validation: ValidationReport | None = None
        self._last_result: ExecutionResult | None = None
        self._last_scene_snapshot: dict[str, Any] | None = None
        self._last_cyberwave_prompt: str | None = None
        self._last_planner_response: PlannerResponse | None = None
        self._apply_controller_state(self._active_controller())

    def _active_controller(self) -> RobotController:
        return self._controllers[self._session.mode]

    def _hardware_controller(self) -> SO100HardwareController:
        return self._controllers["hardware"]  # type: ignore[return-value]

    def _cyberwave_controller(self) -> CyberWaveSimulationController:
        return self._controllers["cyberwave"]  # type: ignore[return-value]

    def _apply_controller_state(self, controller: RobotController) -> None:
        state = controller.get_state()
        self._session.controller_name = str(state.get("controller_name", controller.name))
        self._session.connected = bool(state.get("connected", False))
        self._session.busy = bool(state.get("busy", False))
        self._session.is_home = bool(state.get("is_home", False))
        self._session.held_object = state.get("held_object")
        self._session.current_zone = state.get("current_zone")
        self._session.last_target_handled = state.get("last_target_handled")
        self._session.abort_requested = bool(state.get("abort_requested", False))

        hardware_state = self._hardware_controller().get_state()
        self._session.hardware_backend = str(hardware_state.get("hardware_backend", "bridge:stub"))
        self._session.hardware_backend_type = str(hardware_state.get("hardware_backend_type", "stub"))
        self._session.hardware_dry_run = bool(hardware_state.get("hardware_dry_run", True))
        self._session.hardware_ready = bool(hardware_state.get("hardware_ready", False))
        self._session.hardware_real_ready = bool(hardware_state.get("hardware_real_ready", False))
        self._session.last_hardware_error = str(hardware_state.get("last_hardware_error", ""))
        self._session.bridge_running = bool(hardware_state.get("bridge_running", False))
        self._session.bridge_connected = bool(hardware_state.get("bridge_connected", False))
        self._session.bridge_dry_run = bool(hardware_state.get("bridge_dry_run", True))
        self._session.bridge_last_command = str(hardware_state.get("bridge_last_command", ""))
        self._session.bridge_last_error = str(hardware_state.get("bridge_last_error", ""))
        self._session.bridge_config_source = str(hardware_state.get("bridge_config_source", "defaults"))
        self._session.bridge_config_path = str(hardware_state.get("bridge_config_path", ""))
        self._session.last_connection_attempt = hardware_state.get("last_connection_attempt")
        self._session.lerobot_available = bool(hardware_state.get("lerobot_available", False))
        self._session.feetech_support_available = bool(
            hardware_state.get("feetech_support_available", False)
        )
        self._session.backend_ready_for_real_execution = bool(
            hardware_state.get("backend_ready_for_real_execution", False)
        )
        self._session.can_connect = bool(hardware_state.get("can_connect", False))
        self._session.can_home = bool(hardware_state.get("can_home", False))
        self._session.can_move_above_object = bool(
            hardware_state.get("can_move_above_object", False)
        )
        self._session.can_grasp = bool(hardware_state.get("can_grasp", False))
        self._session.can_lift = bool(hardware_state.get("can_lift", False))
        self._session.can_move_to_zone = bool(hardware_state.get("can_move_to_zone", False))
        self._session.can_place = bool(hardware_state.get("can_place", False))

        cyberwave_state = self._cyberwave_controller().get_state()
        self._session.cyberwave_connected = bool(cyberwave_state.get("cyberwave_connected", False))
        self._session.cyberwave_backend = str(
            cyberwave_state.get("cyberwave_backend", "cyberwave:local-scene")
        )
        self._session.cyberwave_local_ready = bool(
            cyberwave_state.get("cyberwave_local_ready", True)
        )
        self._session.cyberwave_remote_ready = bool(
            cyberwave_state.get("cyberwave_remote_ready", False)
        )
        self._session.cyberwave_active_condition = str(
            cyberwave_state.get("cyberwave_active_condition", "demo_showcase_scene")
        )
        self._session.cyberwave_last_error = str(
            cyberwave_state.get("cyberwave_last_error", "")
        )
        self._session.cyberwave_openai_enabled = bool(
            cyberwave_state.get("cyberwave_openai_enabled", True)
        )
        self._session.cyberwave_openai_configured = bool(
            cyberwave_state.get("cyberwave_openai_configured", False)
        )
        self._session.cyberwave_openai_available = bool(
            cyberwave_state.get("cyberwave_openai_available", False)
        )
        self._session.cyberwave_openai_model = str(
            cyberwave_state.get("cyberwave_openai_model", "gpt-5-mini")
        )

    def _step(self, action: str, description: str, **params: Any) -> ExecutionStep:
        return ExecutionStep(
            step_id=_new_id("step"),
            action=action,
            description=description,
            params=params,
        )

    def _build_steps(
        self,
        action: str,
        target_object: dict[str, Any] | None,
        destination_zone: str | None,
    ) -> list[ExecutionStep]:
        if action == "home" or target_object is None:
            return [self._step("return_home", "Return robot arm to home position.")]

        object_id = str(target_object.get("object_id", target_object.get("id", "")))
        object_name = str(target_object.get("display_name", target_object.get("name", "")))
        source_zone = str(target_object["zone"])
        steps = [
            self._step(
                "move_above_object",
                f"Move above {object_name} in {source_zone} zone.",
                object_id=object_id,
                object_name=object_name,
                zone=source_zone,
            ),
            self._step(
                "grasp_object",
                f"Grasp {object_name}.",
                object_id=object_id,
                object_name=object_name,
            ),
            self._step(
                "lift_object",
                f"Lift {object_name}.",
                object_id=object_id,
                object_name=object_name,
            ),
        ]
        if action == "move" and destination_zone:
            steps.append(
                self._step(
                    "move_to_zone",
                    f"Move to {destination_zone} zone.",
                    zone=destination_zone,
                )
            )
            steps.append(
                self._step(
                    "place_object",
                    f"Place {object_name} in {destination_zone} zone.",
                    object_id=object_id,
                    object_name=object_name,
                    zone=destination_zone,
                )
            )
        steps.append(self._step("return_home", "Return robot arm to home position."))
        return steps

    def _build_steps_from_planner_actions(self, actions: list[PlannerAction]) -> list[ExecutionStep]:
        descriptions = {
            "move_above_object": lambda action: (
                f"Move above {action.target_display_name or action.target_object_id} in the current scene."
            ),
            "grasp_object": lambda action: (
                f"Grasp {action.target_display_name or action.target_object_id}."
            ),
            "lift_object": lambda action: (
                f"Lift {action.target_display_name or action.target_object_id}."
            ),
            "move_to_zone": lambda action: f"Move to {action.zone} zone.",
            "place_object": lambda action: (
                f"Place {action.target_display_name or action.target_object_id} in {action.zone} zone."
            ),
            "return_home": lambda action: "Return robot arm to home position.",
        }
        steps: list[ExecutionStep] = []
        for action in actions:
            params: dict[str, Any] = {}
            if action.target_object_id is not None:
                params["object_id"] = action.target_object_id
            if action.target_display_name is not None:
                params["object_name"] = action.target_display_name
                params["display_name"] = action.target_display_name
            if action.zone is not None:
                params["zone"] = action.zone
            description_builder = descriptions.get(action.action_type, lambda _: action.action_type)
            steps.append(
                self._step(
                    action.action_type,
                    description_builder(action),
                    **params,
                )
            )
        return steps

    def _clone_scene_snapshot(self, scene_snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
        if scene_snapshot is None:
            return None
        return copy.deepcopy(scene_snapshot)

    def build_plan(
        self,
        *,
        action: str,
        target_object: dict[str, Any] | None,
        destination_zone: str | None,
        scene_snapshot: dict[str, Any] | None,
        confirmation_required: bool = False,
        custom_steps: list[ExecutionStep] | None = None,
    ) -> ExecutionPlan:
        target_copy = None if target_object is None else dict(target_object)
        source_zone = None if target_object is None else str(target_object.get("zone", ""))
        target_object_id = (
            None
            if target_object is None
            else str(target_object.get("object_id", target_object.get("id", ""))) or None
        )
        target_name = (
            None
            if target_object is None
            else str(target_object.get("display_name", target_object.get("name", ""))) or None
        )
        target_label = None if target_object is None else str(target_object.get("label", ""))
        scene_source = "unknown"
        fallback_used = False
        scene_stale = False
        scene_age_seconds: float | None = None
        if scene_snapshot is not None:
            scene_source = str(scene_snapshot.get("source", "unknown"))
            fallback_used = bool(scene_snapshot.get("fallback_used", False))
            metadata = scene_snapshot.get("metadata", {})
            scene_stale = bool(metadata.get("scene_stale", False))
            scene_age_seconds = metadata.get("last_update_age_seconds")

        plan = ExecutionPlan(
            plan_id=_new_id("plan"),
            action_type=action,
            target_object=target_copy,
            target_object_id=target_object_id,
            target_name=target_name,
            target_label=target_label,
            source_zone=source_zone or None,
            destination_zone=destination_zone,
            steps=custom_steps
            if custom_steps is not None
            else self._build_steps(action, target_object, destination_zone),
            confirmation_required=confirmation_required,
            execution_mode=self._session.mode,
            scene_source=scene_source,
            fallback_used=fallback_used,
            scene_stale=scene_stale,
            scene_age_seconds=scene_age_seconds,
            created_at=_now(),
        )
        self._last_plan = plan
        self._last_scene_snapshot = self._clone_scene_snapshot(scene_snapshot)
        self._session.last_plan_id = plan.plan_id
        return plan

    def validate_plan(
        self,
        plan: ExecutionPlan,
        scene_snapshot: dict[str, Any] | None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        controller = self._active_controller()
        self._apply_controller_state(controller)

        if plan.action_type not in {"move", "pick", "home", "cyberwave", "simulation"}:
            issues.append(
                ValidationIssue("error", "unsupported_action", f"Unsupported action: {plan.action_type}.")
            )

        if plan.action_type not in {"home", "cyberwave", "simulation"} and plan.target_object is None:
            issues.append(ValidationIssue("error", "no_target", "No target object was resolved."))

        if (
            plan.action_type == "cyberwave"
            and any(step.action in OBJECT_TARGET_STEP_ACTIONS for step in plan.steps)
            and (plan.target_object is None or not plan.target_object_id)
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "no_target",
                    "No canonical simulation target object was resolved.",
                )
            )

        if plan.action_type == "move" and not plan.destination_zone:
            issues.append(ValidationIssue("error", "no_destination", "No destination zone was resolved."))

        if plan.destination_zone and plan.destination_zone not in ALLOWED_ZONES:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_destination",
                    f"Destination '{plan.destination_zone}' is not supported.",
                )
            )

        if plan.target_label and plan.target_label not in SUPPORTED_LABELS:
            issues.append(
                ValidationIssue(
                    "warning",
                    "unsupported_label",
                    f"Object label '{plan.target_label}' is not fully supported yet.",
                )
            )

        if plan.confirmation_required and plan.source_zone == plan.destination_zone:
            issues.append(
                ValidationIssue(
                    "warning",
                    "same_zone_confirmed",
                    f"{plan.target_name} is already in {plan.destination_zone}; user confirmed the move anyway.",
                )
            )

        if scene_snapshot is None:
            if plan.action_type != "home":
                issues.append(
                    ValidationIssue(
                        "error",
                        "no_scene",
                        "No scene snapshot is available for this command.",
                    )
                )
        else:
            metadata = scene_snapshot.get("metadata", {})
            if bool(metadata.get("scene_stale")):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "stale_scene",
                        "The current scene cache is stale.",
                    )
                )
            if bool(scene_snapshot.get("fallback_used")):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "fallback_scene",
                        "The current plan is based on the fallback mock scene.",
                    )
                )

        if self._session.mode == "simulation":
            issues.append(
                ValidationIssue(
                    "warning",
                    "simulation_mode",
                    "Execution is running in simulation mode.",
                )
            )

        if self._session.mode == "cyberwave":
            readiness = self._cyberwave_controller().get_readiness()
            if not readiness["ready"]:
                for error in readiness["errors"]:
                    issues.append(ValidationIssue("error", "cyberwave_error", error))
            for warning in readiness["warnings"]:
                issues.append(ValidationIssue("warning", "cyberwave_warning", warning))

        if self._session.mode == "hardware":
            hardware = self._hardware_controller()
            readiness = hardware.get_readiness()

            if not self._session.bridge_running:
                issues.append(
                    ValidationIssue(
                        "error",
                        "bridge_not_running",
                        "Bridge is not running. Start it with 'bridge start'.",
                    )
                )
            if readiness["dry_run"]:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "bridge_dry_run",
                        "Hardware mode is using the bridge in dry-run mode.",
                    )
                )
            if self._session.bridge_running and not self._session.bridge_connected:
                issues.append(
                    ValidationIssue(
                        "error",
                        "bridge_not_connected",
                        "Bridge is running but not connected. Run 'hardware connect' first.",
                    )
                )
            if (
                self._session.bridge_running
                and self._session.bridge_connected
                and not readiness["dry_run"]
                and not readiness["real_backend_ready"]
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "bridge_real_not_ready",
                        "Bridge real backend is not ready for SO100 execution.",
                    )
                )
            if plan.fallback_used and not readiness["dry_run"]:
                issues.append(
                    ValidationIssue(
                        "error",
                        "fallback_hardware_block",
                        "Real hardware execution is blocked while the scene is fallback-based.",
                    )
                )
            for warning in readiness["warnings"]:
                issues.append(ValidationIssue("warning", "bridge_warning", warning))
            for error in readiness["errors"]:
                if error not in {issue.message for issue in issues if issue.level == "error"}:
                    issues.append(ValidationIssue("error", "bridge_error", error))

        report = ValidationReport(
            can_execute=not any(issue.level == "error" for issue in issues),
            issues=issues,
        )
        self._last_validation = report
        return report

    def _result_for_disabled_execution(self, plan: ExecutionPlan) -> ExecutionResult:
        return ExecutionResult(
            plan_id=plan.plan_id,
            status="not_executed",
            message="Execution is disabled. Plan generated only.",
            controller_mode=self._session.mode,
            controller_name=self._active_controller().name,
            execution_enabled=False,
            completed_steps=0,
            total_steps=len(plan.steps),
            backend_name=self._active_controller().get_backend_name(),
            dry_run=self._active_controller().is_dry_run(),
            step_results=[],
            started_at=None,
            completed_at=None,
        )

    def _result_for_blocked_execution(
        self,
        plan: ExecutionPlan,
        validation: ValidationReport,
        *,
        forced: bool = False,
    ) -> ExecutionResult:
        error_messages = [issue.message for issue in validation.errors]
        message = "Execution blocked by validation."
        if error_messages:
            message = error_messages[0]
        return ExecutionResult(
            plan_id=plan.plan_id,
            status="blocked",
            message=message,
            controller_mode=self._session.mode,
            controller_name=self._active_controller().name,
            execution_enabled=self._session.execution_enabled or forced,
            completed_steps=0,
            total_steps=len(plan.steps),
            backend_name=self._active_controller().get_backend_name(),
            dry_run=self._active_controller().is_dry_run(),
            step_results=[],
            started_at=_now(),
            completed_at=_now(),
        )

    def _store_result(self, result: ExecutionResult) -> ExecutionResult:
        self._last_result = result
        self._session.last_result_status = result.status
        self._apply_controller_state(self._active_controller())
        return result

    def execute_plan(
        self,
        plan: ExecutionPlan,
        validation: ValidationReport,
        *,
        force: bool = False,
    ) -> ExecutionResult:
        should_execute = self._session.execution_enabled or force
        if not should_execute:
            return self._store_result(self._result_for_disabled_execution(plan))

        if not validation.can_execute:
            return self._store_result(self._result_for_blocked_execution(plan, validation, forced=force))

        controller = self._active_controller()
        controller.set_speed(self._session.speed)

        if self._session.mode == "simulation":
            controller.connect()

        result = controller.execute_plan(plan)
        return self._store_result(result)

    def execute_last_plan(self) -> tuple[ExecutionPlan | None, ValidationReport | None, ExecutionResult]:
        if self._last_plan is None:
            result = ExecutionResult(
                plan_id="none",
                status="blocked",
                message="No execution plan is available. Build a plan first.",
                controller_mode=self._session.mode,
                controller_name=self._active_controller().name,
                execution_enabled=True,
                completed_steps=0,
                total_steps=0,
                backend_name=self._active_controller().get_backend_name(),
                dry_run=self._active_controller().is_dry_run(),
                started_at=_now(),
                completed_at=_now(),
            )
            return None, None, self._store_result(result)

        scene_snapshot = self._clone_scene_snapshot(self._last_scene_snapshot)
        validation = self.validate_plan(self._last_plan, scene_snapshot)
        result = self.execute_plan(self._last_plan, validation, force=True)
        return self._last_plan, validation, result

    def prepare_last_plan_execution(self) -> tuple[ExecutionPlan | None, ValidationReport | None]:
        if self._last_plan is None:
            return None, None
        scene_snapshot = self._clone_scene_snapshot(self._last_scene_snapshot)
        validation = self.validate_plan(self._last_plan, scene_snapshot)
        return self._last_plan, validation

    def get_mode(self) -> str:
        return self._session.mode

    def set_mode(self, mode: str) -> tuple[bool, str]:
        if mode not in self._controllers:
            return False, f"Unsupported robot mode: {mode}."

        self._session.mode = mode
        controller = self._active_controller()
        self._session.controller_name = controller.name
        self._apply_controller_state(controller)

        if mode == "cyberwave":
            status = self._cyberwave_controller().get_state()
            backend = status.get("cyberwave_backend", "cyberwave:local-scene")
            if status.get("cyberwave_connected"):
                return True, f"Robot mode set to cyberwave ({backend})."
            return True, f"Robot mode set to cyberwave ({backend}, local-only until connected)."
        if mode == "hardware":
            dry_text = "bridge dry-run" if self._session.bridge_dry_run else "bridge real-mode pending"
            return True, f"Robot mode set to hardware ({dry_text})."
        return True, "Robot mode set to simulation."

    def set_execution_enabled(self, enabled: bool) -> tuple[bool, str]:
        self._session.execution_enabled = enabled
        state_text = "on" if enabled else "off"
        return True, f"Execution is {state_text}."

    def is_execution_enabled(self) -> bool:
        return self._session.execution_enabled

    def set_speed(self, speed: str) -> tuple[bool, str]:
        if speed not in SIMULATION_SPEEDS:
            return False, f"Unsupported speed: {speed}."
        self._session.speed = speed
        self._controllers["simulation"].set_speed(speed)
        self._controllers["cyberwave"].set_speed(speed)
        return True, f"Simulation speed set to {speed}."

    def get_speed(self) -> str:
        return self._session.speed

    def get_last_plan(self) -> ExecutionPlan | None:
        return self._last_plan

    def get_last_result(self) -> ExecutionResult | None:
        return self._last_result

    def get_last_validation(self) -> ValidationReport | None:
        return self._last_validation

    def bridge_start(self) -> tuple[bool, str]:
        success, message = self._hardware_controller().bridge_start()
        self._apply_controller_state(self._active_controller())
        return success, message

    def bridge_stop(self) -> tuple[bool, str]:
        success, message = self._hardware_controller().bridge_stop()
        self._apply_controller_state(self._active_controller())
        return success, message

    def bridge_ping(self) -> tuple[bool, str]:
        success, message = self._hardware_controller().bridge_ping()
        self._apply_controller_state(self._active_controller())
        return success, message

    def set_bridge_dry_run(self, enabled: bool) -> tuple[bool, str]:
        success, message = self._hardware_controller().set_bridge_dry_run(enabled)
        self._apply_controller_state(self._active_controller())
        return success, message

    def hardware_connect(self) -> tuple[bool, str]:
        success, message = self._hardware_controller().connect()
        self._apply_controller_state(self._active_controller())
        return success, message

    def hardware_disconnect(self) -> tuple[bool, str]:
        success, message = self._hardware_controller().disconnect()
        self._apply_controller_state(self._active_controller())
        return success, message

    def set_hardware_dry_run(self, enabled: bool) -> tuple[bool, str]:
        return self.set_bridge_dry_run(enabled)

    def bridge_status(self) -> dict[str, Any]:
        return self._hardware_controller().bridge_status()

    def bridge_capabilities(self) -> dict[str, Any]:
        status = self._hardware_controller().bridge_capabilities()
        self._apply_controller_state(self._active_controller())
        return status

    def bridge_backend_info(self) -> dict[str, Any]:
        status = self._hardware_controller().bridge_backend_info()
        self._apply_controller_state(self._active_controller())
        return status

    def hardware_ready(self) -> dict[str, Any]:
        status = self._hardware_controller().hardware_ready()
        self._apply_controller_state(self._active_controller())
        return status

    def hardware_status(self) -> dict[str, Any]:
        return self._hardware_controller().get_state()

    def cyberwave_status(self) -> dict[str, Any]:
        return self._cyberwave_controller().get_state()

    def list_sim_assets(self) -> str:
        return self._cyberwave_controller().summarize_assets()

    def list_sim_conditions(self) -> str:
        return self._cyberwave_controller().summarize_conditions()

    def cyberwave_connect(self) -> tuple[bool, str]:
        success, message = self._cyberwave_controller().connect()
        self._apply_controller_state(self._active_controller())
        return success, message

    def cyberwave_disconnect(self) -> tuple[bool, str]:
        success, message = self._cyberwave_controller().disconnect()
        self._apply_controller_state(self._active_controller())
        return success, message

    def load_sim_condition(self, name: str) -> tuple[bool, str]:
        success, message = self._cyberwave_controller().load_condition(name)
        self._apply_controller_state(self._active_controller())
        return success, message

    def reset_sim(self) -> tuple[bool, str]:
        success, message = self._cyberwave_controller().reset_simulation()
        self._apply_controller_state(self._active_controller())
        return success, message

    def replay_sim(self) -> tuple[bool, str]:
        success, message = self._cyberwave_controller().replay_last_sequence()
        self._apply_controller_state(self._active_controller())
        return success, message

    def get_sim_scene_snapshot(self) -> dict[str, Any]:
        return self._cyberwave_controller().get_scene_snapshot()

    def summarize_sim_scene(self) -> str:
        return self._cyberwave_controller().summarize_scene()

    def abort_execution(self) -> tuple[bool, str]:
        success, message = self._active_controller().abort()
        self._apply_controller_state(self._active_controller())
        if success:
            result = ExecutionResult(
                plan_id=self._session.last_plan_id or "none",
                status="aborted",
                message=message,
                controller_mode=self._session.mode,
                controller_name=self._active_controller().name,
                execution_enabled=self._session.execution_enabled,
                completed_steps=0,
                total_steps=len(self._last_plan.steps) if self._last_plan is not None else 0,
                backend_name=self._active_controller().get_backend_name(),
                dry_run=self._active_controller().is_dry_run(),
                aborted=True,
                started_at=_now(),
                completed_at=_now(),
            )
            self._store_result(result)
        return success, message

    def get_status(self) -> dict[str, Any]:
        controller = self._active_controller()
        self._apply_controller_state(controller)
        status = self._session.to_dict()
        status["hardware_available"] = self._controllers["hardware"].is_available()
        status["last_result"] = None if self._last_result is None else self._last_result.to_dict()
        status["last_plan"] = None if self._last_plan is None else self._last_plan.to_dict()
        status["bridge_status"] = self.bridge_status()
        status["hardware_status"] = self.hardware_status()
        status["cyberwave_status"] = self.cyberwave_status()
        status["last_planner_response"] = (
            None if self._last_planner_response is None else self._last_planner_response.to_dict()
        )
        status["last_cyberwave_prompt"] = self._last_cyberwave_prompt
        return status

    def plan_cyberwave_command(
        self,
        command_text: str,
        *,
        clarification_context: str | None = None,
    ) -> tuple[PlannerResponse, ExecutionPlan | None, ValidationReport | None]:
        controller = self._cyberwave_controller()
        scene_snapshot = controller.get_scene_snapshot()
        scene_context = controller.client.get_scene_context()
        planner_response = controller.planner.plan_command(
            command_text,
            scene_context,
            clarification_context=clarification_context,
        )
        self._last_cyberwave_prompt = command_text
        self._last_planner_response = planner_response

        if not planner_response.success or planner_response.needs_clarification:
            return planner_response, None, None

        target_object = None
        if planner_response.target_object_id:
            for object_ in scene_snapshot["objects"]:
                if (
                    str(object_.get("object_id", object_.get("id", "")))
                    == planner_response.target_object_id
                ):
                    target_object = dict(object_)
                    break
        destination_zone = planner_response.destination_zone
        if destination_zone is None:
            for action in reversed(planner_response.actions):
                if action.zone in VALID_SIM_ZONES:
                    destination_zone = action.zone
                    break

        plan = self.build_plan(
            action="cyberwave",
            target_object=target_object,
            destination_zone=destination_zone,
            scene_snapshot=scene_snapshot,
            confirmation_required=False,
            custom_steps=self._build_steps_from_planner_actions(planner_response.actions),
        )
        controller.set_goal(planner_response.goal)
        controller.record_action_sequence(planner_response.actions)
        validation = self.validate_plan(plan, scene_snapshot)
        return planner_response, plan, validation

    def summarize_status(self) -> str:
        status = self.get_status()
        lines = [
            f"- mode: {status['mode']}",
            f"- execution enabled: {'yes' if status['execution_enabled'] else 'no'}",
            f"- controller: {status['controller_name']}",
            f"- speed: {status['speed']}",
            f"- connected: {'yes' if status['connected'] else 'no'}",
            f"- busy: {'yes' if status['busy'] else 'no'}",
            f"- home: {'yes' if status['is_home'] else 'no'}",
            f"- current zone: {status['current_zone'] or 'unknown'}",
            f"- held object: {status['held_object'] or 'none'}",
            f"- last target handled: {status['last_target_handled'] or 'none'}",
            f"- last plan id: {status['last_plan_id'] or 'none'}",
            f"- last result: {status['last_result_status'] or 'none'}",
            f"- hardware backend: {status['hardware_backend']}",
            f"- hardware dry run: {'yes' if status['hardware_dry_run'] else 'no'}",
            f"- hardware ready: {'yes' if status['hardware_ready'] else 'no'}",
            f"- backend ready for real execution: {'yes' if status['backend_ready_for_real_execution'] else 'no'}",
            f"- bridge running: {'yes' if status['bridge_running'] else 'no'}",
            f"- bridge connected: {'yes' if status['bridge_connected'] else 'no'}",
            f"- lerobot available: {'yes' if status['lerobot_available'] else 'no'}",
            f"- feetech available: {'yes' if status['feetech_support_available'] else 'no'}",
            f"- can connect: {'yes' if status['can_connect'] else 'no'}",
            f"- can home: {'yes' if status['can_home'] else 'no'}",
            f"- can move above object: {'yes' if status['can_move_above_object'] else 'no'}",
            f"- can grasp: {'yes' if status['can_grasp'] else 'no'}",
            f"- can lift: {'yes' if status['can_lift'] else 'no'}",
            f"- can move to zone: {'yes' if status['can_move_to_zone'] else 'no'}",
            f"- can place: {'yes' if status['can_place'] else 'no'}",
            f"- cyberwave backend: {status['cyberwave_backend']}",
            f"- cyberwave connected: {'yes' if status['cyberwave_connected'] else 'no'}",
            f"- cyberwave local ready: {'yes' if status['cyberwave_local_ready'] else 'no'}",
            f"- cyberwave active condition: {status['cyberwave_active_condition']}",
            f"- cyberwave openai configured: {'yes' if status['cyberwave_openai_configured'] else 'no'}",
            f"- bridge last command: {status['bridge_last_command'] or 'none'}",
            f"- last connection attempt: {_timestamp_label(status['last_connection_attempt'])}",
            f"- last hardware error: {status['last_hardware_error'] or 'none'}",
        ]
        if status["cyberwave_last_error"]:
            lines.append(f"- cyberwave last error: {status['cyberwave_last_error']}")
        return "\n".join(lines)

    def summarize_bridge_status(self) -> str:
        bridge_status = self.bridge_status()
        capabilities = bridge_status.get("capabilities", {})
        reasons = bridge_status.get("reasons", [])
        lines = [
            f"- running: {'yes' if bridge_status['bridge_running'] else 'no'}",
            f"- connected: {'yes' if bridge_status['connected'] else 'no'}",
            f"- dry run: {'yes' if bridge_status['dry_run'] else 'no'}",
            f"- backend: {bridge_status['backend_name']}",
            f"- readiness stage: {capabilities.get('readiness_stage', 'unknown')}",
            f"- busy: {'yes' if bridge_status['busy'] else 'no'}",
            f"- current zone: {bridge_status['current_zone'] or 'unknown'}",
            f"- held object: {bridge_status['held_object'] or 'none'}",
            f"- last target handled: {bridge_status['last_target_handled'] or 'none'}",
            f"- last command: {bridge_status['last_command'] or 'none'}",
            f"- last connection attempt: {_timestamp_label(bridge_status['last_connection_attempt'])}",
            f"- config source: {bridge_status['config_source']}",
            f"- config path: {bridge_status['config_path']}",
            f"- last error: {bridge_status['last_error'] or 'none'}",
            f"- real backend ready: {'yes' if bridge_status['real_backend_ready'] else 'no'}",
            f"- lerobot available: {'yes' if bridge_status['lerobot_available'] else 'no'}",
            f"- feetech available: {'yes' if bridge_status['feetech_support_available'] else 'no'}",
            f"- can connect: {'yes' if bridge_status['can_connect'] else 'no'}",
            f"- can home: {'yes' if bridge_status['can_home'] else 'no'}",
        ]
        for warning in bridge_status.get("warnings", []):
            lines.append(f"- warning: {warning}")
        for reason in reasons:
            lines.append(f"- not ready: {reason}")
        return "\n".join(lines)

    def summarize_hardware_status(self) -> str:
        hardware_status = self.hardware_status()
        lines = [
            f"- backend: {hardware_status['hardware_backend']}",
            f"- backend type: {hardware_status['hardware_backend_type']}",
            f"- connected: {'yes' if hardware_status['connected'] else 'no'}",
            f"- busy: {'yes' if hardware_status['busy'] else 'no'}",
            f"- dry run: {'yes' if hardware_status['hardware_dry_run'] else 'no'}",
            f"- ready: {'yes' if hardware_status['hardware_ready'] else 'no'}",
            f"- real backend ready: {'yes' if hardware_status['backend_ready_for_real_execution'] else 'no'}",
            f"- bridge running: {'yes' if hardware_status['bridge_running'] else 'no'}",
            f"- bridge connected: {'yes' if hardware_status['bridge_connected'] else 'no'}",
            f"- lerobot available: {'yes' if hardware_status['lerobot_available'] else 'no'}",
            f"- feetech available: {'yes' if hardware_status['feetech_support_available'] else 'no'}",
            f"- can connect: {'yes' if hardware_status['can_connect'] else 'no'}",
            f"- can home: {'yes' if hardware_status['can_home'] else 'no'}",
            f"- can move above object: {'yes' if hardware_status['can_move_above_object'] else 'no'}",
            f"- can grasp: {'yes' if hardware_status['can_grasp'] else 'no'}",
            f"- can lift: {'yes' if hardware_status['can_lift'] else 'no'}",
            f"- can move to zone: {'yes' if hardware_status['can_move_to_zone'] else 'no'}",
            f"- can place: {'yes' if hardware_status['can_place'] else 'no'}",
            f"- bridge last command: {hardware_status['bridge_last_command'] or 'none'}",
            f"- current zone: {hardware_status['current_zone'] or 'unknown'}",
            f"- held object: {hardware_status['held_object'] or 'none'}",
            f"- last target handled: {hardware_status['last_target_handled'] or 'none'}",
            f"- config source: {hardware_status['bridge_config_source']}",
            f"- config path: {hardware_status['bridge_config_path']}",
            f"- last connection attempt: {_timestamp_label(hardware_status['last_connection_attempt'])}",
            f"- last error: {hardware_status['last_hardware_error'] or 'none'}",
        ]
        for warning in hardware_status.get("bridge_warnings", []):
            lines.append(f"- warning: {warning}")
        for reason in hardware_status.get("bridge_reasons", []):
            lines.append(f"- not ready: {reason}")
        return "\n".join(lines)

    def summarize_capabilities(self) -> str:
        bridge_status = self.bridge_capabilities()
        capabilities = bridge_status.get("capabilities", {})
        lines = [
            f"- backend selected: {capabilities.get('backend_selected', bridge_status['backend_name'])}",
            f"- readiness stage: {capabilities.get('readiness_stage', 'unknown')}",
            f"- dry run: {'yes' if bridge_status['dry_run'] else 'no'}",
            f"- lerobot available: {'yes' if capabilities.get('lerobot_available', False) else 'no'}",
            f"- feetech available: {'yes' if capabilities.get('feetech_support_available', False) else 'no'}",
            f"- real execution allowed by config: {'yes' if capabilities.get('real_execution_allowed_by_config', False) else 'no'}",
            f"- backend ready for real execution: {'yes' if capabilities.get('backend_ready_for_real_execution', False) else 'no'}",
            f"- can connect: {'yes' if capabilities.get('can_connect', False) else 'no'}",
            f"- can home: {'yes' if capabilities.get('can_home', False) else 'no'}",
            f"- can move above object: {'yes' if capabilities.get('can_move_above_object', False) else 'no'}",
            f"- can grasp: {'yes' if capabilities.get('can_grasp', False) else 'no'}",
            f"- can lift: {'yes' if capabilities.get('can_lift', False) else 'no'}",
            f"- can move to zone: {'yes' if capabilities.get('can_move_to_zone', False) else 'no'}",
            f"- can place: {'yes' if capabilities.get('can_place', False) else 'no'}",
        ]
        for warning in capabilities.get("warnings", []):
            lines.append(f"- warning: {warning}")
        for reason in capabilities.get("not_ready_reasons", []):
            lines.append(f"- not ready: {reason}")
        return "\n".join(lines)

    def summarize_backend_info(self) -> str:
        bridge_status = self.bridge_backend_info()
        backend_info = bridge_status.get("backend_info", {})
        dependencies = backend_info.get("dependencies", {})
        robot_config = backend_info.get("robot_config", {})
        lines = [
            f"- backend name: {backend_info.get('backend_name', bridge_status['backend_name'])}",
            f"- backend type: {backend_info.get('backend_type', bridge_status['backend_type'])}",
            f"- lerobot available: {'yes' if dependencies.get('lerobot_available', False) else 'no'}",
            f"- feetech available: {'yes' if dependencies.get('feetech_support_available', False) else 'no'}",
            f"- resolved factory: {dependencies.get('selected_factory_path', 'unresolved') or 'unresolved'}",
            f"- resolved config class: {dependencies.get('selected_config_class_path', 'unresolved') or 'unresolved'}",
            f"- resolved home method: {backend_info.get('resolved_home_method', 'unresolved')}",
            f"- robot type: {robot_config.get('robot_type', 'unknown')}",
            f"- follower port: {robot_config.get('follower_port', '') or 'unset'}",
            f"- leader port: {robot_config.get('leader_port', '') or 'unset'}",
            f"- robot id: {robot_config.get('robot_id', 'unknown')}",
            f"- allow real execution: {'yes' if robot_config.get('allow_real_execution', False) else 'no'}",
            f"- default dry run: {'yes' if robot_config.get('default_dry_run', True) else 'no'}",
            f"- require connection before execute: {'yes' if robot_config.get('require_connection_before_execute', True) else 'no'}",
            f"- config source: {robot_config.get('config_source', 'defaults')}",
            f"- config path: {robot_config.get('config_path', '') or 'unknown'}",
        ]
        for key in (
            "lerobot_import_error",
            "feetech_import_error",
            "factory_error",
            "config_class_error",
        ):
            value = dependencies.get(key, "")
            if value:
                lines.append(f"- {key.replace('_', ' ')}: {value}")
        if backend_info.get("last_backend_message"):
            lines.append(f"- last backend message: {backend_info['last_backend_message']}")
        if backend_info.get("raw_backend_message"):
            lines.append(f"- raw backend message: {backend_info['raw_backend_message']}")
        return "\n".join(lines)

    def summarize_hardware_ready(self) -> str:
        readiness = self.hardware_ready()
        lines = [
            f"- ready for real execution: {'yes' if readiness.get('ready', False) else 'no'}",
            f"- message: {readiness.get('message', 'unknown')}",
            f"- connected: {'yes' if readiness.get('connected', False) else 'no'}",
            f"- dry run: {'yes' if readiness.get('dry_run', True) else 'no'}",
            f"- backend: {readiness.get('backend_name', 'unknown')}",
        ]
        for warning in readiness.get("warnings", []):
            lines.append(f"- warning: {warning}")
        for reason in readiness.get("reasons", []):
            lines.append(f"- not ready: {reason}")
        return "\n".join(lines)

    def summarize_cyberwave_status(self) -> str:
        return self._cyberwave_controller().summarize_status()

    def summarize_cyberwave_scene(self) -> str:
        return self._cyberwave_controller().summarize_scene()

    def summarize_sim_assets(self) -> str:
        return self._cyberwave_controller().summarize_assets()

    def summarize_sim_conditions(self) -> str:
        return self._cyberwave_controller().summarize_conditions()

    def summarize_last_plan(self) -> str:
        if self._last_plan is None:
            return "- no execution plan has been generated yet"
        plan = self._last_plan
        lines = [
            f"- plan id: {plan.plan_id}",
            f"- action: {plan.action_type}",
            f"- target object id: {plan.target_object_id or 'none'}",
            f"- target: {plan.target_name or 'none'}",
            f"- source zone: {plan.source_zone or 'none'}",
            f"- destination zone: {plan.destination_zone or 'none'}",
            f"- confirmation required: {'yes' if plan.confirmation_required else 'no'}",
            f"- execution mode at build time: {plan.execution_mode}",
            f"- scene source: {plan.scene_source}",
            f"- fallback used: {'yes' if plan.fallback_used else 'no'}",
            f"- scene stale at build time: {'yes' if plan.scene_stale else 'no'}",
            f"- created at: {_timestamp_label(plan.created_at)}",
            "- steps:",
        ]
        for index, step in enumerate(plan.steps, start=1):
            lines.append(f"  {index}. {step.description}")
        return "\n".join(lines)


_EXECUTION_MANAGER = ExecutionManager()


def get_execution_manager() -> ExecutionManager:
    return _EXECUTION_MANAGER


def summarize_validation_report(report: ValidationReport) -> list[str]:
    return [issue.message for issue in report.issues]


def build_execution_plan(
    target_object: dict[str, Any] | None,
    destination_zone: str | None = None,
    *,
    action: str = "move",
) -> list[str]:
    """Compatibility helper for older plan-only call sites."""
    plan = _EXECUTION_MANAGER.build_plan(
        action=action,
        target_object=target_object,
        destination_zone=destination_zone,
        scene_snapshot=None,
        confirmation_required=False,
    )
    return [step.description for step in plan.steps]


def pick(object_name: str, location: str) -> str:
    return f"Pick {object_name} from {location}."


def place(object_name: str, destination: str) -> str:
    return f"Place {object_name} at {destination}."


def home() -> str:
    return "Return robot arm to home position."
