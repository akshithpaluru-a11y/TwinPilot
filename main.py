"""Local tabletop robot agent driven by text commands and live workspace vision."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai_planner import PlannerResponse
from robot_control import (
    ExecutionPlan,
    ExecutionResult,
    ValidationReport,
    get_execution_manager,
)
from vision import (
    canonicalize_object_query,
    close_preview_windows,
    detect_destinations,
    filter_scene_objects,
    get_current_metadata,
    get_current_raw_frame,
    get_current_scene,
    get_debug_enabled,
    list_available_cameras,
    pump_preview_events,
    render_preview_once,
    restart_live_loop,
    run_roi_calibration,
    save_debug_snapshot,
    set_debug_enabled,
    set_preview_enabled,
    shutdown_camera,
    start_live_loop,
    stop_live_loop,
    summarize_current_metadata,
    summarize_roi_status,
    summarize_thresholds,
    switch_camera,
)
from voice import poll_listen, speak

EXIT_COMMANDS = {"exit", "quit"}
HELP_COMMANDS = {"help"}
SCENE_COMMANDS = {"scene"}
SCENE_RAW_COMMANDS = {"scene raw"}
DIAGNOSE_COMMANDS = {"diagnose"}
SAVE_DEBUG_COMMANDS = {"save debug"}
ROI_COMMANDS = {"roi", "workspace roi"}
THRESHOLD_COMMANDS = {"thresholds", "show thresholds"}
RECALIBRATE_COMMANDS = {"recalibrate", "calibrate", "recalibrate workspace", "calibrate workspace"}
DEBUG_ON_COMMANDS = {"debug", "debug on"}
DEBUG_OFF_COMMANDS = {"debug off"}
CAMERAS_COMMANDS = {"cameras", "list cameras"}
CAMERA_STATUS_COMMANDS = {"camera status", "status camera"}
START_LIVE_COMMANDS = {"start live"}
STOP_LIVE_COMMANDS = {"stop live"}
RESTART_LIVE_COMMANDS = {"restart live"}
PREVIEW_ON_COMMANDS = {"preview on"}
PREVIEW_OFF_COMMANDS = {"preview off"}
PREVIEW_ONCE_COMMANDS = {"preview once"}
MODE_COMMANDS = {"mode"}
MODE_SIMULATION_COMMANDS = {"mode simulation"}
MODE_CYBERWAVE_COMMANDS = {"mode cyberwave"}
MODE_HARDWARE_COMMANDS = {"mode hardware"}
EXECUTE_ON_COMMANDS = {"execute on"}
EXECUTE_OFF_COMMANDS = {"execute off"}
ROBOT_STATUS_COMMANDS = {"robot status"}
LAST_PLAN_COMMANDS = {"last plan"}
SPEED_COMMANDS = {"speed"}
SPEED_FAST_COMMANDS = {"speed fast"}
SPEED_NORMAL_COMMANDS = {"speed normal"}
SPEED_SLOW_COMMANDS = {"speed slow"}
HARDWARE_STATUS_COMMANDS = {"hardware status"}
HARDWARE_CONNECT_COMMANDS = {"hardware connect"}
HARDWARE_DISCONNECT_COMMANDS = {"hardware disconnect"}
HARDWARE_DRYRUN_ON_COMMANDS = {"hardware dryrun on", "hardware dry run on"}
HARDWARE_DRYRUN_OFF_COMMANDS = {"hardware dryrun off", "hardware dry run off"}
HOME_ROBOT_COMMANDS = {"home robot"}
EXECUTE_PLAN_COMMANDS = {"execute plan"}
ABORT_COMMANDS = {"abort", "abort execution"}
BRIDGE_START_COMMANDS = {"bridge start"}
BRIDGE_STOP_COMMANDS = {"bridge stop"}
BRIDGE_STATUS_COMMANDS = {"bridge status"}
BRIDGE_PING_COMMANDS = {"bridge ping"}
BRIDGE_DRYRUN_ON_COMMANDS = {"bridge dryrun on", "bridge dry run on"}
BRIDGE_DRYRUN_OFF_COMMANDS = {"bridge dryrun off", "bridge dry run off"}
CAPABILITIES_COMMANDS = {"capabilities"}
BACKEND_INFO_COMMANDS = {"backend info"}
HARDWARE_READY_COMMANDS = {"hardware ready"}
CYBERWAVE_STATUS_COMMANDS = {"cyberwave status"}
CYBERWAVE_CONNECT_COMMANDS = {"cyberwave connect"}
CYBERWAVE_DISCONNECT_COMMANDS = {"cyberwave disconnect"}
LIST_ASSETS_COMMANDS = {"list assets"}
LIST_CONDITIONS_COMMANDS = {"list conditions"}
RESET_SIM_COMMANDS = {"reset sim"}
REPLAY_SIM_COMMANDS = {"replay sim"}
SCENE_SIM_COMMANDS = {"scene sim"}
PLAN_SIM_COMMANDS = {"plan sim"}
EXECUTE_SIM_COMMANDS = {"execute sim"}
DEMO_COMMANDS = {"demo"}
SHOWCASE_COMMANDS = {"showcase"}
RESET_SHOWCASE_COMMANDS = {"reset showcase"}
DEMO_SCRIPT_COMMANDS = {"demo script"}
HOME_PHRASES = {"home", "go home", "return home", "reset"}
POLITE_PREFIXES = (
    "please ",
    "robot ",
    "hey robot ",
    "can you ",
    "could you ",
    "would you ",
)
MOVE_PREFIXES = ("move ", "place ", "put ", "relocate ", "set ")
PICK_PREFIXES = ("pick up ", "pick ", "grab ", "take ")
FOLLOW_ON_SEPARATORS = (" and place ", " and put ", " then place ", " then put ")
DESTINATION_MARKERS = (" to ", " into ", " in ", " onto ", " on ", " at ")
LEADING_FILLERS = {"the", "a", "an", "it", "this", "that", "my"}
YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "continue", "do it"}
NO_WORDS = {"no", "n", "nope", "cancel", "stop", "skip"}
ROBOT_MANAGER = get_execution_manager()
SHOWCASE_CONDITION = "demo_showcase_scene"
SHOWCASE_SAMPLE_COMMANDS = [
    "move the black bottle to the right zone",
    "move the cup to the center",
    "move the red cube to the left zone",
    "put the bottle back in the center",
    "pick up the blue cup and place it on the right",
]


class CommandRedirect(Exception):
    """Raised when a user enters a brand-new command during a follow-up prompt."""

    def __init__(self, command_text: str):
        super().__init__(command_text)
        self.command_text = command_text


@dataclass
class BuiltinCommand:
    """Normalized built-in terminal command."""

    name: str
    argument: str | int | None = None


@dataclass
class ParsedCommand:
    """Normalized tabletop command structure."""

    action: str
    raw_target_phrase: str = ""
    raw_destination_phrase: str = ""


@dataclass
class FollowUpState:
    """Represents a pending clarification or confirmation prompt."""

    kind: str
    prompt: str
    retry_prompt: str


def normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.strip().split())


def strip_prefixes(text: str, prefixes: tuple[str, ...]) -> str:
    updated = text
    changed = True
    while changed and updated:
        changed = False
        for prefix in prefixes:
            if updated.startswith(prefix):
                updated = updated[len(prefix) :].strip()
                changed = True
    return updated


def strip_fillers(text: str) -> str:
    words = normalize(text).split()
    while words and words[0] in LEADING_FILLERS:
        words.pop(0)
    return " ".join(words)


def split_destination_phrase(text: str) -> tuple[str, str]:
    padded = f" {normalize(text)} "
    best_index = -1
    best_marker = ""

    for marker in DESTINATION_MARKERS:
        marker_index = padded.find(marker)
        if marker_index != -1 and (best_index == -1 or marker_index < best_index):
            best_index = marker_index
            best_marker = marker

    if best_index == -1:
        return normalize(text), ""

    before = normalize(padded[:best_index])
    after = normalize(padded[best_index + len(best_marker) :])
    return before, after


def find_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return prefix
    return ""


def parse_builtin_command(command: str) -> BuiltinCommand | None:
    normalized_command = normalize(command)
    if not normalized_command:
        return None

    if normalized_command in EXIT_COMMANDS:
        return BuiltinCommand("exit")
    if normalized_command in HELP_COMMANDS:
        return BuiltinCommand("help")
    if normalized_command in SCENE_COMMANDS:
        return BuiltinCommand("scene")
    if normalized_command in SCENE_RAW_COMMANDS:
        return BuiltinCommand("scene_raw")
    if normalized_command in DIAGNOSE_COMMANDS:
        return BuiltinCommand("diagnose")
    if normalized_command in SAVE_DEBUG_COMMANDS:
        return BuiltinCommand("save_debug")
    if normalized_command in ROI_COMMANDS:
        return BuiltinCommand("roi")
    if normalized_command in THRESHOLD_COMMANDS:
        return BuiltinCommand("thresholds")
    if normalized_command in RECALIBRATE_COMMANDS:
        return BuiltinCommand("recalibrate")
    if normalized_command in DEBUG_ON_COMMANDS:
        return BuiltinCommand("debug_on")
    if normalized_command in DEBUG_OFF_COMMANDS:
        return BuiltinCommand("debug_off")
    if normalized_command in CAMERAS_COMMANDS:
        return BuiltinCommand("cameras")
    if normalized_command in CAMERA_STATUS_COMMANDS:
        return BuiltinCommand("camera_status")
    if normalized_command in START_LIVE_COMMANDS:
        return BuiltinCommand("start_live")
    if normalized_command in STOP_LIVE_COMMANDS:
        return BuiltinCommand("stop_live")
    if normalized_command in RESTART_LIVE_COMMANDS:
        return BuiltinCommand("restart_live")
    if normalized_command in PREVIEW_ON_COMMANDS:
        return BuiltinCommand("preview_on")
    if normalized_command in PREVIEW_OFF_COMMANDS:
        return BuiltinCommand("preview_off")
    if normalized_command in PREVIEW_ONCE_COMMANDS:
        return BuiltinCommand("preview_once")
    if normalized_command in MODE_COMMANDS:
        return BuiltinCommand("mode")
    if normalized_command in MODE_SIMULATION_COMMANDS:
        return BuiltinCommand("mode_set", "simulation")
    if normalized_command in MODE_CYBERWAVE_COMMANDS:
        return BuiltinCommand("mode_set", "cyberwave")
    if normalized_command in MODE_HARDWARE_COMMANDS:
        return BuiltinCommand("mode_set", "hardware")
    if normalized_command in EXECUTE_ON_COMMANDS:
        return BuiltinCommand("execute_set", "on")
    if normalized_command in EXECUTE_OFF_COMMANDS:
        return BuiltinCommand("execute_set", "off")
    if normalized_command in ROBOT_STATUS_COMMANDS:
        return BuiltinCommand("robot_status")
    if normalized_command in LAST_PLAN_COMMANDS:
        return BuiltinCommand("last_plan")
    if normalized_command in SPEED_COMMANDS:
        return BuiltinCommand("speed")
    if normalized_command in SPEED_FAST_COMMANDS:
        return BuiltinCommand("speed_set", "fast")
    if normalized_command in SPEED_NORMAL_COMMANDS:
        return BuiltinCommand("speed_set", "normal")
    if normalized_command in SPEED_SLOW_COMMANDS:
        return BuiltinCommand("speed_set", "slow")
    if normalized_command in HARDWARE_STATUS_COMMANDS:
        return BuiltinCommand("hardware_status")
    if normalized_command in HARDWARE_CONNECT_COMMANDS:
        return BuiltinCommand("hardware_connect")
    if normalized_command in HARDWARE_DISCONNECT_COMMANDS:
        return BuiltinCommand("hardware_disconnect")
    if normalized_command in HARDWARE_DRYRUN_ON_COMMANDS:
        return BuiltinCommand("hardware_dryrun_set", "on")
    if normalized_command in HARDWARE_DRYRUN_OFF_COMMANDS:
        return BuiltinCommand("hardware_dryrun_set", "off")
    if normalized_command in HOME_ROBOT_COMMANDS:
        return BuiltinCommand("home_robot")
    if normalized_command in EXECUTE_PLAN_COMMANDS:
        return BuiltinCommand("execute_plan")
    if normalized_command in ABORT_COMMANDS:
        return BuiltinCommand("abort")
    if normalized_command in BRIDGE_START_COMMANDS:
        return BuiltinCommand("bridge_start")
    if normalized_command in BRIDGE_STOP_COMMANDS:
        return BuiltinCommand("bridge_stop")
    if normalized_command in BRIDGE_STATUS_COMMANDS:
        return BuiltinCommand("bridge_status")
    if normalized_command in BRIDGE_PING_COMMANDS:
        return BuiltinCommand("bridge_ping")
    if normalized_command in BRIDGE_DRYRUN_ON_COMMANDS:
        return BuiltinCommand("bridge_dryrun_set", "on")
    if normalized_command in BRIDGE_DRYRUN_OFF_COMMANDS:
        return BuiltinCommand("bridge_dryrun_set", "off")
    if normalized_command in CAPABILITIES_COMMANDS:
        return BuiltinCommand("capabilities")
    if normalized_command in BACKEND_INFO_COMMANDS:
        return BuiltinCommand("backend_info")
    if normalized_command in HARDWARE_READY_COMMANDS:
        return BuiltinCommand("hardware_ready")
    if normalized_command in CYBERWAVE_STATUS_COMMANDS:
        return BuiltinCommand("cyberwave_status")
    if normalized_command in CYBERWAVE_CONNECT_COMMANDS:
        return BuiltinCommand("cyberwave_connect")
    if normalized_command in CYBERWAVE_DISCONNECT_COMMANDS:
        return BuiltinCommand("cyberwave_disconnect")
    if normalized_command in LIST_ASSETS_COMMANDS:
        return BuiltinCommand("list_assets")
    if normalized_command in LIST_CONDITIONS_COMMANDS:
        return BuiltinCommand("list_conditions")
    if normalized_command in RESET_SIM_COMMANDS:
        return BuiltinCommand("reset_sim")
    if normalized_command in REPLAY_SIM_COMMANDS:
        return BuiltinCommand("replay_sim")
    if normalized_command in SCENE_SIM_COMMANDS:
        return BuiltinCommand("scene_sim")
    if normalized_command in PLAN_SIM_COMMANDS:
        return BuiltinCommand("plan_sim")
    if normalized_command in EXECUTE_SIM_COMMANDS:
        return BuiltinCommand("execute_sim")
    if normalized_command in DEMO_COMMANDS:
        return BuiltinCommand("demo")
    if normalized_command in SHOWCASE_COMMANDS:
        return BuiltinCommand("showcase")
    if normalized_command in RESET_SHOWCASE_COMMANDS:
        return BuiltinCommand("reset_showcase")
    if normalized_command in DEMO_SCRIPT_COMMANDS:
        return BuiltinCommand("demo_script")

    camera_match = re.fullmatch(r"use camera (\d+)", normalized_command)
    if camera_match:
        return BuiltinCommand("use_camera", int(camera_match.group(1)))
    condition_match = re.fullmatch(r"load condition (.+)", normalized_command)
    if condition_match:
        return BuiltinCommand("load_condition", condition_match.group(1))

    return None


def parse_command(command: str) -> ParsedCommand | None:
    normalized_command = normalize(command)
    normalized_command = strip_prefixes(normalized_command, POLITE_PREFIXES)

    if not normalized_command:
        return None

    if normalized_command in HOME_PHRASES:
        return ParsedCommand(action="home")

    pick_prefix = find_prefix(normalized_command, PICK_PREFIXES)
    if pick_prefix:
        remainder = normalized_command[len(pick_prefix) :]
        for separator in FOLLOW_ON_SEPARATORS:
            if separator in remainder:
                object_part, destination_part = remainder.split(separator, 1)
                _, destination_query = split_destination_phrase(destination_part)
                if not destination_query:
                    destination_query = strip_fillers(destination_part)
                return ParsedCommand(
                    action="move",
                    raw_target_phrase=strip_fillers(object_part) or "object",
                    raw_destination_phrase=strip_fillers(destination_query),
                )
        return ParsedCommand(action="pick", raw_target_phrase=strip_fillers(remainder) or "object")

    move_prefix = find_prefix(normalized_command, MOVE_PREFIXES)
    if move_prefix:
        remainder = normalized_command[len(move_prefix) :]
        object_part, destination_part = split_destination_phrase(remainder)
        return ParsedCommand(
            action="move",
            raw_target_phrase=strip_fillers(object_part) or "object",
            raw_destination_phrase=strip_fillers(destination_part),
        )

    return None


def build_destination_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for destination in detect_destinations():
        canonical_name = normalize(str(destination["name"]))
        aliases[canonical_name] = canonical_name
        for alias in destination["aliases"]:
            aliases[normalize(alias)] = canonical_name
    return aliases


DESTINATION_ALIASES = build_destination_aliases()


def canonical_destination_text(query: str) -> str:
    normalized_query = strip_fillers(query)
    normalized_query = strip_prefixes(
        normalized_query,
        ("to ", "into ", "in ", "onto ", "on ", "at "),
    )
    normalized_query = strip_fillers(normalized_query)
    if not normalized_query:
        return ""

    if normalized_query in DESTINATION_ALIASES:
        return DESTINATION_ALIASES[normalized_query]

    query_words = normalized_query.split()
    matches = {
        canonical
        for alias, canonical in DESTINATION_ALIASES.items()
        if all(word in alias for word in query_words)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return ""


def is_yes(text: str) -> bool:
    normalized = normalize(text)
    words = set(normalized.split())
    return normalized in YES_WORDS or bool(words.intersection(YES_WORDS))


def is_no(text: str) -> bool:
    normalized = normalize(text)
    words = set(normalized.split())
    return normalized in NO_WORDS or bool(words.intersection(NO_WORDS))


def is_new_command(text: str) -> bool:
    return parse_builtin_command(text) is not None or parse_command(text) is not None


def prompt_user(state: FollowUpState) -> str:
    speak(state.prompt)
    while True:
        response = wait_for_user_input()
        if not normalize(response):
            speak(state.retry_prompt)
            continue
        return response


def maybe_redirect(response: str) -> None:
    if is_new_command(response):
        raise CommandRedirect(response)


def format_object_option(scene_object: dict[str, Any]) -> str:
    return f"{scene_object['name']} at {scene_object['zone']}"


def resolve_object(query: str, scene_objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = filter_scene_objects(scene_objects, query)
    if len(matches) == 1:
        return matches[0]

    if not matches:
        speak(f"I could not find {canonicalize_object_query(query)} in the current workspace.")
        return None

    options = ", ".join(format_object_option(object_) for object_ in matches)
    state = FollowUpState(
        kind="object_clarification",
        prompt=f"I found multiple matches: {options}. Which object do you mean?",
        retry_prompt="Please name one visible object, or enter a new command.",
    )
    available_choices = matches if len(matches) < len(scene_objects) else scene_objects

    while True:
        response = prompt_user(state)
        maybe_redirect(response)
        clarified_matches = filter_scene_objects(available_choices, response)
        if len(clarified_matches) == 1:
            return clarified_matches[0]
        if len(clarified_matches) > 1:
            options = ", ".join(format_object_option(object_) for object_ in clarified_matches)
            state.prompt = f"I still have multiple matches: {options}. Which one should I use?"
            continue
        state.prompt = "I still could not identify the object. Which visible item should I use?"


def resolve_destination(query: str) -> str | None:
    destination = canonical_destination_text(query)
    if destination:
        return destination

    state = FollowUpState(
        kind="destination_clarification",
        prompt="Where should I place it? Choose left, center, or right.",
        retry_prompt="Please answer with left, center, or right, or enter a new command.",
    )
    if strip_fillers(query):
        state.prompt = "I only support left, center, or right. Which zone do you mean?"

    while True:
        response = prompt_user(state)
        maybe_redirect(response)
        destination = canonical_destination_text(response)
        if destination:
            return destination
        state.prompt = "Please say left, center, or right, or enter a new command."


def confirm_same_zone(target_object: dict[str, Any], destination: str) -> bool:
    if normalize(str(target_object["zone"])) != destination:
        return True

    state = FollowUpState(
        kind="same_zone_confirmation",
        prompt=(
            f"{target_object['name']} is already in the {destination} zone. "
            "Do you still want to move it?"
        ),
        retry_prompt="Please answer yes or no, or enter a new command.",
    )

    while True:
        response = prompt_user(state)
        maybe_redirect(response)
        if is_yes(response):
            return True
        if is_no(response):
            speak("Okay. I will leave it where it is.")
            return False
        state.prompt = "Please answer yes or no, or enter a new command."


def fetch_scene_snapshot(wait_timeout: float = 0.8) -> dict[str, Any] | None:
    snapshot = get_current_scene(wait_for_update=True, timeout=wait_timeout)
    metadata = snapshot["metadata"]
    if metadata["last_update_time"] is None and not snapshot["objects"]:
        return None
    return snapshot


def wait_for_user_input(prompt: str = "You: ") -> str:
    while True:
        pump_preview_events()
        response = poll_listen(prompt=prompt, timeout=0.1)
        if response is not None:
            return response


def print_scene_report(scene_snapshot: dict[str, Any], *, include_metadata: bool = False) -> None:
    metadata = scene_snapshot["metadata"]
    header = "Detected scene:"
    if scene_snapshot["fallback_used"]:
        header = "Detected scene (fallback mock):"
    elif metadata["using_cached_scene"]:
        header = "Detected scene (recent live cache):"

    print(f"\n{header}")
    print(scene_snapshot["summary"])

    if metadata["scene_stale"]:
        print(f"- warning: scene is stale ({metadata['last_update_age_seconds']:.2f}s old)")
    if metadata["using_cached_scene"]:
        print("- note: using the most recent stable live scene cache")
    if metadata["fallback_reason"]:
        print(f"- reason: {metadata['fallback_reason']}")

    if include_metadata or get_debug_enabled():
        print(f"- source: {scene_snapshot['source']}")
        print(f"- detection mode: {metadata['detection_mode']}")
        print(f"- camera index: {metadata['camera_index']}")
        print(f"- roi source: {scene_snapshot['roi_source']}")
        if scene_snapshot["workspace_roi"] is not None:
            print(f"- workspace roi: {tuple(scene_snapshot['workspace_roi'])}")
        print(f"- raw candidates: {metadata['raw_candidates_count']}")
        print(f"- valid detections: {metadata['valid_detections_count']}")


def print_parsed_command(parsed_command: ParsedCommand) -> None:
    print("\nParsed command:")
    print(f"- action: {parsed_command.action}")
    print(f"- raw target phrase: {parsed_command.raw_target_phrase or 'unspecified'}")
    print(f"- raw destination phrase: {parsed_command.raw_destination_phrase or 'unspecified'}")


def print_resolved_target(target_object: dict[str, Any] | None) -> None:
    print("\nResolved target:")
    if target_object is None:
        print("- no valid target object found")
        return
    print(
        f"- {target_object['name']} at {target_object['zone']} "
        f"(confidence {float(target_object['confidence']):.2f})"
    )
    if get_debug_enabled():
        print(
            f"- id: {target_object['id']} | label: {target_object['label']} | "
            f"bbox: {tuple(target_object['bbox'])} | area: {float(target_object['area']):.0f}"
        )


def print_resolved_destination(destination: str | None) -> None:
    print("\nResolved destination:")
    print(f"- {destination or 'unspecified'}")


def print_execution_plan(plan: ExecutionPlan) -> None:
    print("\nExecution plan:")
    for step_number, step in enumerate(plan.steps, start=1):
        print(f"{step_number}. {step.description}")
    print()


def print_execution_mode() -> None:
    status = ROBOT_MANAGER.get_status()
    print("Execution mode:")
    print(f"- mode: {status['mode']}")
    print(f"- execution enabled: {'yes' if status['execution_enabled'] else 'no'}")
    print(f"- controller: {status['controller_name']}")
    print(f"- speed: {status['speed']}")
    if status["mode"] == "cyberwave":
        print(f"- backend: {status['cyberwave_backend']}")
        print(f"- cyberwave connected: {'yes' if status['cyberwave_connected'] else 'no'}")
        print(f"- local sim ready: {'yes' if status['cyberwave_local_ready'] else 'no'}")
        print(f"- active condition: {status['cyberwave_active_condition']}")
        print(f"- openai planner enabled: {'yes' if status['cyberwave_openai_enabled'] else 'no'}")
        print(f"- openai api configured: {'yes' if status['cyberwave_openai_configured'] else 'no'}")
        print(f"- openai available: {'yes' if status['cyberwave_openai_available'] else 'no'}")
        print(f"- openai model: {status['cyberwave_openai_model']}")
        if status["cyberwave_last_error"]:
            print(f"- cyberwave last error: {status['cyberwave_last_error']}")
    if status["mode"] == "hardware":
        print(f"- bridge running: {'yes' if status['bridge_running'] else 'no'}")
        print(f"- bridge connected: {'yes' if status['bridge_connected'] else 'no'}")
        print(f"- dry run: {'on' if status['bridge_dry_run'] else 'off'}")
        print(f"- backend: {status['hardware_backend']}")
        print(f"- ready: {'yes' if status['hardware_ready'] else 'no'}")
        print(f"- real backend ready: {'yes' if status['hardware_real_ready'] else 'no'}")
        print(f"- lerobot available: {'yes' if status['lerobot_available'] else 'no'}")
        print(f"- feetech available: {'yes' if status['feetech_support_available'] else 'no'}")
        print(f"- can home: {'yes' if status['can_home'] else 'no'}")
        if status["last_hardware_error"]:
            print(f"- last hardware error: {status['last_hardware_error']}")
    print()


def print_validation_report(report: ValidationReport) -> None:
    print("Validation:")
    print(f"- executable: {'yes' if report.can_execute else 'no'}")
    if report.errors:
        for issue in report.errors:
            print(f"- error: {issue.message}")
    else:
        print("- no validation errors")
    print()

    print("Warnings:")
    if report.warnings:
        for issue in report.warnings:
            print(f"- {issue.message}")
    else:
        print("- none")
    print()


def print_execution_result(result: ExecutionResult) -> None:
    print("Execution result:")
    print(f"- status: {result.status}")
    print(f"- controller: {result.controller_name}")
    if result.backend_name:
        print(f"- backend: {result.backend_name}")
    print(f"- dry run only: {'yes' if result.dry_run else 'no'}")
    print(f"- supported: {'yes' if result.supported else 'no'}")
    print(f"- steps completed: {result.completed_steps}/{result.total_steps}")
    if result.aborted:
        print("- aborted: yes")
    if result.failed_step:
        print(f"- failed step: {result.failed_step}")
    if result.refusal_reason:
        print(f"- refusal reason: {result.refusal_reason}")
    print(f"- message: {result.message}")
    for warning in result.warnings:
        print(f"- warning: {warning}")
    if result.raw_backend_message:
        print(f"- raw backend message: {result.raw_backend_message}")
    if result.step_results:
        print("- step results:")
        for step_result in result.step_results:
            line = (
                f"  - {step_result.primitive_name or step_result.step_id}: "
                f"{step_result.status} | {step_result.message}"
            )
            if not step_result.supported:
                line += " | unsupported"
            if step_result.dry_run:
                line += " | dry-run"
            print(line)
            if step_result.refusal_reason:
                print(f"    refusal: {step_result.refusal_reason}")
            for warning in step_result.warnings:
                print(f"    warning: {warning}")
            if step_result.raw_backend_message:
                print(f"    raw: {step_result.raw_backend_message}")
    print()


def print_last_plan() -> None:
    print("\nLast plan:")
    print(ROBOT_MANAGER.summarize_last_plan())
    print()


def print_robot_status() -> None:
    print("\nRobot status:")
    print(ROBOT_MANAGER.summarize_status())
    print()


def print_hardware_status() -> None:
    print("\nHardware status:")
    print(ROBOT_MANAGER.summarize_hardware_status())
    print()


def print_bridge_status() -> None:
    print("\nBridge status:")
    print(ROBOT_MANAGER.summarize_bridge_status())
    print()


def print_capabilities() -> None:
    print("\nCapabilities:")
    print(ROBOT_MANAGER.summarize_capabilities())
    print()


def print_backend_info() -> None:
    print("\nBackend info:")
    print(ROBOT_MANAGER.summarize_backend_info())
    print()


def print_hardware_ready() -> None:
    print("\nHardware ready:")
    print(ROBOT_MANAGER.summarize_hardware_ready())
    print()


def print_cyberwave_status() -> None:
    print("\nCyberWave status:")
    print(ROBOT_MANAGER.summarize_cyberwave_status())
    print()


def print_sim_assets() -> None:
    print("\nAssets:")
    print(ROBOT_MANAGER.summarize_sim_assets())
    print()


def print_sim_conditions() -> None:
    print("\nConditions:")
    print(ROBOT_MANAGER.summarize_sim_conditions())
    print()


def print_sim_scene() -> None:
    print("\nSimulation scene:")
    print(ROBOT_MANAGER.summarize_cyberwave_scene())
    print()


def summarize_sim_positions() -> str:
    snapshot = ROBOT_MANAGER.get_sim_scene_snapshot()
    objects = snapshot.get("objects", [])
    if not objects:
        return "none"
    parts = [
        f"{object_.get('display_name', object_.get('name', 'object'))} at {object_['zone']}"
        for object_ in objects
    ]
    return ", ".join(parts)


def print_showcase_banner(*, reset: bool = False) -> None:
    snapshot = ROBOT_MANAGER.get_sim_scene_snapshot()
    action_label = "Showcase reset" if reset else "Showcase ready"
    print(f"\n{action_label}:")
    print(f"- mode: {ROBOT_MANAGER.get_mode()}")
    print(f"- scene: {snapshot.get('active_condition', SHOWCASE_CONDITION)}")
    print(
        f"- execution: {'on' if ROBOT_MANAGER.is_execution_enabled() else 'off'}"
    )
    print(f"- objects: {summarize_sim_positions()}")
    print("- say:")
    for sample in SHOWCASE_SAMPLE_COMMANDS[:3]:
        print(f"  - {sample}")
    print()


def print_demo_script() -> None:
    print("\nDemo script:")
    for sample in SHOWCASE_SAMPLE_COMMANDS:
        print(f"- {sample}")
    print()


def print_simulation_result_summary(plan: ExecutionPlan) -> None:
    if not plan.target_name:
        return
    print("\nSimulation summary:")
    print(f"- moved: {plan.target_name}")
    print(f"- from: {plan.source_zone or 'unknown'}")
    print(f"- to: {plan.destination_zone or plan.source_zone or 'unknown'}")
    print(f"- positions: {summarize_sim_positions()}")
    print()


def print_sim_request(command_text: str) -> None:
    snapshot = ROBOT_MANAGER.get_sim_scene_snapshot()
    print("\nPlanner request:")
    print(f"- raw user command: {command_text}")
    if snapshot["objects"]:
        print("- active objects:")
        for object_ in snapshot["objects"]:
            object_id = object_.get("object_id", object_.get("id", "unknown"))
            display_name = object_.get("display_name", object_.get("name", "object"))
            print(f"  - {object_id} / {display_name} / {object_['zone']}")
    else:
        print("- active objects: none")
    print("- allowed zones: left, center, right, home")
    if get_debug_enabled():
        print(f"- active condition: {snapshot.get('active_condition', 'unknown')}")


def print_planner_response(response: PlannerResponse) -> None:
    print("\nPlanner result:")
    print(f"- success: {'yes' if response.success else 'no'}")
    if response.goal:
        print(f"- goal: {response.goal}")
    if response.target_object_id:
        print(f"- target_object_id: {response.target_object_id}")
    if response.target_display_name:
        print(f"- target_display_name: {response.target_display_name}")
    if response.destination_zone:
        print(f"- destination_zone: {response.destination_zone}")
    print(f"- clarification needed: {'yes' if response.needs_clarification else 'no'}")
    if response.clarification_question:
        print(f"- clarification question: {response.clarification_question}")
    if response.error:
        print(f"- error: {response.error}")
    if response.error_type:
        print(f"- error type: {response.error_type}")
    if response.model:
        print(f"- model: {response.model}")
    if response.actions:
        print("- actions:")
        for action in response.actions:
            detail = f"  - {action.action_type}"
            if action.target_object_id:
                detail += f" | target_object_id={action.target_object_id}"
            if action.target_display_name:
                detail += f" | target_display_name={action.target_display_name}"
            if action.zone:
                detail += f" | zone={action.zone}"
            print(detail)
    print()


def print_scene_command(snapshot: dict[str, Any], *, raw: bool = False) -> None:
    print_scene_report(snapshot, include_metadata=raw)
    if raw:
        print("\nRaw scene state:")
        print(json.dumps(snapshot, indent=2))
        print()
        return

    if get_debug_enabled() and snapshot["objects"]:
        print("\nDebug objects:")
        for object_ in snapshot["objects"]:
            print(
                f"- {object_.get('object_id', object_['id'])}: "
                f"{object_.get('display_name', object_['name'])} | label={object_['label']} | "
                f"zone={object_['zone']} | bbox={tuple(object_['bbox'])} | area={float(object_['area']):.0f}"
            )
        print()


def print_diagnose(snapshot: dict[str, Any]) -> None:
    print_scene_report(snapshot, include_metadata=True)
    print("\nDiagnosis:")
    print(summarize_current_metadata())
    print()


def print_camera_list() -> None:
    print("\nCamera probe:")
    for candidate in list_available_cameras():
        label = f"camera {candidate['index']}: {candidate['status']}"
        if candidate["active"]:
            label += " (active)"
        print(f"- {label}")
    print()


def print_help() -> None:
    print(
        "\nCommands:\n"
        "- help\n"
        "- scene\n"
        "- scene raw\n"
        "- diagnose\n"
        "- save debug\n"
        "- roi\n"
        "- thresholds\n"
        "- cameras\n"
        "- camera status\n"
        "- use camera 0\n"
        "- start live\n"
        "- stop live\n"
        "- restart live\n"
        "- mode\n"
        "- mode simulation\n"
        "- mode cyberwave\n"
        "- mode hardware\n"
        "- execute on\n"
        "- execute off\n"
        "- execute plan\n"
        "- robot status\n"
        "- hardware status\n"
        "- hardware connect\n"
        "- hardware disconnect\n"
        "- hardware dryrun on\n"
        "- hardware dryrun off\n"
        "- bridge start\n"
        "- bridge stop\n"
        "- bridge status\n"
        "- bridge ping\n"
        "- bridge dryrun on\n"
        "- bridge dryrun off\n"
        "- capabilities\n"
        "- backend info\n"
        "- hardware ready\n"
        "- cyberwave status\n"
        "- cyberwave connect\n"
        "- cyberwave disconnect\n"
        "- list assets\n"
        "- list conditions\n"
        "- load condition demo_showcase_scene\n"
        "- scene sim\n"
        "- plan sim\n"
        "- execute sim\n"
        "- reset sim\n"
        "- replay sim\n"
        "- demo\n"
        "- showcase\n"
        "- reset showcase\n"
        "- demo script\n"
        "- last plan\n"
        "- home robot\n"
        "- abort\n"
        "- speed\n"
        "- speed fast\n"
        "- speed normal\n"
        "- speed slow\n"
        "- preview on\n"
        "- preview off\n"
        "- preview once\n"
        "- recalibrate\n"
        "- debug on\n"
        "- debug off\n"
        "- exit\n"
        "\nRobot examples:\n"
        "- move the black bottle to the right side\n"
        "- move the bottle to the left\n"
        "- move the dark object to the center\n"
        "- move the cup to the center\n"
        "- move the object to the center\n"
        "- pick up the black bottle and place it on the right\n"
        "- place the bottle in the left zone\n"
        "- home\n"
        "\nCyberWave demo flow:\n"
        "- showcase\n"
        "- move the black bottle to the right zone\n"
    )


def activate_cyberwave_showcase(*, reset: bool = False) -> None:
    ROBOT_MANAGER.set_mode("cyberwave")
    ROBOT_MANAGER.load_sim_condition(SHOWCASE_CONDITION)
    ROBOT_MANAGER.set_execution_enabled(True)
    set_preview_enabled(False)
    close_preview_windows()
    print_showcase_banner(reset=reset)


def handle_builtin_command(command: BuiltinCommand) -> None:
    if command.name == "help":
        print_help()
        return

    if command.name == "scene":
        snapshot = fetch_scene_snapshot()
        if snapshot is None:
            speak("The live loop has not produced a scene yet.")
            return
        print_scene_command(snapshot)
        return

    if command.name == "scene_raw":
        snapshot = fetch_scene_snapshot()
        if snapshot is None:
            speak("The live loop has not produced a scene yet.")
            return
        print_scene_command(snapshot, raw=True)
        return

    if command.name == "diagnose":
        snapshot = fetch_scene_snapshot()
        if snapshot is None:
            speak("The live loop has not produced a scene yet.")
            return
        print_diagnose(snapshot)
        return

    if command.name == "save_debug":
        success, message = save_debug_snapshot()
        speak(message)
        return

    if command.name == "roi":
        print("\nWorkspace ROI:")
        print(summarize_roi_status())
        print()
        return

    if command.name == "thresholds":
        print("\nDetection thresholds:")
        print(summarize_thresholds())
        print()
        return

    if command.name == "recalibrate":
        metadata = get_current_metadata()
        live_was_running = bool(metadata["live_loop_running"])
        frame = get_current_raw_frame()
        if frame is None:
            fetch_scene_snapshot(wait_timeout=1.0)
            frame = get_current_raw_frame()
        if frame is None:
            speak("I do not have a current camera frame for recalibration.")
            return

        if live_was_running:
            speak("Pausing live vision for workspace calibration.")
            stop_live_loop()

        close_preview_windows()
        speak("Opening workspace calibration. Drag a rectangle over the tabletop and press Enter to save.")
        success, message = run_roi_calibration(frame)

        restart_message = ""
        if live_was_running:
            _, restart_message = start_live_loop()

        speak(message)
        if restart_message:
            speak(restart_message)
        if success:
            print("\nWorkspace ROI:")
            print(summarize_roi_status())
            print()
        return

    if command.name == "debug_on":
        set_debug_enabled(True)
        speak("Debug mode is on.")
        return

    if command.name == "debug_off":
        set_debug_enabled(False)
        speak("Debug mode is off.")
        return

    if command.name == "cameras":
        print_camera_list()
        return

    if command.name == "camera_status":
        print("\nCamera status:")
        print(summarize_current_metadata())
        print()
        return

    if command.name == "mode":
        print_execution_mode()
        return

    if command.name == "mode_set":
        _, message = ROBOT_MANAGER.set_mode(str(command.argument))
        if str(command.argument) == "cyberwave":
            set_preview_enabled(False)
            close_preview_windows()
        speak(message)
        if str(command.argument) == "cyberwave":
            print_cyberwave_status()
        return

    if command.name == "execute_set":
        enabled = str(command.argument) == "on"
        _, message = ROBOT_MANAGER.set_execution_enabled(enabled)
        speak(message)
        return

    if command.name == "robot_status":
        print_robot_status()
        return

    if command.name == "hardware_status":
        print_hardware_status()
        return

    if command.name == "bridge_status":
        print_bridge_status()
        return

    if command.name == "bridge_start":
        success, message = ROBOT_MANAGER.bridge_start()
        speak(message)
        if success or get_debug_enabled():
            print_bridge_status()
        return

    if command.name == "bridge_stop":
        success, message = ROBOT_MANAGER.bridge_stop()
        speak(message)
        if success or get_debug_enabled():
            print_bridge_status()
        return

    if command.name == "bridge_ping":
        success, message = ROBOT_MANAGER.bridge_ping()
        speak(message)
        if success or get_debug_enabled():
            print_bridge_status()
        return

    if command.name == "bridge_dryrun_set":
        enabled = str(command.argument) == "on"
        _, message = ROBOT_MANAGER.set_bridge_dry_run(enabled)
        speak(message)
        print_bridge_status()
        return

    if command.name == "capabilities":
        print_capabilities()
        return

    if command.name == "backend_info":
        print_backend_info()
        return

    if command.name == "hardware_ready":
        print_hardware_ready()
        return

    if command.name == "cyberwave_status":
        print_cyberwave_status()
        return

    if command.name == "cyberwave_connect":
        success, message = ROBOT_MANAGER.cyberwave_connect()
        speak(message)
        if success or get_debug_enabled():
            print_cyberwave_status()
        return

    if command.name == "cyberwave_disconnect":
        success, message = ROBOT_MANAGER.cyberwave_disconnect()
        speak(message)
        if success or get_debug_enabled():
            print_cyberwave_status()
        return

    if command.name == "list_assets":
        print_sim_assets()
        return

    if command.name == "list_conditions":
        print_sim_conditions()
        return

    if command.name == "load_condition":
        success, message = ROBOT_MANAGER.load_sim_condition(str(command.argument or ""))
        speak(message)
        if success:
            print_sim_scene()
        return

    if command.name == "reset_sim":
        success, message = ROBOT_MANAGER.reset_sim()
        speak(message)
        if success:
            print_sim_scene()
        return

    if command.name == "replay_sim":
        success, message = ROBOT_MANAGER.replay_sim()
        speak(message)
        if success or get_debug_enabled():
            print_sim_scene()
        return

    if command.name == "scene_sim":
        print_sim_scene()
        return

    if command.name == "plan_sim":
        status = ROBOT_MANAGER.get_status()
        last_prompt = status.get("last_cyberwave_prompt")
        if not last_prompt:
            speak("Type a natural-language simulation command first, then run plan sim.")
            return
        planner_response, plan, validation = ROBOT_MANAGER.plan_cyberwave_command(str(last_prompt))
        print_sim_scene()
        print_sim_request(str(last_prompt))
        print_planner_response(planner_response)
        if plan is not None and validation is not None:
            print_execution_plan(plan)
            print_execution_mode()
            print_validation_report(validation)
        return

    if command.name == "execute_sim":
        plan = ROBOT_MANAGER.get_last_plan()
        if plan is None or plan.execution_mode != "cyberwave":
            speak("No CyberWave simulation plan is ready. Run a simulation command or plan sim first.")
            return
        validation = ROBOT_MANAGER.get_last_validation()
        if validation is None:
            validation = ROBOT_MANAGER.validate_plan(plan, ROBOT_MANAGER.get_sim_scene_snapshot())
        print_execution_plan(plan)
        print_execution_mode()
        print_validation_report(validation)
        result = ROBOT_MANAGER.execute_plan(plan, validation, force=True)
        print_execution_result(result)
        if result.status == "success":
            print_simulation_result_summary(plan)
        return

    if command.name == "demo":
        activate_cyberwave_showcase()
        return

    if command.name == "showcase":
        activate_cyberwave_showcase()
        return

    if command.name == "reset_showcase":
        activate_cyberwave_showcase(reset=True)
        return

    if command.name == "demo_script":
        print_demo_script()
        return

    if command.name == "hardware_connect":
        success, message = ROBOT_MANAGER.hardware_connect()
        speak(message)
        if success or get_debug_enabled():
            print_hardware_status()
        return

    if command.name == "hardware_disconnect":
        success, message = ROBOT_MANAGER.hardware_disconnect()
        speak(message)
        if success or get_debug_enabled():
            print_hardware_status()
        return

    if command.name == "hardware_dryrun_set":
        enabled = str(command.argument) == "on"
        _, message = ROBOT_MANAGER.set_hardware_dry_run(enabled)
        speak(message)
        print_hardware_status()
        return

    if command.name == "last_plan":
        print_last_plan()
        return

    if command.name == "execute_plan":
        plan, validation = ROBOT_MANAGER.prepare_last_plan_execution()
        if plan is None or validation is None:
            _, _, result = ROBOT_MANAGER.execute_last_plan()
            print_execution_result(result)
            return
        print_execution_plan(plan)
        print_execution_mode()
        print_validation_report(validation)
        result = ROBOT_MANAGER.execute_plan(plan, validation, force=True)
        print_execution_result(result)
        if result.status == "success" and plan.execution_mode == "cyberwave":
            print_simulation_result_summary(plan)
        return

    if command.name == "home_robot":
        scene_snapshot = (
            ROBOT_MANAGER.get_sim_scene_snapshot()
            if ROBOT_MANAGER.get_mode() == "cyberwave"
            else fetch_scene_snapshot(wait_timeout=0.5)
        )
        plan = ROBOT_MANAGER.build_plan(
            action="home",
            target_object=None,
            destination_zone=None,
            scene_snapshot=scene_snapshot,
            confirmation_required=False,
        )
        validation = ROBOT_MANAGER.validate_plan(plan, scene_snapshot)
        if scene_snapshot is not None:
            if ROBOT_MANAGER.get_mode() == "cyberwave":
                print_sim_scene()
            else:
                print_scene_report(scene_snapshot)
        print_execution_plan(plan)
        print_execution_mode()
        print_validation_report(validation)
        result = ROBOT_MANAGER.execute_plan(plan, validation, force=True)
        print_execution_result(result)
        return

    if command.name == "abort":
        success, message = ROBOT_MANAGER.abort_execution()
        speak(message)
        if success or get_debug_enabled():
            print_robot_status()
        return

    if command.name == "speed":
        print("\nSimulation speed:")
        print(f"- {ROBOT_MANAGER.get_speed()}")
        print()
        return

    if command.name == "speed_set":
        _, message = ROBOT_MANAGER.set_speed(str(command.argument))
        speak(message)
        return

    if command.name == "start_live":
        _, message = start_live_loop()
        speak(message)
        return

    if command.name == "stop_live":
        _, message = stop_live_loop()
        speak(message)
        return

    if command.name == "restart_live":
        _, message = restart_live_loop()
        speak(message)
        return

    if command.name == "preview_on":
        _, message = set_preview_enabled(True)
        render_preview_once()
        speak(message)
        return

    if command.name == "preview_off":
        _, message = set_preview_enabled(False)
        close_preview_windows()
        speak(message)
        return

    if command.name == "preview_once":
        success, message = render_preview_once(force=True)
        speak(message if not success else "Rendered the current preview frame once.")
        return

    if command.name == "use_camera":
        _, message = switch_camera(int(command.argument or 0))
        speak(message)
        return


def handle_robot_command(command: str) -> str | None:
    if ROBOT_MANAGER.get_mode() == "cyberwave":
        return handle_cyberwave_command(command)

    parsed_command = parse_command(command)
    if parsed_command is None:
        speak(
            "I can handle move, place, pick, home, scene, camera, bridge, hardware, cyberwave, capabilities, backend, mode, execute, preview, recalibrate, and debug commands."
        )
        return None

    try:
        if parsed_command.action == "home":
            scene_snapshot = fetch_scene_snapshot(wait_timeout=0.5)
            plan = ROBOT_MANAGER.build_plan(
                action="home",
                target_object=None,
                destination_zone=None,
                scene_snapshot=scene_snapshot,
                confirmation_required=False,
            )
            validation = ROBOT_MANAGER.validate_plan(plan, scene_snapshot)
            if scene_snapshot is not None:
                print_scene_report(scene_snapshot)
            print_parsed_command(parsed_command)
            print_execution_plan(plan)
            print_execution_mode()
            print_validation_report(validation)
            result = ROBOT_MANAGER.execute_plan(plan, validation)
            if ROBOT_MANAGER.is_execution_enabled():
                print_execution_result(result)
            return None

        scene_snapshot = fetch_scene_snapshot(wait_timeout=1.0)
        if scene_snapshot is None:
            speak("The live loop has not produced a scene yet. Try 'camera status' or 'scene' in a moment.")
            return None

        scene_objects = scene_snapshot["objects"]

        target_object = resolve_object(parsed_command.raw_target_phrase, scene_objects)
        if target_object is None:
            print_scene_report(scene_snapshot)
            print_parsed_command(parsed_command)
            print_resolved_target(None)
            return None

        destination = None
        confirmation_required = False
        if parsed_command.action == "move":
            destination = resolve_destination(parsed_command.raw_destination_phrase)
            if destination is None:
                print_scene_report(scene_snapshot)
                print_parsed_command(parsed_command)
                print_resolved_target(target_object)
                print_resolved_destination(None)
                return None
            confirmation_required = normalize(str(target_object["zone"])) == destination
            if confirmation_required and not confirm_same_zone(target_object, destination):
                print_scene_report(scene_snapshot)
                print_parsed_command(parsed_command)
                print_resolved_target(target_object)
                print_resolved_destination(destination)
                return None

        plan = ROBOT_MANAGER.build_plan(
            action=parsed_command.action,
            target_object=target_object,
            destination_zone=destination,
            scene_snapshot=scene_snapshot,
            confirmation_required=confirmation_required,
        )
        validation = ROBOT_MANAGER.validate_plan(plan, scene_snapshot)
        print_scene_report(scene_snapshot)
        print_parsed_command(parsed_command)
        print_resolved_target(target_object)
        if parsed_command.action == "move":
            print_resolved_destination(destination)
        print_execution_plan(plan)
        print_execution_mode()
        print_validation_report(validation)
        result = ROBOT_MANAGER.execute_plan(plan, validation)
        if ROBOT_MANAGER.is_execution_enabled():
            print_execution_result(result)
        return None
    except CommandRedirect as redirect:
        speak("Switching to the new command.")
        return redirect.command_text


def handle_cyberwave_command(command: str) -> str | None:
    if normalize(command) in HOME_PHRASES:
        plan = ROBOT_MANAGER.build_plan(
            action="home",
            target_object=None,
            destination_zone=None,
            scene_snapshot=ROBOT_MANAGER.get_sim_scene_snapshot(),
            confirmation_required=False,
        )
        validation = ROBOT_MANAGER.validate_plan(plan, ROBOT_MANAGER.get_sim_scene_snapshot())
        print_sim_scene()
        print_sim_request(command)
        print_execution_plan(plan)
        print_execution_mode()
        print_validation_report(validation)
        result = ROBOT_MANAGER.execute_plan(plan, validation)
        if ROBOT_MANAGER.is_execution_enabled():
            print_execution_result(result)
            if result.status == "success":
                print_simulation_result_summary(plan)
        return None

    clarification_context: str | None = None
    try:
        while True:
            planner_response, plan, validation = ROBOT_MANAGER.plan_cyberwave_command(
                command,
                clarification_context=clarification_context,
            )
            print_sim_scene()
            print_sim_request(command)
            print_planner_response(planner_response)

            if not planner_response.success:
                speak(planner_response.error or "Simulation planning failed.")
                return None

            if planner_response.needs_clarification:
                state = FollowUpState(
                    kind="cyberwave_clarification",
                    prompt=planner_response.clarification_question
                    or "Which object or zone do you mean?",
                    retry_prompt="Please answer the clarification question, or enter a new command.",
                )
                response = prompt_user(state)
                maybe_redirect(response)
                clarification_context = response
                continue

            if plan is None or validation is None:
                speak("The simulation planner did not return an executable plan.")
                return None

            print_execution_plan(plan)
            print_execution_mode()
            print_validation_report(validation)
            result = ROBOT_MANAGER.execute_plan(plan, validation)
            if ROBOT_MANAGER.is_execution_enabled():
                print_execution_result(result)
                if result.status == "success":
                    print_simulation_result_summary(plan)
            return None
    except CommandRedirect as redirect:
        speak("Switching to the new command.")
        return redirect.command_text


def main() -> None:
    _, startup_message = start_live_loop()
    speak("Voice tabletop robot ready. Type a command or 'help' to begin.")
    if startup_message and get_debug_enabled():
        print(f"\nLive vision:\n- {startup_message}\n")

    queued_command: str | None = None

    try:
        while True:
            command = queued_command if queued_command is not None else wait_for_user_input()
            queued_command = None

            builtin_command = parse_builtin_command(command)
            if builtin_command is not None and builtin_command.name == "exit":
                speak("Shutting down.")
                break
            if builtin_command is not None:
                handle_builtin_command(builtin_command)
                continue

            queued_command = handle_robot_command(command)
    finally:
        shutdown_camera()
        close_preview_windows()


if __name__ == "__main__":
    main()
