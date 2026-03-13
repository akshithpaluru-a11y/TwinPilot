"""OpenCV-based live tabletop detection with ROI calibration and fallback mode."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment.
    cv2 = None

PREVIEW_WINDOW = "Robot Table View - Workspace Detection"
CALIBRATION_WINDOW = "Robot Table Calibration"
PREVIEW_SIZE = (560, 340)
CAMERA_PROBE_INDICES = range(6)
PROCESS_INTERVAL_SECONDS = 0.25
FRAME_HISTORY = 4
LIVE_CACHE_GRACE_SECONDS = 2.0
STALE_SCENE_SECONDS = 3.0

MIN_CONTOUR_AREA = 1400
MAX_CONTOUR_AREA_RATIO = 0.18
MAX_BBOX_AREA_RATIO = 0.24
MIN_SOLIDITY = 0.45
MIN_EXTENT = 0.18
MIN_CENTER_Y_RATIO = 0.15
DEFAULT_WORKSPACE_ROI = {"x": 0.18, "y": 0.42, "w": 0.64, "h": 0.50}

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
DEBUG_DIR = PROJECT_DIR / "debug"
WORKSPACE_CONFIG_PATH = CONFIG_DIR / "table_config.json"


@dataclass
class SceneObject:
    """Structured tabletop object description used by the planner."""

    id: str
    name: str
    label: str
    color: str
    position: str
    zone: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "color": self.color,
            "position": self.position,
            "zone": self.zone,
            "confidence": round(float(self.confidence), 2),
            "bbox": list(self.bbox),
            "area": round(float(self.area), 2),
            "source": self.source,
        }


@dataclass
class SceneMetadata:
    """Runtime metadata describing the current live vision state."""

    detection_mode: str
    fallback_reason: str
    roi_source: str
    camera_index: int
    camera_open: bool
    live_loop_running: bool
    last_update_time: float | None
    last_live_detection_time: float | None
    frames_processed: int
    raw_candidates_count: int
    valid_detections_count: int
    rejection_summary: dict[str, int]
    workspace_roi: tuple[int, int, int, int] | None
    preview_enabled: bool
    preview_available: bool
    preview_error: str
    using_cached_scene: bool = False

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        current_time = time.time() if now is None else now
        last_update_age = (
            round(current_time - self.last_update_time, 2)
            if self.last_update_time is not None
            else None
        )
        last_live_age = (
            round(current_time - self.last_live_detection_time, 2)
            if self.last_live_detection_time is not None
            else None
        )
        scene_stale = bool(
            last_update_age is not None and last_update_age > STALE_SCENE_SECONDS
        )
        return {
            "detection_mode": self.detection_mode,
            "fallback_reason": self.fallback_reason,
            "roi_source": self.roi_source,
            "camera_index": self.camera_index,
            "camera_open": self.camera_open,
            "live_loop_running": self.live_loop_running,
            "last_update_time": self.last_update_time,
            "last_live_detection_time": self.last_live_detection_time,
            "last_update_age_seconds": last_update_age,
            "last_live_detection_age_seconds": last_live_age,
            "frames_processed": self.frames_processed,
            "raw_candidates_count": self.raw_candidates_count,
            "valid_detections_count": self.valid_detections_count,
            "rejection_summary": dict(self.rejection_summary),
            "workspace_roi": list(self.workspace_roi) if self.workspace_roi is not None else None,
            "preview_enabled": self.preview_enabled,
            "preview_available": self.preview_available,
            "preview_error": self.preview_error,
            "using_cached_scene": self.using_cached_scene,
            "scene_stale": scene_stale,
        }


@dataclass
class SceneState:
    """Latest planner-facing tabletop scene snapshot."""

    objects: list[SceneObject]
    source: str
    fallback_used: bool
    summary: str
    metadata: SceneMetadata
    raw_frame: Any | None = None
    roi_frame: Any | None = None
    contour_frame: Any | None = None
    annotated_frame: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [object_.to_dict() for object_ in self.objects],
            "source": self.source,
            "fallback_used": self.fallback_used,
            "summary": self.summary,
            "workspace_roi": (
                list(self.metadata.workspace_roi)
                if self.metadata.workspace_roi is not None
                else None
            ),
            "roi_source": self.metadata.roi_source,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class DetectionPassResult:
    """Per-frame detector output before temporal stabilization."""

    objects: list[SceneObject]
    workspace_roi: tuple[int, int, int, int] | None
    roi_source: str
    raw_candidates_count: int
    rejection_summary: dict[str, int]
    roi_frame: Any | None
    contour_frame: Any | None


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_debug_dir() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _strip_fillers(text: str) -> str:
    normalized = normalize_text(text)
    while normalized.startswith(("the ", "a ", "an ", "my ", "this ", "that ")):
        normalized = normalized.split(" ", 1)[1]
    return normalized


def canonicalize_object_query(query: str) -> str:
    normalized = _strip_fillers(query)
    if not normalized:
        return "object"
    if any(term in normalized for term in ("black bottle", "dark object", "dark bottle")):
        return "black bottle"
    if "water bottle" in normalized or normalized == "bottle":
        return "bottle"
    if "mug" in normalized or "cup" in normalized:
        return "cup"
    if normalized in {"object", "item", "thing", "something"}:
        return "object"
    return normalized


def scene_object_aliases(scene_object: dict[str, Any]) -> set[str]:
    name = normalize_text(scene_object["name"])
    label = normalize_text(scene_object["label"])
    color = normalize_text(scene_object["color"])
    zone = normalize_text(scene_object["zone"])

    aliases = {
        name,
        label,
        color,
        zone,
        f"{color} {label}".strip(),
        f"{zone} {label}".strip(),
        f"{zone} {name}".strip(),
    }

    if label == "bottle":
        aliases.update({"bottle", "water bottle"})
    if label == "cup":
        aliases.update({"cup", "mug"})
    if color in {"black", "dark"} or name in {"black bottle", "dark object"}:
        aliases.update({"black bottle", "dark bottle", "dark object"})
    if label == "object":
        aliases.update({"object", "item", "thing"})

    return {alias for alias in aliases if alias}


def filter_scene_objects(objects: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Filter scene objects using tabletop-friendly label and synonym matching."""
    if not objects:
        return []

    normalized_query = canonicalize_object_query(query)
    if normalized_query == "object":
        return objects

    if normalized_query == "black bottle":
        dark_matches = [
            object_
            for object_ in objects
            if normalize_text(object_["color"]) in {"black", "dark"}
            or normalize_text(object_["name"]) in {"black bottle", "dark object"}
        ]
        bottle_matches = [
            object_
            for object_ in dark_matches
            if normalize_text(object_["label"]) == "bottle"
        ]
        return bottle_matches or dark_matches

    if normalized_query == "bottle":
        bottle_matches = [
            object_
            for object_ in objects
            if normalize_text(object_["label"]) == "bottle"
        ]
        if bottle_matches:
            return bottle_matches
        return [
            object_
            for object_ in objects
            if normalize_text(object_["name"]) in {"black bottle", "dark object"}
        ]

    if normalized_query == "cup":
        return [
            object_
            for object_ in objects
            if normalize_text(object_["label"]) == "cup"
        ]

    exact_matches = [
        object_
        for object_ in objects
        if normalized_query in scene_object_aliases(object_)
    ]
    if exact_matches:
        return exact_matches

    query_words = normalized_query.split()
    return [
        object_
        for object_ in objects
        if all(word in " ".join(sorted(scene_object_aliases(object_))) for word in query_words)
    ]


def _validate_normalized_roi(roi: Any) -> bool:
    if not isinstance(roi, dict):
        return False

    try:
        x = float(roi["x"])
        y = float(roi["y"])
        width = float(roi["w"])
        height = float(roi["h"])
    except (KeyError, TypeError, ValueError):
        return False

    if width <= 0 or height <= 0:
        return False
    if x < 0 or y < 0 or x + width > 1 or y + height > 1:
        return False
    return True


def _load_saved_normalized_roi() -> dict[str, float] | None:
    if not WORKSPACE_CONFIG_PATH.exists():
        return None

    try:
        data = json.loads(WORKSPACE_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    roi = data.get("workspace_roi")
    if not _validate_normalized_roi(roi):
        return None
    return {
        "x": float(roi["x"]),
        "y": float(roi["y"]),
        "w": float(roi["w"]),
        "h": float(roi["h"]),
    }


def _default_normalized_roi() -> dict[str, float]:
    return dict(DEFAULT_WORKSPACE_ROI)


def _normalized_to_absolute(
    roi: dict[str, float], frame_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    x = int(round(roi["x"] * frame_width))
    y = int(round(roi["y"] * frame_height))
    width = int(round(roi["w"] * frame_width))
    height = int(round(roi["h"] * frame_height))

    x = max(0, min(x, frame_width - 2))
    y = max(0, min(y, frame_height - 2))
    width = max(40, min(width, frame_width - x))
    height = max(40, min(height, frame_height - y))
    return (x, y, width, height)


def _absolute_to_normalized(
    roi: tuple[int, int, int, int], frame_shape: tuple[int, ...]
) -> dict[str, float]:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = roi
    return {
        "x": round(x / max(frame_width, 1), 4),
        "y": round(y / max(frame_height, 1), 4),
        "w": round(width / max(frame_width, 1), 4),
        "h": round(height / max(frame_height, 1), 4),
    }


def get_workspace_roi(frame_shape: tuple[int, ...]) -> tuple[tuple[int, int, int, int], str]:
    saved_roi = _load_saved_normalized_roi()
    if saved_roi is not None:
        return _normalized_to_absolute(saved_roi, frame_shape), "calibrated"
    return _normalized_to_absolute(_default_normalized_roi(), frame_shape), "default"


def _mock_scene_objects(
    frame_shape: tuple[int, ...] | None = None,
    workspace_roi: tuple[int, int, int, int] | None = None,
) -> list[SceneObject]:
    if frame_shape is None:
        frame_shape = (480, 640, 3)
    if workspace_roi is None:
        workspace_roi = _normalized_to_absolute(_default_normalized_roi(), frame_shape)

    workspace_x, workspace_y, workspace_width, workspace_height = workspace_roi
    bottle_width = max(34, int(workspace_width * 0.10))
    bottle_height = max(88, int(workspace_height * 0.42))
    cup_width = max(44, int(workspace_width * 0.12))
    cup_height = max(58, int(workspace_height * 0.26))

    left_bbox = (
        workspace_x + int(workspace_width * 0.08),
        workspace_y + int(workspace_height * 0.42),
        bottle_width,
        bottle_height,
    )
    center_bbox = (
        workspace_x + int(workspace_width * 0.44),
        workspace_y + int(workspace_height * 0.36),
        bottle_width,
        bottle_height,
    )
    right_bbox = (
        workspace_x + int(workspace_width * 0.76),
        workspace_y + int(workspace_height * 0.50),
        cup_width,
        cup_height,
    )

    return [
        SceneObject(
            id="obj_1",
            name="silver bottle",
            label="bottle",
            color="silver",
            position="left",
            zone="left",
            confidence=0.45,
            bbox=left_bbox,
            area=float(left_bbox[2] * left_bbox[3]),
            source="fallback",
        ),
        SceneObject(
            id="obj_2",
            name="black bottle",
            label="bottle",
            color="black",
            position="center",
            zone="center",
            confidence=0.58,
            bbox=center_bbox,
            area=float(center_bbox[2] * center_bbox[3]),
            source="fallback",
        ),
        SceneObject(
            id="obj_3",
            name="blue cup",
            label="cup",
            color="blue",
            position="right",
            zone="right",
            confidence=0.52,
            bbox=right_bbox,
            area=float(right_bbox[2] * right_bbox[3]),
            source="fallback",
        ),
    ]


@contextmanager
def _suppress_native_stderr():
    stderr_copy = None
    devnull = None

    try:
        stderr_copy = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    except OSError:
        yield
    finally:
        if stderr_copy is not None:
            os.dup2(stderr_copy, 2)
            os.close(stderr_copy)
        if devnull is not None:
            os.close(devnull)


def _open_camera_handle(index: int):
    if cv2 is None:
        return None

    backend_candidates = [None]
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backend_candidates.append(cv2.CAP_AVFOUNDATION)

    for backend in backend_candidates:
        with _suppress_native_stderr():
            if backend is None:
                camera = cv2.VideoCapture(index)
            else:
                camera = cv2.VideoCapture(index, backend)

        if camera is None or not camera.isOpened():
            if camera is not None:
                camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        readable = False
        for _ in range(3):
            with _suppress_native_stderr():
                success, frame = camera.read()
            if success and frame is not None and frame.size:
                readable = True
                break
            time.sleep(0.05)

        if readable:
            return camera

        camera.release()

    return None


def _capture_single_frame(camera_index: int):
    camera = _open_camera_handle(camera_index)
    if camera is None:
        return None
    with _suppress_native_stderr():
        success, frame = camera.read()
    camera.release()
    if not success:
        return None
    return frame


def _mask_for_dark_object(hsv_frame):
    blurred = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, (0, 0, 0), (180, 255, 75))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _mask_for_color(hsv_frame, lower, upper):
    blurred = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _zone_from_center(x_center: int, workspace_width: int) -> str:
    if x_center < workspace_width / 3:
        return "left"
    if x_center < (2 * workspace_width) / 3:
        return "center"
    return "right"


def _contour_solidity(contour) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return cv2.contourArea(contour) / hull_area


def _contour_extent(contour, width: int, height: int) -> float:
    bbox_area = width * height
    if bbox_area <= 0:
        return 0.0
    return cv2.contourArea(contour) / bbox_area


def _find_candidate_contours(mask, limit: int = 4):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    return valid_contours[:limit]


def _evaluate_contour(
    contour,
    workspace_shape: tuple[int, int],
    label: str,
) -> tuple[bool, str, dict[str, float]]:
    workspace_height, workspace_width = workspace_shape[:2]
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    bbox_area = width * height
    aspect_ratio = height / max(width, 1)
    area_ratio = area / max(workspace_height * workspace_width, 1)
    bbox_ratio = bbox_area / max(workspace_height * workspace_width, 1)
    center_y_ratio = (y + height / 2) / max(workspace_height, 1)
    top_y_ratio = y / max(workspace_height, 1)
    solidity = _contour_solidity(contour)
    extent = _contour_extent(contour, width, height)

    metrics = {
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "area": float(area),
        "bbox_area": float(bbox_area),
        "aspect_ratio": float(aspect_ratio),
        "area_ratio": float(area_ratio),
        "bbox_ratio": float(bbox_ratio),
        "center_y_ratio": float(center_y_ratio),
        "top_y_ratio": float(top_y_ratio),
        "solidity": float(solidity),
        "extent": float(extent),
    }

    if (
        area_ratio > MAX_CONTOUR_AREA_RATIO
        or bbox_ratio > MAX_BBOX_AREA_RATIO
        or width > workspace_width * 0.72
        or height > workspace_height * 0.88
    ):
        return False, "too_large", metrics
    if top_y_ratio < 0.02 or center_y_ratio < MIN_CENTER_Y_RATIO:
        return False, "too_high", metrics
    if solidity < MIN_SOLIDITY:
        return False, "low_solidity", metrics
    if extent < MIN_EXTENT:
        return False, "low_extent", metrics

    if label == "bottle" and not (0.85 <= aspect_ratio <= 4.8):
        return False, "bad_aspect", metrics
    if label == "cup" and not (0.55 <= aspect_ratio <= 2.7):
        return False, "bad_aspect", metrics
    if label == "object" and not (0.60 <= aspect_ratio <= 3.8):
        return False, "bad_aspect", metrics

    return True, "accepted", metrics


def _confidence_from_metrics(
    area: float,
    workspace_area: float,
    solidity: float,
    center_y_ratio: float,
    bonus: float = 0.0,
) -> float:
    size_score = min(area / max(workspace_area, 1.0), 0.14) * 1.8
    lower_bonus = max(0.0, center_y_ratio - 0.35) * 0.18
    confidence = 0.46 + size_score + lower_bonus + min(solidity, 1.0) * 0.12 + bonus
    return round(min(0.98, confidence), 2)


def _build_scene_object(
    contour,
    *,
    name: str,
    label: str,
    color: str,
    workspace_roi: tuple[int, int, int, int],
    workspace_shape: tuple[int, int],
    source: str,
    bonus: float = 0.0,
) -> SceneObject:
    workspace_x, workspace_y, _, _ = workspace_roi
    workspace_height, workspace_width = workspace_shape[:2]
    x, y, width, height = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    solidity = _contour_solidity(contour)
    center_y_ratio = (y + height / 2) / max(workspace_height, 1)
    zone = _zone_from_center(x + width // 2, workspace_width)
    confidence = _confidence_from_metrics(
        area,
        float(workspace_height * workspace_width),
        solidity,
        center_y_ratio,
        bonus=bonus,
    )
    return SceneObject(
        id="",
        name=name,
        label=label,
        color=color,
        position=zone,
        zone=zone,
        confidence=confidence,
        bbox=(workspace_x + x, workspace_y + y, width, height),
        area=round(area, 2),
        source=source,
    )


def _iou(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _dedupe_scene_objects(objects: list[SceneObject]) -> list[SceneObject]:
    kept: list[SceneObject] = []
    for object_ in sorted(objects, key=lambda item: (item.confidence, item.area), reverse=True):
        overlaps = [
            existing
            for existing in kept
            if _iou(existing.bbox, object_.bbox) > 0.45
            and (existing.label == object_.label or existing.color == object_.color)
        ]
        if not overlaps:
            kept.append(object_)
    kept.sort(key=lambda item: (item.zone, item.bbox[0], item.name))
    return kept


def _assign_object_ids(objects: list[SceneObject]) -> list[SceneObject]:
    ordered = sorted(objects, key=lambda object_: (object_.zone, object_.bbox[0], object_.name))
    assigned: list[SceneObject] = []
    for index, object_ in enumerate(ordered, start=1):
        assigned.append(
            SceneObject(
                id=f"obj_{index}",
                name=object_.name,
                label=object_.label,
                color=object_.color,
                position=object_.zone,
                zone=object_.zone,
                confidence=round(object_.confidence, 2),
                bbox=object_.bbox,
                area=object_.area,
                source=object_.source,
            )
        )
    return assigned


def _merge_scene_objects(objects: list[SceneObject]) -> SceneObject:
    count = len(objects)
    avg_bbox = tuple(
        int(round(sum(object_.bbox[index] for object_ in objects) / count))
        for index in range(4)
    )
    avg_area = round(sum(object_.area for object_ in objects) / count, 2)
    avg_confidence = round(sum(object_.confidence for object_ in objects) / count, 2)
    zone = Counter(object_.zone for object_ in objects).most_common(1)[0][0]
    exemplar = objects[0]
    return SceneObject(
        id="",
        name=Counter(object_.name for object_ in objects).most_common(1)[0][0],
        label=Counter(object_.label for object_ in objects).most_common(1)[0][0],
        color=Counter(object_.color for object_ in objects).most_common(1)[0][0],
        position=zone,
        zone=zone,
        confidence=min(0.99, avg_confidence + min(0.08, 0.02 * (count - 1))),
        bbox=avg_bbox,
        area=avg_area,
        source=exemplar.source,
    )


def _stabilize_object_sets(object_sets: list[list[SceneObject]]) -> list[SceneObject]:
    if not object_sets:
        return []

    minimum_appearances = 2 if len(object_sets) > 1 else 1
    grouped: dict[tuple[str, str, str], list[SceneObject]] = {}
    for objects in object_sets:
        for object_ in objects:
            key = (object_.name, object_.label, object_.color)
            grouped.setdefault(key, []).append(object_)

    stable_objects = [
        _merge_scene_objects(group)
        for group in grouped.values()
        if len(group) >= minimum_appearances
    ]
    if stable_objects:
        return _assign_object_ids(_dedupe_scene_objects(stable_objects))

    best_frame = max(
        object_sets,
        key=lambda objects: (len(objects), sum(object_.confidence for object_ in objects)),
        default=[],
    )
    return _assign_object_ids(_dedupe_scene_objects(best_frame))


def _draw_zone_guides(frame) -> None:
    height, width = frame.shape[:2]
    for divider in (width // 3, (2 * width) // 3):
        cv2.line(frame, (divider, 0), (divider, height), (90, 90, 90), 1)

    labels = [("left", 14), ("center", width // 2 - 32), ("right", width - 72)]
    for label, x_pos in labels:
        cv2.putText(
            frame,
            label,
            (x_pos, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def _build_roi_debug_frame(workspace_frame, objects: list[SceneObject]):
    roi_frame = workspace_frame.copy()
    _draw_zone_guides(roi_frame)
    workspace_x = 0
    workspace_y = 0
    for object_ in objects:
        box_x, box_y, box_width, box_height = object_.bbox
        rel_x = box_x - workspace_x
        rel_y = box_y - workspace_y
        cv2.rectangle(
            roi_frame,
            (rel_x, rel_y),
            (rel_x + box_width, rel_y + box_height),
            (0, 220, 120),
            2,
        )
        cv2.putText(
            roi_frame,
            f"{object_.name} | {object_.zone}",
            (rel_x, max(rel_y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 220, 120),
            2,
        )
    return roi_frame


def _build_contour_debug_frame(workspace_frame, contour_entries: list[dict[str, Any]]):
    debug_frame = workspace_frame.copy()
    _draw_zone_guides(debug_frame)
    for entry in contour_entries:
        contour = entry["contour"]
        color = (0, 220, 120) if entry["accepted"] else (0, 90, 255)
        x, y, width, height = cv2.boundingRect(contour)
        cv2.drawContours(debug_frame, [contour], -1, color, 2)
        cv2.rectangle(debug_frame, (x, y), (x + width, y + height), color, 1)
        text = entry["text"]
        cv2.putText(
            debug_frame,
            text,
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )
    return debug_frame


def _detect_tabletop_objects(frame) -> DetectionPassResult:
    workspace_roi, roi_source = get_workspace_roi(frame.shape)
    workspace_x, workspace_y, workspace_width, workspace_height = workspace_roi
    workspace_frame = frame[
        workspace_y : workspace_y + workspace_height,
        workspace_x : workspace_x + workspace_width,
    ]
    if workspace_frame.size == 0:
        return DetectionPassResult([], workspace_roi, roi_source, 0, {}, None, None)

    hsv = cv2.cvtColor(workspace_frame, cv2.COLOR_BGR2HSV)
    workspace_shape = workspace_frame.shape[:2]

    detected_objects: list[SceneObject] = []
    contour_entries: list[dict[str, Any]] = []
    rejection_summary: Counter[str] = Counter()
    raw_candidates_count = 0

    dark_contours = _find_candidate_contours(_mask_for_dark_object(hsv), limit=3)
    for contour in dark_contours:
        raw_candidates_count += 1
        _, _, width, height = cv2.boundingRect(contour)
        dark_label = "bottle" if height / max(width, 1) >= 1.1 else "object"
        accepted, reason, _ = _evaluate_contour(contour, workspace_shape, dark_label)
        if not accepted:
            rejection_summary[reason] += 1
            contour_entries.append(
                {
                    "contour": contour,
                    "accepted": False,
                    "text": f"dark {dark_label} | {reason}",
                }
            )
            continue
        name = "black bottle" if dark_label == "bottle" else "dark object"
        detected_objects.append(
            _build_scene_object(
                contour,
                name=name,
                label=dark_label,
                color="black",
                workspace_roi=workspace_roi,
                workspace_shape=workspace_shape,
                source="live",
                bonus=0.08 if dark_label == "bottle" else 0.02,
            )
        )
        contour_entries.append(
            {
                "contour": contour,
                "accepted": True,
                "text": name,
            }
        )

    color_specs = [
        ("green cup", "cup", "green", (35, 50, 40), (90, 255, 255)),
        ("blue cup", "cup", "blue", (90, 70, 40), (130, 255, 255)),
    ]
    for name, label, color, lower, upper in color_specs:
        contours = _find_candidate_contours(_mask_for_color(hsv, lower, upper), limit=2)
        for contour in contours:
            raw_candidates_count += 1
            accepted, reason, _ = _evaluate_contour(contour, workspace_shape, label)
            if not accepted:
                rejection_summary[reason] += 1
                contour_entries.append(
                    {
                        "contour": contour,
                        "accepted": False,
                        "text": f"{color} {label} | {reason}",
                    }
                )
                continue
            detected_objects.append(
                _build_scene_object(
                    contour,
                    name=name,
                    label=label,
                    color=color,
                    workspace_roi=workspace_roi,
                    workspace_shape=workspace_shape,
                    source="live",
                    bonus=0.05,
                )
            )
            contour_entries.append(
                {
                    "contour": contour,
                    "accepted": True,
                    "text": name,
                }
            )

    assigned_objects = _assign_object_ids(_dedupe_scene_objects(detected_objects))
    roi_objects = [
        SceneObject(
            id=object_.id,
            name=object_.name,
            label=object_.label,
            color=object_.color,
            position=object_.position,
            zone=object_.zone,
            confidence=object_.confidence,
            bbox=(
                object_.bbox[0] - workspace_x,
                object_.bbox[1] - workspace_y,
                object_.bbox[2],
                object_.bbox[3],
            ),
            area=object_.area,
            source=object_.source,
        )
        for object_ in assigned_objects
    ]
    roi_frame = _build_roi_debug_frame(workspace_frame, roi_objects)
    contour_frame = _build_contour_debug_frame(workspace_frame, contour_entries)
    return DetectionPassResult(
        objects=assigned_objects,
        workspace_roi=workspace_roi,
        roi_source=roi_source,
        raw_candidates_count=raw_candidates_count,
        rejection_summary=dict(rejection_summary),
        roi_frame=roi_frame,
        contour_frame=contour_frame,
    )


def _dim_non_workspace(frame, workspace_roi: tuple[int, int, int, int]):
    shaded = cv2.addWeighted(frame, 0.35, frame * 0, 0.65, 0)
    x, y, width, height = workspace_roi
    shaded[y : y + height, x : x + width] = frame[y : y + height, x : x + width]
    return shaded


def _annotate_workspace_view(
    frame,
    objects: list[SceneObject],
    metadata: SceneMetadata,
):
    workspace_roi = metadata.workspace_roi
    if workspace_roi is None:
        annotated = frame.copy()
    else:
        annotated = _dim_non_workspace(frame, workspace_roi)
        x, y, width, height = workspace_roi
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (50, 180, 240), 2)
        for divider in (x + width // 3, x + (2 * width) // 3):
            cv2.line(annotated, (divider, y), (divider, y + height), (90, 90, 90), 1)

        zone_labels = [
            ("left", x + 16),
            ("center", x + width // 2 - 38),
            ("right", x + width - 76),
        ]
        for label, x_pos in zone_labels:
            cv2.putText(
                annotated,
                label,
                (x_pos, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    for object_ in objects:
        box_x, box_y, box_width, box_height = object_.bbox
        cv2.rectangle(
            annotated,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 220, 120),
            2,
        )
        cv2.putText(
            annotated,
            f"{object_.name} | {object_.zone} | {object_.confidence:.2f}",
            (box_x, max(box_y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 220, 120),
            2,
        )

    last_live_age = (
        0.0
        if metadata.last_live_detection_time is None
        else max(0.0, time.time() - metadata.last_live_detection_time)
    )
    mode_labels = {
        "live": "LIVE DETECTION",
        "live_cache": "LIVE CACHE",
        "fallback": "FALLBACK MOCK",
        "waiting": "WAITING FOR CAMERA",
    }
    cv2.putText(
        annotated,
        mode_labels.get(metadata.detection_mode, metadata.detection_mode.upper()),
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"camera {metadata.camera_index} | roi {metadata.roi_source}",
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"valid {metadata.valid_detections_count} | last live {last_live_age:.1f}s",
        (14, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
    if metadata.fallback_reason:
        cv2.putText(
            annotated,
            metadata.fallback_reason[:72],
            (14, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )
    if _SERVICE.get_debug_enabled():
        cv2.putText(
            annotated,
            (
                f"frames={metadata.frames_processed} raw={metadata.raw_candidates_count} "
                f"rejects={sum(metadata.rejection_summary.values())}"
            ),
            (14, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
    return annotated


def summarize_scene(objects: list[dict[str, Any]]) -> str:
    """Return a concise human-readable scene summary for the terminal UI."""
    if not objects:
        return "- no tabletop objects detected"

    zone_order = {"left": 0, "center": 1, "right": 2}
    ordered = sorted(
        objects,
        key=lambda object_: (zone_order.get(normalize_text(object_["zone"]), 99), object_["name"]),
    )
    return "\n".join(
        f"- {object_['name']} at {object_['zone']} (confidence {float(object_['confidence']):.2f})"
        for object_ in ordered
    )


class LiveVisionService:
    """Background OpenCV worker that keeps a current tabletop scene cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._camera = None
        self._camera_index = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._first_update_event = threading.Event()
        self._debug_enabled = False
        self._preview_enabled = True
        self._preview_available = True
        self._preview_error = ""
        self._state = self._build_waiting_state()

    def _preview_tuple_locked(self) -> tuple[bool, bool, str]:
        return (
            self._preview_enabled,
            self._preview_available,
            self._preview_error,
        )

    def _apply_preview_metadata_locked(self, state: SceneState) -> None:
        state.metadata.preview_enabled = self._preview_enabled
        state.metadata.preview_available = self._preview_available
        state.metadata.preview_error = self._preview_error

    def _build_waiting_state(self) -> SceneState:
        metadata = SceneMetadata(
            detection_mode="waiting",
            fallback_reason="",
            roi_source="default",
            camera_index=self._camera_index,
            camera_open=False,
            live_loop_running=False,
            last_update_time=None,
            last_live_detection_time=None,
            frames_processed=0,
            raw_candidates_count=0,
            valid_detections_count=0,
            rejection_summary={},
            workspace_roi=None,
            preview_enabled=self._preview_enabled,
            preview_available=self._preview_available,
            preview_error=self._preview_error,
        )
        return SceneState(
            objects=[],
            source="waiting",
            fallback_used=False,
            summary="- no live scene available yet",
            metadata=metadata,
        )

    def _copy_state(self) -> SceneState:
        with self._lock:
            return SceneState(
                objects=[
                    SceneObject(
                        id=object_.id,
                        name=object_.name,
                        label=object_.label,
                        color=object_.color,
                        position=object_.position,
                        zone=object_.zone,
                        confidence=object_.confidence,
                        bbox=tuple(object_.bbox),
                        area=object_.area,
                        source=object_.source,
                    )
                    for object_ in self._state.objects
                ],
                source=self._state.source,
                fallback_used=self._state.fallback_used,
                summary=self._state.summary,
                metadata=SceneMetadata(
                    detection_mode=self._state.metadata.detection_mode,
                    fallback_reason=self._state.metadata.fallback_reason,
                    roi_source=self._state.metadata.roi_source,
                    camera_index=self._state.metadata.camera_index,
                    camera_open=self._state.metadata.camera_open,
                    live_loop_running=self._state.metadata.live_loop_running,
                    last_update_time=self._state.metadata.last_update_time,
                    last_live_detection_time=self._state.metadata.last_live_detection_time,
                    frames_processed=self._state.metadata.frames_processed,
                    raw_candidates_count=self._state.metadata.raw_candidates_count,
                    valid_detections_count=self._state.metadata.valid_detections_count,
                    rejection_summary=dict(self._state.metadata.rejection_summary),
                    workspace_roi=(
                        tuple(self._state.metadata.workspace_roi)
                        if self._state.metadata.workspace_roi is not None
                        else None
                    ),
                    preview_enabled=self._state.metadata.preview_enabled,
                    preview_available=self._state.metadata.preview_available,
                    preview_error=self._state.metadata.preview_error,
                    using_cached_scene=self._state.metadata.using_cached_scene,
                ),
                raw_frame=None if self._state.raw_frame is None else self._state.raw_frame.copy(),
                roi_frame=None if self._state.roi_frame is None else self._state.roi_frame.copy(),
                contour_frame=(
                    None if self._state.contour_frame is None else self._state.contour_frame.copy()
                ),
                annotated_frame=(
                    None
                    if self._state.annotated_frame is None
                    else self._state.annotated_frame.copy()
                ),
            )

    def _update_state(self, state: SceneState) -> None:
        with self._lock:
            self._apply_preview_metadata_locked(state)
            self._state = state

    def get_debug_enabled(self) -> bool:
        return self._debug_enabled

    def set_debug_enabled(self, enabled: bool) -> None:
        self._debug_enabled = enabled

    def get_active_camera_index(self) -> int:
        with self._lock:
            return self._camera_index

    def _mark_stopped(self) -> None:
        state = self._copy_state()
        state.metadata.live_loop_running = False
        state.metadata.camera_open = False
        self._update_state(state)

    def get_preview_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "preview_enabled": self._preview_enabled,
                "preview_available": self._preview_available,
                "preview_error": self._preview_error,
            }

    def set_preview_enabled(self, enabled: bool) -> tuple[bool, str]:
        with self._lock:
            self._preview_enabled = enabled
            if enabled:
                self._preview_available = True
                self._preview_error = ""
            self._apply_preview_metadata_locked(self._state)
        if enabled:
            return True, "Preview is enabled. The window will refresh on the main thread."
        return True, "Preview is disabled."

    def mark_preview_error(self, error_message: str) -> None:
        with self._lock:
            self._preview_available = False
            self._preview_error = error_message
            self._apply_preview_metadata_locked(self._state)

    def clear_preview_error(self) -> None:
        with self._lock:
            self._preview_available = True
            self._preview_error = ""
            self._apply_preview_metadata_locked(self._state)

    def _build_fallback_state(
        self,
        *,
        reason: str,
        frame=None,
        workspace_roi: tuple[int, int, int, int] | None = None,
        roi_source: str = "default",
        frames_processed: int = 0,
        raw_candidates_count: int = 0,
        rejection_summary: dict[str, int] | None = None,
        live_loop_running: bool,
        camera_open: bool,
    ) -> SceneState:
        current_time = time.time()
        preview_enabled, preview_available, preview_error = self._preview_tuple_locked()
        fallback_objects = _mock_scene_objects(
            frame.shape if frame is not None else None,
            workspace_roi=workspace_roi,
        )
        metadata = SceneMetadata(
            detection_mode="fallback",
            fallback_reason=reason,
            roi_source=roi_source,
            camera_index=self.get_active_camera_index(),
            camera_open=camera_open,
            live_loop_running=live_loop_running,
            last_update_time=current_time,
            last_live_detection_time=None,
            frames_processed=frames_processed,
            raw_candidates_count=raw_candidates_count,
            valid_detections_count=0,
            rejection_summary=dict(rejection_summary or {}),
            workspace_roi=workspace_roi,
            preview_enabled=preview_enabled,
            preview_available=preview_available,
            preview_error=preview_error,
        )
        annotated = None
        if frame is not None and workspace_roi is not None:
            annotated = _annotate_workspace_view(frame, fallback_objects, metadata)
        return SceneState(
            objects=fallback_objects,
            source="fallback",
            fallback_used=True,
            summary=summarize_scene([object_.to_dict() for object_ in fallback_objects]),
            metadata=metadata,
            raw_frame=None if frame is None else frame.copy(),
            annotated_frame=None if annotated is None else annotated.copy(),
        )

    def _run_live_loop(self) -> None:
        history: deque[list[SceneObject]] = deque(maxlen=FRAME_HISTORY)
        last_live_objects: list[SceneObject] = []
        last_live_detection_time: float | None = None
        frames_processed = 0

        while not self._stop_event.is_set():
            loop_started = time.time()
            with self._lock:
                camera = self._camera
                camera_index = self._camera_index

            if camera is None:
                break

            with _suppress_native_stderr():
                success, frame = camera.read()

            if not success or frame is None or not getattr(frame, "size", 0):
                state = self._build_fallback_state(
                    reason=f"Camera {camera_index} is open but frame reads are failing.",
                    frame=None,
                    workspace_roi=None,
                    roi_source="default",
                    frames_processed=frames_processed,
                    live_loop_running=True,
                    camera_open=False,
                )
                self._update_state(state)
                self._first_update_event.set()
                time.sleep(PROCESS_INTERVAL_SECONDS)
                continue

            frames_processed += 1
            detection = _detect_tabletop_objects(frame)
            history.append(detection.objects)
            stable_objects = _stabilize_object_sets(list(history))
            using_cached_scene = False
            fallback_used = False
            fallback_reason = ""
            output_objects = stable_objects
            detection_mode = "live"

            if stable_objects:
                last_live_objects = stable_objects
                last_live_detection_time = time.time()
            elif (
                last_live_objects
                and last_live_detection_time is not None
                and (time.time() - last_live_detection_time) <= LIVE_CACHE_GRACE_SECONDS
            ):
                output_objects = last_live_objects
                detection_mode = "live_cache"
                using_cached_scene = True
                fallback_reason = "Using the most recent stable live scene while detections settle."
            else:
                output_objects = _mock_scene_objects(frame.shape, detection.workspace_roi)
                detection_mode = "fallback"
                fallback_used = True
                if detection.raw_candidates_count == 0:
                    fallback_reason = "No valid tabletop detections inside the workspace ROI."
                else:
                    fallback_reason = "Workspace contours were rejected by tabletop filters."

            preview_enabled, preview_available, preview_error = self._preview_tuple_locked()
            metadata = SceneMetadata(
                detection_mode=detection_mode,
                fallback_reason=fallback_reason,
                roi_source=detection.roi_source,
                camera_index=camera_index,
                camera_open=True,
                live_loop_running=True,
                last_update_time=time.time(),
                last_live_detection_time=last_live_detection_time,
                frames_processed=frames_processed,
                raw_candidates_count=detection.raw_candidates_count,
                valid_detections_count=len(stable_objects),
                rejection_summary=dict(detection.rejection_summary),
                workspace_roi=detection.workspace_roi,
                preview_enabled=preview_enabled,
                preview_available=preview_available,
                preview_error=preview_error,
                using_cached_scene=using_cached_scene,
            )
            annotated_frame = _annotate_workspace_view(frame, output_objects, metadata)
            state = SceneState(
                objects=output_objects,
                source="fallback" if fallback_used else "live",
                fallback_used=fallback_used,
                summary=summarize_scene([object_.to_dict() for object_ in output_objects]),
                metadata=metadata,
                raw_frame=frame.copy(),
                roi_frame=None if detection.roi_frame is None else detection.roi_frame.copy(),
                contour_frame=(
                    None if detection.contour_frame is None else detection.contour_frame.copy()
                ),
                annotated_frame=annotated_frame.copy(),
            )
            self._update_state(state)
            self._first_update_event.set()

            remaining = PROCESS_INTERVAL_SECONDS - (time.time() - loop_started)
            if remaining > 0:
                time.sleep(remaining)

        with self._lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
            self._thread = None
        self._mark_stopped()

    def start_live_loop(self, camera_index: int | None = None, *, _camera_override=None) -> tuple[bool, str]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if camera_index is None or camera_index == self._camera_index:
                    return True, f"Live analysis is already running on camera {self._camera_index}."
                return False, "Stop or switch the current live loop before changing cameras."

            if camera_index is not None:
                self._camera_index = camera_index

        if cv2 is None:
            state = self._build_fallback_state(
                reason="OpenCV is not installed, so live camera detection is unavailable.",
                live_loop_running=False,
                camera_open=False,
            )
            self._update_state(state)
            return False, "OpenCV is not installed, so live camera detection is unavailable."

        camera = (
            _camera_override
            if _camera_override is not None
            else _open_camera_handle(self.get_active_camera_index())
        )
        if camera is None:
            state = self._build_fallback_state(
                reason=f"Could not open camera {self.get_active_camera_index()}. Using fallback mock scene.",
                live_loop_running=False,
                camera_open=False,
            )
            self._update_state(state)
            return False, f"Could not open camera {self.get_active_camera_index()}. Using fallback mock scene."

        with self._lock:
            self._camera = camera
            self._stop_event = threading.Event()
            self._first_update_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_live_loop,
                name="robot-table-live-vision",
                daemon=True,
            )
            self._thread.start()

        self._first_update_event.wait(timeout=1.2)
        metadata = self.get_current_metadata()
        if metadata["last_update_time"] is None:
            return True, f"Live analysis started on camera {self.get_active_camera_index()} and is warming up."
        return True, f"Live analysis started on camera {self.get_active_camera_index()}."

    def stop_live_loop(self) -> tuple[bool, str]:
        with self._lock:
            thread = self._thread
            stop_event = self._stop_event

        if thread is None or not thread.is_alive():
            self._mark_stopped()
            return True, "Live analysis is already stopped."

        stop_event.set()
        thread.join(timeout=2.0)
        self._mark_stopped()
        return True, "Live analysis stopped."

    def restart_live_loop(self) -> tuple[bool, str]:
        self.stop_live_loop()
        return self.start_live_loop()

    def switch_camera(self, camera_index: int) -> tuple[bool, str]:
        current_index = self.get_active_camera_index()
        metadata = self.get_current_metadata()
        if camera_index == current_index and metadata["live_loop_running"] and metadata["camera_open"]:
            return True, f"Already using camera {camera_index}."

        new_camera = _open_camera_handle(camera_index)
        if new_camera is None:
            return (
                False,
                f"Could not open camera {camera_index}. Keeping camera {current_index}.",
            )

        was_running = metadata["live_loop_running"]
        self.stop_live_loop()
        with self._lock:
            self._camera_index = camera_index

        started, message = self.start_live_loop(_camera_override=new_camera)
        if started:
            return True, f"Switched to camera {camera_index}. {message}"

        if was_running and camera_index != current_index:
            old_camera = _open_camera_handle(current_index)
            with self._lock:
                self._camera_index = current_index
            if old_camera is not None:
                self.start_live_loop(_camera_override=old_camera)
        return False, message

    def list_available_cameras(self) -> list[dict[str, Any]]:
        metadata = self.get_current_metadata()
        active_index = self.get_active_camera_index()
        results: list[dict[str, Any]] = []

        for index in CAMERA_PROBE_INDICES:
            active = index == active_index
            if active and metadata["camera_open"]:
                results.append(
                    {
                        "index": index,
                        "available": True,
                        "active": True,
                        "status": "active",
                    }
                )
                continue

            camera = _open_camera_handle(index)
            available = camera is not None
            if camera is not None:
                camera.release()
            results.append(
                {
                    "index": index,
                    "available": available,
                    "active": active,
                    "status": "available" if available else "unavailable",
                }
            )

        return results

    def get_current_scene(self, *, wait_for_update: bool = False, timeout: float = 0.0) -> dict[str, Any]:
        if wait_for_update:
            self._first_update_event.wait(timeout=timeout)
        return self._copy_state().to_dict()

    def get_current_metadata(self) -> dict[str, Any]:
        state = self._copy_state()
        return state.metadata.to_dict()

    def get_current_annotated_frame(self):
        return self._copy_state().annotated_frame

    def get_current_raw_frame(self):
        return self._copy_state().raw_frame

    def summarize_current_scene(self) -> str:
        return self.get_current_scene().get("summary", "- no live scene available yet")

    def summarize_current_metadata(self) -> str:
        metadata = self.get_current_metadata()
        lines = [
            f"- active camera index: {metadata['camera_index']}",
            f"- camera open: {'yes' if metadata['camera_open'] else 'no'}",
            f"- live loop running: {'yes' if metadata['live_loop_running'] else 'no'}",
            f"- detection mode: {metadata['detection_mode']}",
            f"- roi source: {metadata['roi_source']}",
        ]
        if metadata["workspace_roi"] is not None:
            lines.append(f"- workspace roi: {tuple(metadata['workspace_roi'])}")
        if metadata["last_update_age_seconds"] is None:
            lines.append("- last update age: no scene yet")
        else:
            lines.append(f"- last update age: {metadata['last_update_age_seconds']:.2f}s")
        if metadata["last_live_detection_age_seconds"] is not None:
            lines.append(
                f"- last live detection age: {metadata['last_live_detection_age_seconds']:.2f}s"
            )
        lines.append(f"- frames processed: {metadata['frames_processed']}")
        lines.append(f"- raw candidates: {metadata['raw_candidates_count']}")
        lines.append(f"- valid detections: {metadata['valid_detections_count']}")
        lines.append(f"- preview enabled: {'yes' if metadata['preview_enabled'] else 'no'}")
        lines.append(f"- preview available: {'yes' if metadata['preview_available'] else 'no'}")
        if metadata["preview_error"]:
            lines.append(f"- preview error: {metadata['preview_error']}")
        if metadata["using_cached_scene"]:
            lines.append("- note: using the recent live scene cache")
        if metadata["scene_stale"]:
            lines.append("- warning: current scene cache is stale")
        if metadata["fallback_reason"]:
            lines.append(f"- fallback reason: {metadata['fallback_reason']}")
        if metadata["rejection_summary"]:
            rejection_text = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(metadata["rejection_summary"].items())
            )
            lines.append(f"- rejections: {rejection_text}")
        return "\n".join(lines)

    def summarize_roi_status(self) -> str:
        saved_roi = _load_saved_normalized_roi()
        current_metadata = self.get_current_metadata()
        lines = [
            f"- roi source: {current_metadata['roi_source']}",
            f"- config path: {WORKSPACE_CONFIG_PATH.relative_to(PROJECT_DIR)}",
        ]
        if saved_roi is not None:
            lines.append(f"- saved normalized roi: {saved_roi}")
        else:
            lines.append(f"- default normalized roi: {_default_normalized_roi()}")
        if current_metadata["workspace_roi"] is not None:
            lines.append(f"- current absolute roi: {tuple(current_metadata['workspace_roi'])}")
        else:
            lines.append("- current absolute roi: unavailable")
        return "\n".join(lines)

    def save_debug_snapshot(self) -> tuple[bool, str]:
        if cv2 is None:
            return False, "OpenCV is not installed, so debug image saving is unavailable."

        state = self._copy_state()
        if state.raw_frame is None:
            return False, "There is no current live frame to save."

        _ensure_debug_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        target_dir = DEBUG_DIR / f"snapshot-{timestamp}"
        target_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(target_dir / "raw.png"), state.raw_frame)
        if state.roi_frame is not None:
            cv2.imwrite(str(target_dir / "roi.png"), state.roi_frame)
        if state.contour_frame is not None:
            cv2.imwrite(str(target_dir / "contours.png"), state.contour_frame)
        if state.annotated_frame is not None:
            cv2.imwrite(str(target_dir / "annotated.png"), state.annotated_frame)

        metadata_payload = state.to_dict()
        metadata_payload["saved_at"] = timestamp
        target_dir.joinpath("metadata.json").write_text(json.dumps(metadata_payload, indent=2))
        return True, f"Saved debug snapshot to {target_dir.relative_to(PROJECT_DIR)}."

    def shutdown(self) -> None:
        self.stop_live_loop()


_SERVICE = LiveVisionService()


def _main_thread_only(task_name: str) -> tuple[bool, str]:
    # macOS OpenCV HighGUI uses AppKit and must stay on the main thread.
    if threading.current_thread() is not threading.main_thread():
        return False, f"{task_name} must run on the main thread."
    return True, ""


def _save_workspace_roi(normalized_roi: dict[str, float]) -> None:
    _ensure_config_dir()
    config_payload = {
        "version": 1,
        "workspace_roi": normalized_roi,
    }
    WORKSPACE_CONFIG_PATH.write_text(json.dumps(config_payload, indent=2))


def _render_preview_frame(frame, *, wait_ms: int = 1) -> tuple[bool, str]:
    ok, message = _main_thread_only("Preview rendering")
    if not ok:
        return False, message
    if cv2 is None:
        return False, "OpenCV is not installed, so preview rendering is unavailable."
    if frame is None or not getattr(frame, "size", 0):
        return False, "There is no frame available to preview."

    try:
        preview = cv2.resize(frame, PREVIEW_SIZE)
        cv2.imshow(PREVIEW_WINDOW, preview)
        cv2.waitKey(wait_ms)
        _SERVICE.clear_preview_error()
        return True, "Preview rendered."
    except cv2.error as exc:
        error_message = f"Preview rendering failed: {exc}"
        _SERVICE.mark_preview_error(error_message)
        return False, error_message


def set_debug_enabled(enabled: bool) -> None:
    _SERVICE.set_debug_enabled(enabled)


def get_debug_enabled() -> bool:
    return _SERVICE.get_debug_enabled()


def start_live_loop(camera_index: int | None = None) -> tuple[bool, str]:
    return _SERVICE.start_live_loop(camera_index)


def stop_live_loop() -> tuple[bool, str]:
    return _SERVICE.stop_live_loop()


def restart_live_loop() -> tuple[bool, str]:
    return _SERVICE.restart_live_loop()


def list_available_cameras() -> list[dict[str, Any]]:
    return _SERVICE.list_available_cameras()


def switch_camera(camera_index: int) -> tuple[bool, str]:
    return _SERVICE.switch_camera(camera_index)


def get_active_camera_index() -> int:
    return _SERVICE.get_active_camera_index()


def get_current_scene(*, wait_for_update: bool = False, timeout: float = 0.0) -> dict[str, Any]:
    return _SERVICE.get_current_scene(wait_for_update=wait_for_update, timeout=timeout)


def get_current_metadata() -> dict[str, Any]:
    return _SERVICE.get_current_metadata()


def get_current_annotated_frame():
    return _SERVICE.get_current_annotated_frame()


def get_current_raw_frame():
    return _SERVICE.get_current_raw_frame()


def get_preview_status() -> dict[str, Any]:
    return _SERVICE.get_preview_status()


def set_preview_enabled(enabled: bool) -> tuple[bool, str]:
    return _SERVICE.set_preview_enabled(enabled)


def summarize_current_scene() -> str:
    return _SERVICE.summarize_current_scene()


def summarize_current_metadata() -> str:
    return _SERVICE.summarize_current_metadata()


def summarize_roi_status() -> str:
    return _SERVICE.summarize_roi_status()


def save_debug_snapshot() -> tuple[bool, str]:
    return _SERVICE.save_debug_snapshot()


def close_preview_windows() -> tuple[bool, str]:
    ok, message = _main_thread_only("Preview window cleanup")
    if not ok:
        return False, message
    if cv2 is None:
        return False, "OpenCV is not installed, so preview windows are unavailable."

    try:
        for window_name in (PREVIEW_WINDOW, CALIBRATION_WINDOW):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                continue
        cv2.destroyAllWindows()
        _SERVICE.clear_preview_error()
        return True, "Preview windows closed."
    except cv2.error as exc:
        error_message = f"Could not close preview windows cleanly: {exc}"
        _SERVICE.mark_preview_error(error_message)
        return False, error_message


def pump_preview_events() -> tuple[bool, str]:
    ok, message = _main_thread_only("Preview pumping")
    if not ok:
        return False, message

    preview_status = _SERVICE.get_preview_status()
    if not preview_status["preview_enabled"]:
        return True, "Preview is disabled."
    if not preview_status["preview_available"] and preview_status["preview_error"]:
        return False, preview_status["preview_error"]

    frame = _SERVICE.get_current_annotated_frame()
    if frame is None:
        frame = _SERVICE.get_current_raw_frame()
    if frame is None:
        return False, "There is no current frame to preview."

    return _render_preview_frame(frame, wait_ms=1)


def render_preview_once(*, force: bool = False) -> tuple[bool, str]:
    preview_status = _SERVICE.get_preview_status()
    if not force and not preview_status["preview_enabled"]:
        return True, "Preview is disabled."

    frame = _SERVICE.get_current_annotated_frame()
    if frame is None:
        frame = _SERVICE.get_current_raw_frame()
    if frame is None:
        return False, "There is no current frame available for preview."

    return _render_preview_frame(frame, wait_ms=1)


def run_roi_calibration(frame) -> tuple[bool, str]:
    ok, message = _main_thread_only("Workspace calibration")
    if not ok:
        return False, message
    if cv2 is None:
        return False, "OpenCV is not installed, so calibration is unavailable."
    if frame is None or not getattr(frame, "size", 0):
        return False, "Could not open workspace calibration because no valid frame is available."

    close_preview_windows()

    roi, _ = get_workspace_roi(frame.shape)
    overlay = frame.copy()
    x, y, width, height = roi
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 220, 120), 2)
    cv2.putText(
        overlay,
        "Drag a rectangle around the tabletop, then press ENTER or SPACE to save.",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        "Press C to cancel.",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    try:
        selected = cv2.selectROI(
            CALIBRATION_WINDOW,
            overlay,
            fromCenter=False,
            showCrosshair=True,
        )
    except cv2.error as exc:
        error_message = f"Calibration UI failed: {exc}"
        _SERVICE.mark_preview_error(error_message)
        return False, "Calibration requires a GUI-enabled OpenCV environment on the main thread."
    finally:
        try:
            cv2.destroyWindow(CALIBRATION_WINDOW)
        except cv2.error:
            pass

    if selected[2] < 40 or selected[3] < 40:
        return False, "Calibration cancelled."

    normalized = _absolute_to_normalized(tuple(map(int, selected)), frame.shape)
    if not _validate_normalized_roi(normalized):
        return False, "The selected workspace ROI was invalid."

    _save_workspace_roi(normalized)
    _SERVICE.clear_preview_error()
    return True, "Workspace ROI saved to config/table_config.json."


def calibrate_workspace(frame=None) -> tuple[bool, str]:
    if frame is None:
        frame = _SERVICE.get_current_raw_frame()
    return run_roi_calibration(frame)


def capture_scene() -> dict[str, Any]:
    """Compatibility wrapper that returns the current live scene cache."""
    return _SERVICE.get_current_scene(wait_for_update=True, timeout=0.8)


def detect_objects() -> list[dict[str, Any]]:
    """Compatibility wrapper for older code paths."""
    return capture_scene()["objects"]


def detect_destinations() -> list[dict[str, list[str] | str]]:
    """Return canonical tabletop placement zones and common synonyms."""
    return [
        {"name": "left", "aliases": ["left", "left side", "left zone"]},
        {"name": "center", "aliases": ["center", "middle", "middle zone", "center zone"]},
        {"name": "right", "aliases": ["right", "right side", "right zone"]},
    ]


def get_threshold_values() -> dict[str, float]:
    return {
        "min_contour_area": MIN_CONTOUR_AREA,
        "max_contour_area_ratio": MAX_CONTOUR_AREA_RATIO,
        "max_bbox_area_ratio": MAX_BBOX_AREA_RATIO,
        "min_solidity": MIN_SOLIDITY,
        "min_extent": MIN_EXTENT,
        "min_center_y_ratio": MIN_CENTER_Y_RATIO,
        "process_interval_seconds": PROCESS_INTERVAL_SECONDS,
        "frame_history": FRAME_HISTORY,
        "live_cache_grace_seconds": LIVE_CACHE_GRACE_SECONDS,
        "stale_scene_seconds": STALE_SCENE_SECONDS,
    }


def summarize_thresholds() -> str:
    values = get_threshold_values()
    return "\n".join(
        f"- {name}: {value}"
        for name, value in values.items()
    )


def shutdown_camera() -> None:
    """Release the live camera loop without touching GUI state."""
    _SERVICE.shutdown()


atexit.register(shutdown_camera)
