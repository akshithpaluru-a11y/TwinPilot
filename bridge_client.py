"""Small client for talking to the local SO100 bridge service."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEBUG_DIR = Path(__file__).resolve().parent / "debug"
BRIDGE_CONFIG_PATH = CONFIG_DIR / "so100_bridge_config.json"
BRIDGE_SCRIPT_PATH = Path(__file__).resolve().parent / "so100_bridge.py"
BRIDGE_LOG_PATH = DEBUG_DIR / "so100_bridge.log"
DEFAULT_BRIDGE_CONFIG = {
    "host": "127.0.0.1",
    "port": 8765,
    "backend_type": "lerobot_so100",
    "dry_run_default": True,
    "startup_mode": "manual",
    "request_timeout_seconds": 3.0,
}


@dataclass
class BridgeConfig:
    """Small local config for the SO100 bridge service."""

    host: str = "127.0.0.1"
    port: int = 8765
    backend_type: str = "lerobot_so100"
    dry_run_default: bool = True
    startup_mode: str = "manual"
    request_timeout_seconds: float = 3.0
    config_source: str = "defaults"
    config_path: str = str(BRIDGE_CONFIG_PATH)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "backend_type": self.backend_type,
            "dry_run_default": self.dry_run_default,
            "startup_mode": self.startup_mode,
            "request_timeout_seconds": self.request_timeout_seconds,
            "config_source": self.config_source,
            "config_path": self.config_path,
        }


def load_bridge_config() -> BridgeConfig:
    """Load bridge config safely, falling back to defaults."""

    values = dict(DEFAULT_BRIDGE_CONFIG)
    config_source = "defaults"

    if BRIDGE_CONFIG_PATH.exists():
        try:
            loaded = json.loads(BRIDGE_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            for key in DEFAULT_BRIDGE_CONFIG:
                if key in loaded:
                    values[key] = loaded[key]
            config_source = str(BRIDGE_CONFIG_PATH)

    return BridgeConfig(
        host=str(values["host"]),
        port=int(values["port"]),
        backend_type=str(values["backend_type"]),
        dry_run_default=bool(values["dry_run_default"]),
        startup_mode=str(values["startup_mode"]),
        request_timeout_seconds=float(values["request_timeout_seconds"]),
        config_source=config_source,
        config_path=str(BRIDGE_CONFIG_PATH),
    )


class BridgeClient:
    """HTTP client and lightweight subprocess manager for the SO100 bridge."""

    def __init__(self, config: BridgeConfig | None = None) -> None:
        self.config = config or load_bridge_config()
        self._process: subprocess.Popen[bytes] | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, data=data, method=method, headers=headers)

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout or self.config.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8")
                parsed = json.loads(body)
            except (OSError, json.JSONDecodeError):
                parsed = {
                    "success": False,
                    "message": f"Bridge HTTP error: {error.code}",
                    "error": str(error),
                }
            if "success" not in parsed:
                parsed["success"] = False
            return parsed
        except OSError as error:
            return {
                "success": False,
                "message": "Bridge request failed.",
                "error": str(error),
                "bridge_running": False,
            }

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": "Bridge returned invalid JSON.",
                "error": body,
            }
        if "success" not in parsed:
            parsed["success"] = True
        return parsed

    def ping(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/capabilities")

    def backend_info(self) -> dict[str, Any]:
        return self._request("GET", "/backend-info")

    def hardware_ready(self) -> dict[str, Any]:
        return self._request("GET", "/hardware-ready")

    def connect(self) -> dict[str, Any]:
        return self._request("POST", "/connect", {})

    def disconnect(self) -> dict[str, Any]:
        return self._request("POST", "/disconnect", {})

    def set_dry_run(self, enabled: bool) -> dict[str, Any]:
        return self._request("POST", "/dryrun", {"enabled": enabled})

    def send_primitive(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/primitive", payload)

    def send_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/plan", payload)

    def abort(self) -> dict[str, Any]:
        return self._request("POST", "/abort", {})

    def shutdown(self) -> dict[str, Any]:
        return self._request("POST", "/shutdown", {})

    def start_bridge(self) -> dict[str, Any]:
        ping_response = self.ping()
        if ping_response.get("success"):
            ping_response["message"] = "Bridge is already running."
            return ping_response

        if not BRIDGE_SCRIPT_PATH.exists():
            return {
                "success": False,
                "message": "Bridge script is missing.",
                "error": str(BRIDGE_SCRIPT_PATH),
            }

        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = BRIDGE_LOG_PATH.open("ab")
        command = [
            sys.executable,
            str(BRIDGE_SCRIPT_PATH),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--backend",
            self.config.backend_type,
            "--dry-run",
            "on" if self.config.dry_run_default else "off",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(BRIDGE_SCRIPT_PATH.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        for _ in range(20):
            time.sleep(0.25)
            ping_response = self.ping()
            if ping_response.get("success"):
                ping_response["message"] = "Bridge started successfully."
                return ping_response

        return {
            "success": False,
            "message": "Bridge process started but did not become ready in time.",
            "error": str(BRIDGE_LOG_PATH),
        }

    def stop_bridge(self) -> dict[str, Any]:
        response = self.shutdown()
        if self._process is not None:
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            finally:
                self._process = None
        return response


_BRIDGE_CLIENT = BridgeClient()


def get_bridge_client() -> BridgeClient:
    return _BRIDGE_CLIENT
