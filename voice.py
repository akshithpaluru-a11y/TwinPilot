"""Stub voice interface backed by standard input and output."""

from __future__ import annotations

import select
import sys

_prompt_visible = False


def _dedupe_accidental_double_entry(text: str) -> str:
    stripped = text.strip()
    if stripped and len(stripped) % 2 == 0:
        midpoint = len(stripped) // 2
        if stripped[:midpoint] == stripped[midpoint:]:
            return stripped[:midpoint]
    return stripped


def speak(message: str) -> None:
    print(f"Robot: {message}")


def poll_listen(prompt: str = "You: ", timeout: float = 0.1) -> str | None:
    """Poll stdin without blocking forever so the main thread can keep pumping preview UI."""
    global _prompt_visible

    if not _prompt_visible:
        print(prompt, end="", flush=True)
        _prompt_visible = True

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        ready = [sys.stdin]

    if not ready:
        return None

    line = sys.stdin.readline()
    _prompt_visible = False
    if line == "":
        return "exit"
    return _dedupe_accidental_double_entry(line)


def listen(prompt: str = "You: ") -> str:
    while True:
        response = poll_listen(prompt=prompt, timeout=0.1)
        if response is not None:
            return response
