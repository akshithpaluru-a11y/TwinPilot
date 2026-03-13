"""Stub robot-control actions."""

from __future__ import annotations


def pick(object_name: str, location: str) -> str:
    return f"Pick {object_name} from {location}."


def place(object_name: str, destination: str) -> str:
    return f"Place {object_name} at {destination}."


def home() -> str:
    return "Return robot arm to home position."
