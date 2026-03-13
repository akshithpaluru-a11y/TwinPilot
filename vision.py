"""Mock vision module for a tabletop robot."""

from __future__ import annotations


def detect_objects() -> list[dict[str, str]]:
    """Return a static scene description for local testing."""
    return [
        {"name": "black bottle", "category": "bottle", "location": "center mat"},
        {"name": "silver bottle", "category": "bottle", "location": "left tray"},
        {"name": "green cup", "category": "cup", "location": "front shelf"},
        {"name": "blue cup", "category": "cup", "location": "back shelf"},
    ]


def detect_destinations() -> list[dict[str, list[str] | str]]:
    """Return a static set of valid placement destinations."""
    return [
        {"name": "left zone", "aliases": ["left zone", "left"]},
        {"name": "left side", "aliases": ["left side", "left"]},
        {"name": "right zone", "aliases": ["right zone", "right"]},
        {"name": "right side", "aliases": ["right side", "right"]},
        {"name": "center mat", "aliases": ["center", "center mat"]},
    ]
