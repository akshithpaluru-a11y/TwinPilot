"""Stub voice interface backed by standard input and output."""

from __future__ import annotations


def speak(message: str) -> None:
    print(f"Robot: {message}")


def listen(prompt: str = "You: ") -> str:
    return input(prompt).strip()
