"""Minimal voice-controlled tabletop robot demo."""

from __future__ import annotations

from robot_control import home, pick, place
from vision import detect_destinations, detect_objects
from voice import listen, speak

EXIT_COMMANDS = {"exit", "quit"}
PICK_PREFIXES = ("pick up ", "pick ", "grab ", "take ", "move ")
TRANSFER_SEPARATORS = (" and place ", " and put ", " then place ", " then put ")
DESTINATION_MARKERS = (" to ", " onto ", " on ", " into ", " in ", " at ")
LEADING_FILLERS = {"the", "a", "an", "it"}


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def strip_leading_fillers(text: str) -> str:
    words = normalize(text).split()
    while words and words[0] in LEADING_FILLERS:
        words.pop(0)
    return " ".join(words)


def split_on_destination_marker(text: str) -> tuple[str, str]:
    padded = f" {normalize(text)} "
    marker_index = -1
    marker_used = ""

    for marker in DESTINATION_MARKERS:
        index = padded.find(marker)
        if index != -1 and (marker_index == -1 or index < marker_index):
            marker_index = index
            marker_used = marker

    if marker_index == -1:
        return normalize(text), ""

    before = normalize(padded[:marker_index])
    after = normalize(padded[marker_index + len(marker_used) :])
    return before, after


def extract_object_query(command: str) -> str:
    lowered = normalize(command)
    remainder = ""
    prefix_used = ""

    for prefix in PICK_PREFIXES:
        if lowered.startswith(prefix):
            remainder = lowered[len(prefix) :]
            prefix_used = prefix
            break

    if not remainder:
        return ""

    for separator in TRANSFER_SEPARATORS:
        if separator in remainder:
            remainder = remainder.split(separator, 1)[0]
            return strip_leading_fillers(remainder)

    if prefix_used == "move ":
        remainder, _ = split_on_destination_marker(remainder)

    return strip_leading_fillers(remainder)


def extract_destination_query(command: str) -> str:
    lowered = normalize(command)

    if lowered.startswith("move "):
        _, destination = split_on_destination_marker(lowered[len("move ") :])
        return strip_leading_fillers(destination)

    remainder = ""
    for separator in TRANSFER_SEPARATORS:
        if separator in lowered:
            remainder = lowered.split(separator, 1)[1]
            break

    if not remainder:
        return ""

    remainder = strip_leading_fillers(remainder)
    _, destination = split_on_destination_marker(remainder)
    if destination:
        return strip_leading_fillers(destination)

    return strip_leading_fillers(remainder)


def is_transfer_command(command: str) -> bool:
    lowered = normalize(command)
    return lowered.startswith("move ") or any(separator in lowered for separator in TRANSFER_SEPARATORS)


def match_objects(query: str, detections: list[dict[str, str]]) -> list[dict[str, str]]:
    if not query:
        return []

    query = normalize(query)
    query_words = query.split()
    matches: list[dict[str, str]] = []

    for detected in detections:
        haystack = f"{detected['name']} {detected['category']}".lower()
        if query == detected["name"].lower() or query == detected["category"].lower():
            matches.append(detected)
            continue
        if all(word in haystack for word in query_words):
            matches.append(detected)

    return matches


def match_destinations(query: str, destinations: list[dict[str, list[str] | str]]) -> list[dict[str, list[str] | str]]:
    if not query:
        return []

    query = normalize(query)
    query_words = query.split()
    matches: list[dict[str, list[str] | str]] = []

    for destination in destinations:
        names = [destination["name"], *destination["aliases"]]
        normalized_names = [normalize(name) for name in names]
        if query in normalized_names:
            matches.append(destination)
            continue
        if any(all(word in name for word in query_words) for name in normalized_names):
            matches.append(destination)

    return matches


def resolve_object(query: str, detections: list[dict[str, str]]) -> dict[str, str] | None:
    matches = match_objects(query, detections)
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        options = ", ".join(item["name"] for item in matches)
        speak(f"I found multiple objects: {options}. Which one do you mean?")
    else:
        visible = ", ".join(f"{item['name']} on the {item['location']}" for item in detections)
        speak(f"Which object do you mean? I can see {visible}.")

    clarification = normalize(listen())
    matches = match_objects(clarification, detections)
    if len(matches) == 1:
        return matches[0]

    speak("I still could not determine the object.")
    return None


def resolve_destination(
    query: str, destinations: list[dict[str, list[str] | str]]
) -> str | None:
    matches = match_destinations(query, destinations)
    if len(matches) == 1:
        return str(matches[0]["name"])

    if len(matches) > 1:
        options = ", ".join(str(item["name"]) for item in matches)
        speak(f"I found multiple destinations: {options}. Which one do you mean?")
    else:
        options = ", ".join(str(item["name"]) for item in destinations)
        speak(f"Where should I place it? Available destinations: {options}.")

    clarification = normalize(listen())
    matches = match_destinations(clarification, destinations)
    if len(matches) == 1:
        return str(matches[0]["name"])

    speak("I still could not determine the destination.")
    return None


def build_action_plan(command: str) -> list[str]:
    detections = detect_objects()
    destinations = detect_destinations()
    lowered = normalize(command)

    if lowered == "home":
        return [home()]

    if not any(lowered.startswith(prefix) for prefix in PICK_PREFIXES):
        speak("I can handle move, pick, and home commands. What should I do?")
        return build_action_plan(listen())

    object_query = extract_object_query(lowered)
    target = resolve_object(object_query, detections)
    if target is None:
        return []

    action_plan = [pick(target["name"], target["location"])]

    if is_transfer_command(lowered):
        destination_query = extract_destination_query(lowered)
        destination_name = resolve_destination(destination_query, destinations)
        if destination_name is None:
            return []
        action_plan.append(place(target["name"], destination_name))

    action_plan.append(home())
    return action_plan


def main() -> None:
    speak("Voice tabletop robot ready. Type a command or 'exit' to stop.")

    while True:
        command = listen()
        if normalize(command) in EXIT_COMMANDS:
            speak("Shutting down.")
            break

        action_plan = build_action_plan(command)
        if action_plan:
            print("\nFinal action plan:")
            for step_number, step in enumerate(action_plan, start=1):
                print(f"{step_number}. {step}")
            print()


if __name__ == "__main__":
    main()
