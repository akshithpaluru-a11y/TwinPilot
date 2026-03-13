# Voice-Controlled Tabletop Robot Agent

Minimal local Python project for a tabletop robot demo with mock voice, vision, and robot control modules.

## Files

- `main.py` runs the command loop.
- `vision.py` returns mock detected objects, locations, and placement destinations.
- `voice.py` provides `speak()` and `listen()` stubs.
- `robot_control.py` provides `pick()`, `place()`, and `home()` stubs.

## Run

```bash
python3 main.py
```

## Example commands

- `move the black bottle to the right side`
- `move the cup to the left zone`
- `pick up the bottle and place it on the right`

If the object or destination is ambiguous, the app asks a clarification question and then prints the final action plan.
