# Voice-Controlled Tabletop Robot Agent

CyberWave-first simulation demo for a tabletop robot agent. The main app stays local and REPL-driven, but the primary showcase path is now:

1. load a preset simulation scene
2. send a natural-language command
3. plan the action with the OpenAI Responses API
4. execute the plan in a local CyberWave-style scene model
5. optionally sync or extend the scene through CyberWave later

The older live-vision and hardware paths are still in the repo, but they are no longer the main demo story.

## What the app does now

- runs locally with `python3 main.py`
- keeps the existing terminal REPL
- supports three execution modes:
  - `simulation`
  - `cyberwave`
  - `hardware`
- uses a condition library instead of training
- uses an asset registry for demo scene objects
- can call the OpenAI Responses API to turn natural-language commands into structured simulation actions
- executes those actions through a reusable execution-plan system
- can optionally initialize a CyberWave SDK session if config and credentials are present

## Architecture

The project now has these layers:

- REPL / planner app
  - [`main.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/main.py)
  - built-in commands
  - user-facing command flow
- execution layer
  - [`robot_control.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/robot_control.py)
  - structured plans
  - validation
  - local simulation
  - CyberWave controller
  - hardware controller
- OpenAI simulation planner
  - [`openai_planner.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/openai_planner.py)
  - uses the Responses API
  - validates structured JSON output
- CyberWave simulation client
  - [`cyberwave_client.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/cyberwave_client.py)
  - owns local demo scene state
  - owns CyberWave config and optional SDK session
  - is the only place that knows about CyberWave-specific integration details
- condition library
  - [`config/demo_conditions.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/demo_conditions.json)
- asset registry
  - [`config/assets.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/assets.json)

## CyberWave-first demo flow

Recommended showcase flow:

1. `python3 main.py`
2. `mode cyberwave`
3. `cyberwave status`
4. `list conditions`
5. `load condition demo_showcase_scene`
6. `scene sim`
7. `execute on`
8. type `move the black bottle to the right zone`
9. inspect the printed plan and result
10. `replay sim` if you want to replay the last action sequence

If `OPENAI_API_KEY` is missing, planning fails cleanly and the rest of the simulation commands still work.

## OpenAI setup

Set your API key in the environment before running the app:

```bash
export OPENAI_API_KEY="your_key_here"
```

The planner uses the Responses API through [`openai_planner.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/openai_planner.py). It asks the model for valid JSON only, sends the exact active scene objects, and rejects invalid or unsupported plans before execution.

Default model config lives in:

- [`config/cyberwave_config.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/cyberwave_config.json)

## CyberWave setup

CyberWave config lives in:

- [`config/cyberwave_config.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/cyberwave_config.json)

Current fields:

- `enabled`
- `project_id`
- `environment_id`
- `twin_id`
- `scene_id`
- `api_base`
- `default_condition`
- `use_openai_planner`
- `openai_model`
- `notes`

For a real CyberWave-connected environment, also set:

```bash
export CYBERWAVE_API_TOKEN="your_token_here"
```

Current behavior:

- if the CyberWave SDK or token is missing, the app stays in local-only simulation mode
- if config is incomplete, `cyberwave connect` refuses clearly
- local simulation still works without CyberWave connectivity

## Demo conditions

The app uses preset conditions instead of any training pipeline.

Condition library:

- [`config/demo_conditions.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/demo_conditions.json)

Included conditions:

- `bottle_left_cup_right`
- `cup_center_block_left`
- `two_bottles_left_and_right`
- `bottle_near_edge`
- `object_in_drop_zone`
- `cluttered_table_simple`
- `single_black_bottle_center`
- `blue_cup_right`
- `empty_workspace`
- `bottle_and_cube`
- `cube_stack_left`
- `cup_in_center_zone`
- `bottle_in_target_zone_already`
- `two_objects_same_zone`
- `object_blocking_target`
- `pick_and_place_basic`
- `pick_and_place_reverse`
- `reset_home_scene`
- `demo_showcase_scene`
- `ambiguity_test_scene`

To add more conditions:

1. open [`config/demo_conditions.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/demo_conditions.json)
2. add a new condition entry
3. reference assets by `asset_id`
4. set zones like `left`, `center`, or `right`
5. optionally add notes or object metadata

When a condition is loaded, each active object instance gets a stable canonical `object_id` plus a human-readable `display_name`. The planner and execution layer use `object_id` internally so the model cannot invent targets.

Example active object:

```json
{
  "object_id": "obj_black_bottle_1",
  "display_name": "black bottle",
  "label": "bottle",
  "color": "black",
  "zone": "center",
  "aliases": ["bottle", "black bottle", "water bottle", "dark object"]
}
```

## Assets

Asset registry:

- [`config/assets.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/assets.json)

Current assets include:

- black bottle
- blue cup
- red cube
- green block
- gray tray
- left / center / right zone markers
- claw target marker

Each asset entry includes:

- id
- display name
- type
- color
- default size
- aliases
- CyberWave reference info

To add more assets:

1. open [`config/assets.json`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/config/assets.json)
2. add a new asset entry
3. set a stable `id`
4. add useful aliases for natural-language planning
5. optionally add `cyberwave_ref` metadata

## Commands

Core demo commands:

- `showcase`
- `reset showcase`
- `demo script`
- `mode cyberwave`
- `cyberwave status`
- `cyberwave connect`
- `cyberwave disconnect`
- `list assets`
- `list conditions`
- `load condition demo_showcase_scene`
- `scene sim`
- `plan sim`
- `execute sim`
- `reset sim`
- `replay sim`
- `demo`

Existing execution commands still work:

- `mode`
- `mode simulation`
- `mode hardware`
- `execute on`
- `execute off`
- `execute plan`
- `robot status`
- `last plan`
- `home robot`
- `abort`
- `speed`

Vision / legacy commands are still present:

- `scene`
- `scene raw`
- `diagnose`
- `save debug`
- `camera status`
- `cameras`
- `preview on`
- `preview off`
- `recalibrate`

## How CyberWave mode works

`mode cyberwave` switches execution to the CyberWave simulation controller.

That controller:

- uses the local scene state from [`cyberwave_client.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/cyberwave_client.py)
- uses the OpenAI planner if enabled and configured
- maps planner actions into the existing `ExecutionPlan` model
- executes those steps locally
- reports whether CyberWave is actually connected or still local-only

This means the demo is still usable even when CyberWave credentials or SDK support are not available.

## How OpenAI planning works

The OpenAI planner:

- reads the user’s natural-language command
- reads the current simulation scene state
- reads the exact active scene objects and their canonical `object_id` values
- reads available actions and zones
- asks the model to return strict JSON only
- validates action names, targets, object ids, and zones before execution
- deterministically maps a returned display name only when exactly one active object matches
- asks for clarification when the result is ambiguous instead of guessing
- rejects invented object ids or unknown targets cleanly

Allowed action types are:

- `move_above_object`
- `grasp_object`
- `lift_object`
- `move_to_zone`
- `place_object`
- `return_home`

## Real vs mocked

Real today:

- local REPL command flow
- structured execution plans
- local simulation scene state
- asset registry
- condition loading
- OpenAI Responses API planning when `OPENAI_API_KEY` is set
- honest CyberWave SDK/token/config checks

Still mocked or partial:

- full CyberWave scene sync and asset import automation
- browser-side object animation driven by a verified CyberWave API call
- real robot hardware execution for the showcase
- trained perception or object models
- motion planning / IK for physical manipulation

Important honesty rule:

- if CyberWave is not connected, the app says it is local-only
- if OpenAI is unavailable, the planner says so clearly
- if asset sync is not implemented, the app does not pretend assets were imported remotely
- if the planner returns an unknown or ambiguous target, the app blocks execution and explains why

## Testing the demo flow

Recommended CyberWave showcase test:

1. `python3 main.py`
2. `showcase`
3. `move the black bottle to the right zone`
4. `move the cup to the center`
5. `reset showcase`
6. `demo script`

Expected behavior:

- the terminal prints the current simulation scene
- `showcase` switches to CyberWave mode, loads the demo scene, enables execution, and hides the live camera preview
- the planner request prints the raw command and the exact active objects
- the planner result prints `target_object_id`, `target_display_name`, and `destination_zone`
- the execution plan uses the canonical object id internally
- ambiguity causes a clarification prompt instead of a guessed target

## Run locally

```bash
cd /Users/akshith/Documents/Hackathon_Proj/robot-agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```

## Other modes still preserved

The repo still contains:

- OpenCV live-vision code in [`vision.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/vision.py)
- bridge / hardware code in [`bridge_client.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/bridge_client.py) and [`so100_bridge.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/so100_bridge.py)
- the older local simulation and hardware execution paths in [`robot_control.py`](/Users/akshith/Documents/Hackathon_Proj/robot-agent/robot_control.py)

They are preserved for future work, but the main demo story is now CyberWave-first simulation plus OpenAI planning.
# TwinPilot
