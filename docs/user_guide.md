# User Guide

## Interface Overview

The COMPOSE interface has three sections:

**Left — Workspace arena:** 2D top-down view of the robot workspace powered by PyBullet physics simulation. Objects animate to their goal positions when commands execute. Falls back to matplotlib rendering in environments without display.

**Right — Chat panel:** Type natural language commands. See parsed intent chips, confidence scores, NOVEL tags, and execution results after each command.

**Bottom — Metrics bar:** Live KPIs updating after each command: Task Success Rate, Goal Accuracy, Novel Colour Generalisation, Compositional Generalisation.

---

## Command Format

```
move [colour] [shape] [spatial relation] [colour] [shape]
```

### Supported colours
`red` `blue` `green` `yellow` `purple` `cyan` `orange` `pink`

Novel colours (never in training): `cyan` `orange` `pink`

### Supported shapes
`cube` `cylinder` `sphere` `container`

Aliases: `box`→cube, `ball`→sphere, `bin`→container, `tray`→container

### Supported spatial relations
`right of` `left of` `behind` `in front of` `beside` `above` `below`

### Supported sizes (optional)
`small` `medium` `large` — aliases: `tiny`→small, `big`→large

---

## The 5 Demo Scenarios

Click the Demo buttons or type commands manually:

| Demo | Command | Tests |
|---|---|---|
| Demo 1 | `move red cube right of blue cube` | Baseline — both objects known |
| Demo 2 | `move green cylinder behind yellow container` | Attribute reasoning — shape + spatial |
| Demo 3 | `move cyan sphere left of green cylinder` | **NOVEL colour** — cyan never in training |
| Demo 4 | `move purple container beside red cube` | Novel composition |
| Demo 5 | `move blue cube in front of yellow container` | Multi-constraint spatial |

**Demo 3 is the key demonstration.** Cyan is not in the training colour set. COMPOSE resolves it at 85%+ confidence via compositional embedding (cyan = 0.4×blue + 0.6×green). Standard VLA models fail here with ~20% accuracy.

---

## Adding Objects to the Scene

Click **➕ Add object to scene** to expand the panel. Select:
- Colour (all 8 options including novel colours)
- Shape (cube / cylinder / sphere / container)
- Size (small / medium / large)

Click **Add to scene**. The object appears at a free position in the arena. You can then manipulate it via natural language commands immediately.

Example workflow:
```
1. Add: orange cube, medium
2. Type: move orange cube right of red cube
3. System resolves orange (novel colour) via compositional embedding
```

---

## Ambiguity Handling

If your command is ambiguous (e.g. `move that box`), COMPOSE asks for clarification instead of executing blindly:

```
User: move that box
COMPOSE: Ambiguous — 3 objects match. Which one: RED CUBE, BLUE CUBE, or PURPLE CONTAINER?
```

This is Innovation 3: safety by design. Confidence threshold is 0.7 — below this, the system always asks.

---

## Understanding the Chat Response

Each response shows:

```
Parsed: [action:move] [target:red] [shape:cube] [spatial:right_of] [ref:blue cube] [conf:0.82]
Executing: Moving red cube right of blue cube → (431, 200)
TSR: 100%
```

- **Parsed chips:** what the system extracted from your command
- **conf:** confidence score (0.0-1.0, threshold 0.7)
- **NOVEL tag:** appears when the target colour was never in training
- **TSR:** running task success rate for this session

---

## Running the Benchmark

```bash
cd src
python evaluate.py
```

Runs 6 automated test scenarios and prints KPI summary to stdout. Saves results to `results.csv`.

Expected output:
```
[PASS] Demo 1 — Baseline
[PASS] Demo 2 — Attribute reasoning
[PASS] Demo 3 — Novel colour generalisation [NOVEL]
[PASS] Demo 4 — Novel composition
[PASS] Demo 5 — Multi-constraint
[PASS] Demo 6 — Ambiguity handling

TSR: 100.0%   Goal Accuracy: 100.0%   Novel Colour Gen: 100.0%
Tests passed: 6/6
```
