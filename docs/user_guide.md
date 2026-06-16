# User Guide

## Interface Overview

The COMPOSE interface has three sections:

- **Left — Workspace arena**: 2D top-down view of the robot workspace. Objects animate to their goal positions when commands execute.
- **Right — Chat panel**: Type natural language commands. See parsed intent, confidence scores, and execution results.
- **Bottom — Metrics bar**: Live KPIs updating after each command.

## Command Format

```
move [colour] [shape] [spatial relation] [colour] [shape]
```

Examples:
```
move red cube right of blue block
move green cylinder behind yellow container
move cyan sphere left of green cylinder     ← novel colour
move purple container beside red cube
move blue block in front of yellow container
```

## Demo Scenarios

| # | Command | Tests |
|---|---|---|
| 1 | move red cube right of blue block | Baseline |
| 2 | move green cylinder behind yellow container | Attribute reasoning |
| 3 | move cyan sphere left of green cylinder | **Novel colour generalisation** |
| 4 | move purple container beside red cube | Novel composition |
| 5 | move blue block in front of yellow container | Multi-constraint |

## Ambiguity Handling

If your command is ambiguous (e.g. `move that box`), COMPOSE asks for clarification instead of executing blindly. This is by design — confidence threshold is 0.7.
