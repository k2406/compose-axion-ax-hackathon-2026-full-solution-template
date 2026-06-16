"""
COMPOSE — Evaluation Script
Runs all 5 benchmark scenarios and prints KPIs to stdout + saves results.csv
Usage: python evaluate.py
"""

import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))

from core.scene import make_default_scene
from core.reasoning import reason

DEMOS = [
    {
        "id": 1,
        "scenario": "Baseline",
        "command": "move red cube right of blue block",
        "expect_success": True,
        "expect_novel": False,
    },
    {
        "id": 2,
        "scenario": "Attribute reasoning",
        "command": "move green cylinder behind yellow container",
        "expect_success": True,
        "expect_novel": False,
    },
    {
        "id": 3,
        "scenario": "Novel colour generalisation",
        "command": "move cyan sphere left of green cylinder",
        "expect_success": True,
        "expect_novel": True,
    },
    {
        "id": 4,
        "scenario": "Novel composition",
        "command": "move purple container beside red cube",
        "expect_success": True,
        "expect_novel": False,
    },
    {
        "id": 5,
        "scenario": "Multi-constraint",
        "command": "move blue block in front of yellow container",
        "expect_success": True,
        "expect_novel": False,
    },
]

AMBIGUITY_DEMO = {
    "id": 6,
    "scenario": "Ambiguity handling",
    "command": "move that box",
    "expect_success": False,
    "expect_novel": False,
}


def run_evaluation():
    print("=" * 60)
    print("COMPOSE — Benchmark Evaluation")
    print("=" * 60)

    scene = make_default_scene()
    results = []
    total = correct = novel_total = novel_ok = spatial_total = spatial_ok = 0

    all_demos = DEMOS + [AMBIGUITY_DEMO]

    for demo in all_demos:
        t0 = time.time()
        r  = reason(demo["command"], scene)
        ms = round((time.time() - t0) * 1000, 1)

        got_ok   = r.success
        expected = demo["expect_success"]
        passed   = got_ok == expected

        total   += 1
        correct += int(passed)

        if demo["expect_novel"]:
            novel_total += 1
            novel_ok    += int(r.success)

        if r.intent and r.intent.spatial_rel:
            spatial_total += 1
            spatial_ok    += int(r.success)

        status = "PASS" if passed else "FAIL"
        novel  = " [NOVEL]" if r.is_novel else ""
        print(f"\n[{status}] Demo {demo['id']} — {demo['scenario']}{novel}")
        print(f"  Command : {demo['command']}")
        print(f"  Result  : {r.message[:80]}")
        print(f"  Conf    : {r.confidence}   Time: {ms}ms")

        results.append({
            "demo_id":       demo["id"],
            "scenario":      demo["scenario"],
            "command":       demo["command"],
            "passed":        passed,
            "success":       r.success,
            "is_novel":      r.is_novel,
            "confidence":    r.confidence,
            "latency_ms":    ms,
        })

    # ── KPIs ──────────────────────────────────────────────────────────────────
    tsr       = correct / total * 100
    novel_gen = (novel_ok / novel_total * 100) if novel_total else 0
    goal_acc  = (spatial_ok / spatial_total * 100) if spatial_total else 0

    print("\n" + "=" * 60)
    print("KPI SUMMARY")
    print("=" * 60)
    print(f"  Task Success Rate (TSR)        : {tsr:.1f}%   (target ≥ 80%)")
    print(f"  Goal Condition Accuracy        : {goal_acc:.1f}%   (target ≥ 90%)")
    print(f"  Novel Colour Generalisation    : {novel_gen:.1f}%   (baseline ~20%)")
    print(f"  Tests passed                   : {correct}/{total}")
    print("=" * 60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")

    return tsr, goal_acc, novel_gen


if __name__ == "__main__":
    run_evaluation()
