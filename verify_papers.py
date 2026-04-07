"""verify_papers.py — checks all papers have all required ground truth keys"""
import json
from pathlib import Path

all_ok = True
for i in range(1, 4):
    path = Path("data") / "papers" / f"paper_00{i}.json"
    d = json.loads(path.read_text())
    keys = list(d["ground_truth"].keys())
    required = ["task1_violations", "task2_inconsistencies", "task3_discrepancies", "task4_citations"]
    missing = [k for k in required if k not in keys]
    if missing:
        print(f"FAIL paper_00{i}: missing {missing}")
        all_ok = False
    else:
        t4 = d["ground_truth"]["task4_citations"]
        ghosts = [r for r in t4 if r.get("injected")]
        print(f"OK   paper_00{i}: {len(t4)} refs, {len(ghosts)} ghost(s) — keys: {keys}")

print()
if all_ok:
    print("ALL PAPERS OK — ready to push")
else:
    print("ERRORS FOUND — run fix_papers.py first")
