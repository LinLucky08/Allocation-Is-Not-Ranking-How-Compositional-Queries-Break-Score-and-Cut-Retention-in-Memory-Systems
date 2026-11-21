# -*- coding: utf-8 -*-
"""Run every reproduction script in the per-table index (README.md).

Reads the table index, checks which scripts exist, runs them in order, and
reports coverage: which paper tables are currently reproducible from this
repository. Fill the TODO filenames in README.md as you copy scripts in.
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")

def indexed_scripts():
    txt = open(README, encoding="utf-8").read()
    return re.findall(r"`([A-Za-z0-9_/]+\.py)`", txt)

def main():
    me = os.path.basename(os.path.abspath(__file__))
    scripts = [s for s in dict.fromkeys(indexed_scripts())
               if os.path.basename(s) != me and "TODO" not in s]
    todo = [s for s in dict.fromkeys(indexed_scripts()) if "TODO" in s and os.path.basename(s) != me]
    ok = missing = ran = failed = 0
    for s in scripts:
        path = os.path.join(HERE, s.replace("/", os.sep))
        if "TODO" in s:
            missing += 1
            continue
        if not os.path.exists(path):
            print(f"[MISSING FILE] {s}")
            missing += 1
            continue
        ok += 1
        print(f"[run] {s}")
        r = subprocess.run([sys.executable, path], cwd=os.path.dirname(path) or ".")
        if r.returncode == 0:
            ran += 1
        else:
            failed += 1
            print(f"[FAILED] {s} (exit {r.returncode})")
    print("-" * 60)
    print(f"indexed: {len(scripts) + len(todo)} | ready: {ok} (ran {ran}, failed {failed}) | TODO placeholders: {len(todo)}")
    print("Coverage improves as you replace TODO filenames in README.md with real scripts.")

if __name__ == "__main__":
    main()
