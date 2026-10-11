#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BATCH_DIR = Path(__file__).resolve().parent / "batches"
STATE_PATH = BATCH_DIR / "loop_state.json"
CYCLES_PATH = BATCH_DIR / "loop_cycles.jsonl"
BUILDS_PATH = BATCH_DIR / "smb_builds.jsonl"
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

def run(cmd, cwd=ROOT, timeout=None):
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def load_state():
    with open(STATE_PATH) as f:
        return json.load(f)

def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=1)

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def git_commit_push(msg):
    run(["git", "add", "-A"])
    rc, out, err = run(["git", "commit", "-m", msg])
    if rc != 0:
        pass  # might be no changes; ignore
    run(["git", "push", "origin", "academic-evaluation"])

# Simple continuous loop
def main():
    print(f"Continuous loop starting at {utcnow()}", file=sys.stderr)
    batch_size = 12
    while True:
        try:
            st = load_state()
        except Exception as e:
            print(f"ERR load state: {e}", flush=True)
            time.sleep(10)
            continue
        start_cycle = st.get("cycles_done", 0) + 1
        batch_id = f"auto-{start_cycle:04d}-{start_cycle+batch_size-1:04d}"
        st["batch_id"] = batch_id
        save_state(st)

        print(f"[{utcnow()}] Starting batch {batch_id} from cycle {start_cycle}", flush=True)

        # Run loop_pivot to generate seeds/candidates as needed - we'll use word_pivot style draws
        # For now, just simulate by running word_pivot draws 12 times
        builds_in_batch = 0
        for i in range(batch_size):
            cycle = start_cycle + i
            # Draw 1 word with word_pivot
            rc, out, err = run([PYTHON, "-m", "cognitive.word_pivot", "--draw", "1"])
            if rc != 0:
                continue
            # Parse results
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except:
                    continue
                if r.get("seed_word") and r.get("min_inter") is not None and r.get("min_inter") <= 0.42 and r.get("substantive", 0) >= 2 and r.get("bridge_id"):
                    w = r["word"]
                    bi = r["bridge_id"]
                    cm1 = r.get("A_sense_id")
                    cm2 = r.get("B_sense_id")
                    if not cm1 or not cm2:
                        continue
                    # Try gate/build
                    rcg, outg, errg = run([PYTHON, "-m", "cognitive.build_gate", "--check", cm1, cm2, bi, "--pivot", w, "--json"])
                    if rcg == 0:
                        try:
                            jg = json.loads(outg)
                        except:
                            jg = {}
                        if jg.get("ok"):
                            cc = jg.get("info", {}).get("child_cosine")
                            if cc is not None and cc <= 0.51:
                                # build
                                rcb, outb, errb = run([PYTHON, "-m", "cognitive.build_gate", cm1, cm2, bi, "--pivot", w, "--json"])
                                if rcb == 0:
                                    try:
                                        jb = json.loads(outb)
                                    except:
                                        jb = {}
                                    if jb.get("ok") and jb.get("inserted_id"):
                                        builds_in_batch += 1
            # record cycle
            try:
                from cognitive import ledger
                ledger.record_cycle(cycle, seed_mb_id="", seed_level=14, outcome="scan" if builds_in_batch == 0 else "build", probes_run=[f"word_pivot draw {w}" if 'w' in locals() else "word_pivot draw"], rationale="continuous batch scan with word_pivot", note=f"builds={builds_in_batch}")
            except:
                pass

        st = load_state()
        st["cycles_done"] = start_cycle + batch_size - 1
        st["last_commit_cycle"] = st["cycles_done"]
        st["next_commit_cycle"] = st["cycles_done"] + 1
        save_state(st)

        msg = f"loop: batch {batch_id} ({start_cycle}-{st['cycles_done']}) — builds={builds_in_batch}"
        # commit only, no push
        try:
            run([GIT, "add", "cognitive/batches/loop_cycles.jsonl", "cognitive/batches/instigations.jsonl", "cognitive/batches/smb_builds.jsonl", "cognitive/batches/loop_state.json", "opencode.json"])
            run([GIT, "commit", "-q", "-m", msg])
        except Exception as e:
            print(f"commit err {e}", flush=True)
        print(f"[{utcnow()}] DONE {msg} (no push)", flush=True)
        time.sleep(5)

if __name__ == "__main__":
    main()
