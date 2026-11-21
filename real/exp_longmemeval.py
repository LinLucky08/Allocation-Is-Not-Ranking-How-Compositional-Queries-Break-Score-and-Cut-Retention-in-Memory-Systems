# -*- coding: utf-8 -*-
"""Regeneration of the LongMemEval comparison (manuscript Table 7).

Protocol, as described in the manuscript:
  * Data: the public LongMemEval S release (500 questions, ~48 sessions and
    ~115k tokens of timestamped chat history per question). This script uses
    the longmemeval-cleaned release (2025/09) hosted at
    huggingface.co/datasets/xiaowu0162/longmemeval-cleaned; the draft's
    recorded run used the original 2024 release, so session contents may
    differ modestly - the qualitative pattern is the claim under test.
  * A session is a group, a turn is an entry. Questions whose answer is
    supported by one or two sessions are kept (one session = disjunctive
    form, two sessions = conjunctive form); abstention questions carry no
    evidence sessions and are excluded.
  * A question is covered iff all of its evidence sessions are retained.
  * Budgets: retain B = f x U turns, f in {0.2%, 1%, 5%, 10%, 15%, 30%, 50%},
    U = total turns of the question's history.
  * Allocators:
      recency - the newest B turns of the timestamped timeline (the default
                of every tiered store);
      random  - B turns uniformly at random (5 seeds, averaged);
      spread  - the marginal-gain allocator on group activation: with
                evidence-blind uniform gains this takes one turn per session
                in timeline order, maximising the number of distinct sessions
                touched;
      oracle  - retains the evidence sessions first, then spreads the
                remaining budget one turn per remaining session.
  * Reported: binary coverage per budget, and the mean number of sessions
    touched by recency and spread.

Data: place longmemeval_s_cleaned.json in code_repository/data/
(source: https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned or
 https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

Run:  python exp_longmemeval.py
"""
import os, re, json, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CANDIDATES = [
    os.path.join(DATA, "longmemeval_s_cleaned.json"),
    os.path.join(DATA, "longmemeval_s.json"),
]
OUT = os.path.join(os.path.dirname(HERE), "results", "exp_longmemeval_results.json")
BUDGETS = [0.002, 0.01, 0.05, 0.10, 0.15, 0.30, 0.50]
SEEDS = 5

def main():
    path = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if path is None:
        print("FATAL: LongMemEval data not found. Download")
        print("  https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")
        print("  (~264 MB) and save it as code_repository/data/longmemeval_s_cleaned.json")
        raise SystemExit(1)
    print("loading", path, "...")
    data = json.load(open(path, encoding="utf-8"))
    print("questions in release:", len(data))

    kept = []
    for q in data:
        ev = set(q.get("answer_session_ids") or [])
        if 1 <= len(ev) <= 2:
            kept.append(q)
    print("kept questions (1-2 evidence sessions):", len(kept),
          "(paper's recorded run: 426 on the 2024 release)")
    kept_ids = {q["question_id"] for q in kept}
    abst = sum(1 for q in data if q["question_id"].endswith("_abs"))
    print("abstention questions excluded:", abst)

    # build per-question worlds
    worlds = []
    for q in kept:
        sessions = q["haystack_sessions"]
        ids = q["haystack_session_ids"]          # sorted by timestamp (S release)
        ev = set(q["answer_session_ids"])
        turns_by_session = []
        for sid, sess in zip(ids, sessions):
            n_turns = len(sess) if isinstance(sess, list) else len(sess.get("turns", []))
            turns_by_session.append((sid, n_turns))
        U = sum(t for _, t in turns_by_session)
        worlds.append({"qid": q["question_id"], "U": U, "ev": ev,
                       "n_sessions": len(ids), "turns_by_session": turns_by_session})

    U_list = sorted(w["U"] for w in worlds)
    med_U = U_list[len(U_list)//2]
    sess_list = sorted(w["n_sessions"] for w in worlds)
    med_S = sess_list[len(sess_list)//2]
    print(f"median world: {med_U} turns / {med_S} sessions")
    print("evidence sizes:", {k: sum(1 for w in worlds if len(w['ev'])==k) for k in (1,2)})

    def covered(w, kept_turns_by_session):
        """kept_turns_by_session: list of (sid, n_kept) -> session active iff n_kept>0"""
        active = {sid for sid, k in kept_turns_by_session if k > 0}
        return 1 if w["ev"] <= active else 0

    def alloc_recency(w, B):
        # newest B turns: walk sessions from the end of the timeline
        rem, kept = B, []
        for sid, t in reversed(w["turns_by_session"]):
            take = min(t, rem)
            kept.append((sid, take)); rem -= take
            if rem <= 0:
                break
        return kept

    def alloc_spread(w, B):
        # one turn per session in timeline order (evidence-blind spreading)
        rem, kept = B, []
        for sid, t in w["turns_by_session"]:
            if rem <= 0:
                break
            take = 1 if rem >= 1 else 0
            kept.append((sid, take)); rem -= take
        return kept

    def alloc_oracle(w, B):
        # evidence sessions first, then spread the rest one turn per session
        rem, kept = B, []
        for sid, t in w["turns_by_session"]:
            if rem <= 0:
                break
            if sid in w["ev"]:
                kept.append((sid, min(t, 1))); rem -= 1
        for sid, t in w["turns_by_session"]:
            if rem <= 0:
                break
            if sid not in w["ev"] and all(sid != s for s, _ in kept):
                kept.append((sid, 1)); rem -= 1
        return kept

    def alloc_random(w, B, rng):
        all_turns = [(sid, k) for sid, t in w["turns_by_session"] for k in range(t)]
        idx = rng.sample(range(len(all_turns)), min(B, len(all_turns)))
        cnt = {}
        for i in idx:
            sid, _ = all_turns[i]
            cnt[sid] = cnt.get(sid, 0) + 1
        return list(cnt.items())

    rows = []
    for f in BUDGETS:
        cov = {"recency": [], "random": [], "spread": [], "oracle": []}
        grec, gspread = [], []
        for w in worlds:
            B = max(1, round(f * w["U"]))
            kr = alloc_recency(w, B)
            cov["recency"].append(covered(w, kr))
            grec.append(len({sid for sid, k in kr if k > 0}))
            ks = alloc_spread(w, B)
            cov["spread"].append(covered(w, ks))
            gspread.append(len({sid for sid, k in ks if k > 0}))
            ko = alloc_oracle(w, B)
            cov["oracle"].append(covered(w, ko))
            rsum = []
            for seed in range(SEEDS):
                rng = random.Random(seed * 1000 + int(w["qid"][-4:], 16) if w["qid"][-4:].isalnum() else seed)
                rsum.append(covered(w, alloc_random(w, B, rng)))
            cov["random"].append(statistics.mean(rsum))
        row = {"budget": f, "recency": round(statistics.mean(cov["recency"]), 3),
               "random": round(statistics.mean(cov["random"]), 3),
               "spread": round(statistics.mean(cov["spread"]), 3),
               "oracle": round(statistics.mean(cov["oracle"]), 3),
               "groups_recency": round(statistics.mean(grec), 1),
               "groups_spread": round(statistics.mean(gspread), 1)}
        rows.append(row)
        print(f"  budget {f*100:5.1f}% | recency {row['recency']:.3f} random {row['random']:.3f} "
              f"spread {row['spread']:.3f} oracle {row['oracle']:.3f} | groups (rec/sp) "
              f"{row['groups_recency']}/{row['groups_spread']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"conventions": {
                 "data": "longmemeval-cleaned release (2025/09); draft used the 2024 original",
                 "kept": "questions with 1-2 evidence sessions (abstention excluded)",
                 "coverage": "binary: all evidence sessions retained",
                 "spread": "evidence-blind one-turn-per-session in timeline order",
                 "oracle": "evidence sessions first, then one turn per remaining session",
                 "random": "5 seeds averaged",
               },
               "n_kept": len(kept), "median_turns": med_U, "median_sessions": med_S,
               "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
