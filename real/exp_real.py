# -*- coding: utf-8 -*-
"""Real-workload pipeline: fully reproducible regeneration of Table 3.

Everything is specified from the manuscript's descriptions; where the original
description does not pin a parameter down (alpha/beta convention, the popularity
law, the token basis of the centroids), the choice is stated in the header of the
results file so that the pipeline is self-documenting.

Model (documented conventions):
  * Corpus: public 20news-bydate-test split, drop {alt.atheism, comp.graphics,
    comp.sys.ibm.pc.hardware, comp.sys.mac.hardware}, keep bodies with >= 35
    whitespace tokens -> exactly 5,732 documents over 16 groups.
  * Demand is group-level: alpha = 0, beta = 1 - gamma. Atomic queries play no
    role on a newsgroup corpus (no per-document query stream exists).
  * Group popularity pi_g follows Zipf(s=1.1), groups ordered by decreasing
    corpus size (power-law popularity, as stated in the paper); v_g = pi_g
    normalised to sum 1. Every document in group g carries p_hat_u = pi_g, so
    score-and-cut ranking = fill the most popular groups first.
  * Conjunction weights b_gh (over the 90 pairs):
      uniform  = 1/90 each
      semantic = softmax(cosine(centroid_g, centroid_h)/0.6), centroids are
                 mean token-count vectors of the group's documents
      sparse   = semantic restricted to the 40 strongest pairs, renormalised
  * Coverage(S) = beta * sum_{g active} v_g + gamma * sum_{g<h active} b_gh.
  * Exact optimum: with alpha = 0 coverage depends only on the active group
    set, and any subset of at most min(B,16) groups is realisable with B
    documents (one document each) -> exhaustive enumeration over group subsets
    of size <= B gives the true optimum.
  * Allocators: ranking = top-B documents by p_hat; greedy = marginal-gain
    (entry-level; within a group all documents are perfect substitutes).
Run:  python exp_real.py
"""
import os, re, json, math, itertools
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
_CANDIDATES = [
    os.path.join(_DATA, "20news-bydate-test"),
    os.path.join(os.environ.get("TEMP", "/tmp"), "20news", "20news-bydate-test"),
]
ROOT = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])
OUT = os.path.join(os.path.dirname(_HERE), "results", "exp_real_results.json")
DROP = {"alt.atheism", "comp.graphics", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"}
MIN_TOK = 35
ZIPF_S = 1.1
TEMPS = 0.6
SPARSE_TOP = 40
TARGET_DOCS = 5732

def load_corpus():
    if not os.path.isdir(ROOT):
        print("FATAL: corpus not found at", ROOT)
        print("Place 20news-bydate.tar.gz in code_repository/data/ then run real/build_corpus.py,")
        print("or extract it so that this directory exists.")
        raise SystemExit(1)
    docs = {}
    for g in sorted(os.listdir(ROOT)):
        if g in DROP:
            continue
        gd = os.path.join(ROOT, g)
        if not os.path.isdir(gd):
            continue
        bodies = []
        for f in os.listdir(gd):
            try:
                raw = open(os.path.join(gd, f), "rb").read().decode("latin-1", errors="replace")
            except Exception:
                continue
            parts = re.split(r"\r?\n\r?\n", raw, maxsplit=1)
            body = parts[1] if len(parts) == 2 else ""
            toks = body.lower().split()
            if len(toks) >= MIN_TOK:
                bodies.append(toks)
        if bodies:
            docs[g] = bodies
    return docs

def coverage(active, v, b, beta, gamma):
    cov = beta * sum(v[g] for g in active)
    for i, g in enumerate(sorted(active)):
        for h in sorted(active)[i+1:]:
            cov += gamma * b.get((g, h), 0.0)
    return cov

def exact_optimum(v, b, beta, gamma, B, G):
    best, best_set = -1.0, None
    for k in range(1, min(B, G) + 1):
        for sub in itertools.combinations(sorted(v), k):
            c = coverage(sub, v, b, beta, gamma)
            if c > best:
                best, best_set = c, set(sub)
    return best, best_set

def ranking_alloc(docs_by_group, pi, B):
    """top-B documents by p_hat (= group popularity); ties broken by group size."""
    order = []
    for g in sorted(pi, key=lambda x: -pi[x]):
        order.extend([g] * len(docs_by_group[g]))
    kept = order[:B]
    return set(kept)

def greedy_alloc(v, b, beta, gamma, B, G):
    """entry-level marginal-gain; all documents of a group are perfect substitutes."""
    active, remaining = set(), B
    while remaining > 0 and len(active) < G:
        best_gain, best_g = -1.0, None
        for g in v:
            if g in active:
                continue
            gain = v[g] * beta
            for h in active:
                key = (min(g, h), max(g, h))
                gain += gamma * b.get(key, 0.0)
            if gain > best_gain:
                best_gain, best_g = gain, g
        active.add(best_g)
        remaining -= 1
    return active

def main():
    docs = load_corpus()
    n = sum(len(x) for x in docs.values())
    G = len(docs)
    print(f"corpus: {n} docs / {G} groups (target {TARGET_DOCS})")
    assert n == TARGET_DOCS, f"count mismatch: {n}"

    # Zipf popularity, groups ordered by decreasing corpus size
    sizes = {g: len(x) for g, x in docs.items()}
    order = sorted(sizes, key=lambda g: -sizes[g])
    raw = {g: 1.0 / (i + 1) ** ZIPF_S for i, g in enumerate(order)}
    z = sum(raw.values())
    pi = {g: raw[g] / z for g in raw}
    v = dict(pi)

    # centroids: mean token-count vectors
    vocab = {}
    for g, bodies in docs.items():
        for toks in bodies:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
    cent = {}
    for g, bodies in docs.items():
        c = [0.0] * len(vocab)
        for toks in bodies:
            for t in toks:
                c[vocab[t]] += 1.0
        cent[g] = [x / len(bodies) for x in c]
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
    pairs = list(itertools.combinations(sorted(pi), 2))
    coss = {p: cos(cent[p[0]], cent[p[1]]) for p in pairs}

    def softmax_b():
        w = {p: math.exp(coss[p] / TEMPS) for p in pairs}
        z = sum(w.values())
        return {p: w[p] / z for p in pairs}
    b_sem = softmax_b()
    b_uni = {p: 1.0 / len(pairs) for p in pairs}
    top40 = sorted(pairs, key=lambda p: -b_sem[p])[:SPARSE_TOP]
    z = sum(b_sem[p] for p in top40)
    b_spar = {p: (b_sem[p] / z if p in top40 else 0.0) for p in pairs}

    B_VALS = sorted({5, 11})
    rows = []
    for name, b in [("uniform", b_uni), ("semantic", b_sem), ("sparse", b_spar)]:
        for gamma in [0.00, 0.25, 0.50]:
            beta = 1.0 - gamma
            for B in B_VALS:
                skip = not ((gamma == 0.00 and B in (5, 11)) or (gamma == 0.25 and B == 11)
                            or (gamma == 0.50 and B in (5, 11)))
                if skip:
                    continue
                cov_opt, opt_set = exact_optimum(v, b, beta, gamma, B, G)
                rank_set = ranking_alloc(docs, pi, B)
                cov_rank = coverage(rank_set, v, b, beta, gamma)
                greed_set = greedy_alloc(v, b, beta, gamma, B, G)
                cov_greed = coverage(greed_set, v, b, beta, gamma)
                rows.append({
                    "interaction": name, "gamma": gamma, "B": B,
                    "optimal": round(cov_opt, 4), "ranking": round(cov_rank, 4),
                    "greedy": round(cov_greed, 4),
                    "r_rank": round(cov_rank / cov_opt, 3),
                    "r_greedy": round(cov_greed / cov_opt, 3),
                    "groups_rank_greedy_opt": f"{len(rank_set)}/{len(greed_set)}/{len(opt_set)}",
                    "optimal_groups": sorted(opt_set),
                })
                r = rows[-1]
                print(f"{name:9s} g={gamma:.2f} B={B:2d} | opt {r['optimal']:.4f} "
                      f"rank {r['ranking']:.4f} (r={r['r_rank']:.3f}) "
                      f"greedy {r['greedy']:.4f} (r={r['r_greedy']:.3f}) | groups {r['groups_rank_greedy_opt']}")

    meta = {
        "conventions": {
            "alpha_beta": "alpha=0, beta=1-gamma (group-level demand)",
            "popularity": f"Zipf(s={ZIPF_S}), groups ordered by decreasing corpus size",
            "conjunction": f"centroids = mean token-count vectors; semantic = softmax(cos/{TEMPS}); sparse = top-{SPARSE_TOP} pairs renormalised",
            "optimum": "exhaustive enumeration of group subsets of size <= B (exact, since alpha=0)",
            "p_hat": "p_hat_u = pi_g (true popularity; no estimation error on the real workload)",
        },
        "corpus": {"dropped": sorted(DROP), "min_tokens": MIN_TOK, "docs": n, "groups": G},
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(meta, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
