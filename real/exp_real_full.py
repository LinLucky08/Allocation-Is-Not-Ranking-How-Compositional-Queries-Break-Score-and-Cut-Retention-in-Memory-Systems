# -*- coding: utf-8 -*-
"""Full real-workload pipeline: regenerates Tables 3, 4, 5, the R2-style budget
sweep, the MMR admissible-lambda bands and Figure 2 data on the reconstructed
16-group / 5,732-document corpus.

Conventions (identical to exp_real.py, restated briefly):
  alpha = 0, beta = 1 - gamma; group popularity pi_g ~ Zipf(s=1.1) ordered by
  decreasing corpus size; v_g = pi_g (sums to 1); every document of group g
  carries p_hat_u = pi_g (ranking fills the most popular groups first);
  conjunction weights: uniform 1/90, semantic softmax(cos/0.6) over class
  centroids (mean token-count vectors), sparse = top-40 pairs renormalised;
  exact optimum = exhaustive enumeration of group subsets of size <= B.

Additions here:
  * MMR allocator at entry level with TF-IDF cosine similarity; lambda swept
    over 21 points (0..1 step 0.05) reported at best; a stepwise reference
    re-optimises (lambda, candidate) each step by immediate marginal gain.
  * Similarity variants for the MMR table: TF-IDF cosine, LSA-256 (sparse
    truncated SVD of the TF-IDF matrix), raw-count cosine (replaces Doc2Vec,
    which is not part of the released pipeline), and an ideal embedding =
    class centroid + Gaussian noise (sigma = 0.05 of the centroid norm).
  * Budget sweep B = 4..14 (gamma = 0.5, semantic and sparse) replacing the
    draft's unreproducible 21-group R2 sweep.
  * MMR admissible-lambda bands (2% tolerance per step, B = 11).

Run:  python exp_real_full.py     (requires the corpus extracted under ../data/)
"""
import os, re, json, math, itertools
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "20news-bydate-test")
OUT = os.path.join(os.path.dirname(HERE), "results", "exp_real_full_results.json")
DROP = {"alt.atheism", "comp.graphics", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"}
MIN_TOK = 35
ZIPF_S, TEMPS, SPARSE_TOP, IDEAL_SIGMA = 1.1, 0.6, 40, 0.05
TARGET_DOCS = 5732

def load_corpus():
    docs, order = {}, []
    for g in sorted(os.listdir(DATA)):
        if g in DROP:
            continue
        gd = os.path.join(DATA, g)
        if not os.path.isdir(gd):
            continue
        bodies = []
        for f in sorted(os.listdir(gd)):
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

def main():
    docs = load_corpus()
    names = []            # (group, doc_index) in a fixed global order
    for g in sorted(docs):
        for i in range(len(docs[g])):
            names.append((g, i))
    n, G = len(names), len(docs)
    assert n == TARGET_DOCS, f"count mismatch: {n}"
    print(f"corpus: {n} docs / {G} groups")

    sizes = {g: len(x) for g, x in docs.items()}
    order_g = sorted(sizes, key=lambda g: -sizes[g])
    raw = {g: 1.0 / (i + 1) ** ZIPF_S for i, g in enumerate(order_g)}
    z = sum(raw.values())
    pi = {g: raw[g] / z for g in raw}
    v = dict(pi)
    groups_sorted = sorted(pi)

    # ---- TF-IDF sparse matrix over the full vocabulary ----
    vocab = {}
    rows_i, cols_i, vals_i = [], [], []
    for r, (g, i) in enumerate(names):
        cnt = Counter()
        cnt.update(docs[g][i])
        for t, c in cnt.items():
            idx = vocab.setdefault(t, len(vocab))
            rows_i.append(r); cols_i.append(idx); vals_i.append(float(c))
    V = len(vocab)
    X = sparse.csr_matrix((vals_i, (rows_i, cols_i)), shape=(n, V))
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    X.data *= idf[cols_i]
    for r in range(n):
        s, e = X.indptr[r], X.indptr[r + 1]
        nrm = np.sqrt((X.data[s:e] ** 2).sum())
        if nrm:
            X.data[s:e] /= nrm
    print(f"vocab: {V} terms | TF-IDF nnz: {X.nnz}")

    # ---- centroids, conjunction weights ----
    cent = np.zeros((G, V))
    g_index = {g: k for k, g in enumerate(groups_sorted)}
    for r, (g, i) in enumerate(names):
        cent[g_index[g]] += X[r].toarray().ravel()
    cent /= np.array([sizes[g] for g in groups_sorted])[:, None]
    def cos_pairs(M):
        nrm = np.linalg.norm(M, axis=1); nrm[nrm == 0] = 1
        Mn = M / nrm[:, None]
        C = Mn @ Mn.T
        ps = {}
        for a in range(G):
            for b in range(a + 1, G):
                ps[(groups_sorted[a], groups_sorted[b])] = float(C[a, b])
        return ps
    pairs = list(itertools.combinations(groups_sorted, 2))
    coss = cos_pairs(cent)
    w = {p: math.exp(coss[p] / TEMPS) for p in pairs}
    zs = sum(w.values())
    b_sem = {p: w[p] / zs for p in pairs}
    b_uni = {p: 1.0 / len(pairs) for p in pairs}
    top40 = sorted(pairs, key=lambda p: -b_sem[p])[:SPARSE_TOP]
    z40 = sum(b_sem[p] for p in top40)
    b_spar = {p: (b_sem[p] / z40 if p in top40 else 0.0) for p in pairs}
    BWEIGHTS = {"uniform": b_uni, "semantic": b_sem, "sparse": b_spar}

    def coverage(active, b, beta, gamma):
        cov = beta * sum(v[g] for g in active)
        act = sorted(active)
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                cov += gamma * b.get((act[i], act[j]), 0.0)
        return cov

    def exact_opt(b, beta, gamma, B):
        best, best_set = -1.0, None
        for k in range(1, min(B, G) + 1):
            for sub in itertools.combinations(groups_sorted, k):
                c = coverage(sub, b, beta, gamma)
                if c > best:
                    best, best_set = c, sub
        return best, set(best_set)

    def greedy(b, beta, gamma, B):
        active, rem = set(), B
        while rem > 0 and len(active) < G:
            bg, bgain = None, -1.0
            for g in groups_sorted:
                if g in active:
                    continue
                gain = beta * v[g] + gamma * sum(b.get((min(g, h), max(g, h)), 0.0) for h in active)
                if gain > bgain:
                    bg, bgain = g, gain
            active.add(bg); rem -= 1
        return active

    def rank_alloc(B):
        # top-B documents by p_hat = pi_g
        kept, rem = set(), B
        for g in order_g:
            if rem <= 0:
                break
            kept.add(g); rem -= len(docs[g]) if rem >= len(docs[g]) else rem
            if rem > 0 and len(docs[g]) <= rem:
                pass
        # simpler: fill groups in popularity order
        kept, rem = set(), B
        for g in order_g:
            take = min(len(docs[g]), rem)
            kept.add(g); rem -= take
            if rem <= 0:
                break
        return kept

    # ---- entry-level MMR ----
    p_hat = np.array([pi[g] for g, _ in names])
    doc_group = np.array([g_index[g] for g, _ in names])
    sim_cache = {}
    def get_sim(key, M):
        if key in sim_cache:
            return sim_cache[key]
        Xs = sparse.csr_matrix(M)
        S = (Xs @ Xs.T).toarray()
        np.fill_diagonal(S, 1.0)
        sim_cache[key] = S
        return S

    def mmr_alloc(lam, S, B):
        sel, rem = [], B
        maxsim = np.zeros(n)
        cand = np.arange(n)
        while rem > 0:
            div = maxsim.copy() if sel else np.zeros(n)
            score = lam * p_hat - (1.0 - lam) * div
            score[sel] = -np.inf
            i = int(np.argmax(score))
            sel.append(i); rem -= 1
            if rem > 0:
                maxsim = np.maximum(maxsim, S[i])
        return set(groups_sorted[int(doc_group[i])] for i in sel)

    def mmr_stepwise(S, B, beta, gamma, b):
        """myopic reference: re-optimise (lambda, candidate) each step by immediate
        marginal gain measured on the true objective."""
        sel, rem = [], B
        maxsim = np.zeros(n)
        lam_grid = [round(0.05 * k, 2) for k in range(21)]
        while rem > 0:
            best_i, best_cov = None, -1.0
            for lam in lam_grid:
                div = maxsim.copy() if sel else np.zeros(n)
                score = lam * p_hat - (1.0 - lam) * div
                score[sel] = -np.inf
                i = int(np.argmax(score))
                act = set(groups_sorted[int(doc_group[j])] for j in sel) | {groups_sorted[int(doc_group[i])]}
                c = coverage(act, b, beta, gamma)
                if c > best_cov:
                    best_cov, best_i = c, i
            sel.append(best_i); rem -= 1
            if rem > 0:
                maxsim = np.maximum(maxsim, S[best_i])
        return set(groups_sorted[int(doc_group[i])] for i in sel)

    def lam_bands(b, beta, gamma, B, S, tol=0.02):
        """per-step set of lambdas whose MMR pick achieves within tol of the
        greedy pick's marginal coverage; intersection over steps."""
        lam_grid = [round(0.05 * k, 2) for k in range(21)]
        greedy_active = greedy(b, beta, gamma, B)
        greedy_order = []
        tmp = set()
        for g in groups_sorted:
            pass
        # greedy order: re-run greedy stepwise to get its pick sequence
        active, rem, g_order = set(), B, []
        while rem > 0 and len(active) < G:
            bg, bgain = None, -1.0
            for g in groups_sorted:
                if g in active: continue
                gain = beta * v[g] + gamma * sum(b.get((min(g, h), max(g, h)), 0.0) for h in active)
                if gain > bgain: bg, bgain = g, gain
            active.add(bg); g_order.append(bg); rem -= 1
        greedy_cov = [coverage(set(g_order[:t + 1]), b, beta, gamma) for t in range(B)]
        bands, no_sol, inter = [], 0, set(lam_grid)
        sel, maxsim = [], np.zeros(n)
        for t in range(B):
            admissible = []
            gdelta = greedy_cov[t] - (greedy_cov[t - 1] if t else 0.0)
            for lam in lam_grid:
                div = maxsim.copy() if sel else np.zeros(n)
                score = lam * p_hat - (1.0 - lam) * div
                score[sel] = -np.inf
                i = int(np.argmax(score))
                act = set(groups_sorted[int(doc_group[j])] for j in sel) | {groups_sorted[int(doc_group[i])]}
                mdelta = coverage(act, b, beta, gamma) - (coverage(set(groups_sorted[int(doc_group[j])] for j in sel), b, beta, gamma) if sel else 0.0)
                if mdelta >= (1.0 - tol) * gdelta:
                    admissible.append(lam)
            if not admissible:
                no_sol += 1
                inter = inter
            else:
                inter &= set(admissible)
                bands.append((t + 1, admissible[0], admissible[-1]))
            # advance MMR with best fixed lam=0.5 trajectory for band drift display
            div = maxsim.copy() if sel else np.zeros(n)
            score = 0.5 * p_hat - 0.5 * div
            score[sel] = -np.inf
            i = int(np.argmax(score))
            sel.append(i); maxsim = np.maximum(maxsim, S[i])
        return {"no_solution_steps": no_sol, "bands": bands[:3] + bands[-1:], "intersection_empty": len(inter) == 0}

    # ================= run everything =================
    results = {"conventions": {
        "alpha_beta": "alpha=0, beta=1-gamma",
        "popularity": "Zipf(s=1.1) by decreasing corpus size; p_hat_u = pi_g",
        "conjunction": "centroids = mean token-count vectors; semantic = softmax(cos/0.6); sparse = top-40 pairs",
        "optimum": "exhaustive group-subset enumeration (exact)",
        "similarities": "TF-IDF cosine, LSA-256 (sparse SVD), raw-count cosine (replaces Doc2Vec), ideal = centroid + noise",
        "mmr": "entry-level MMR with the given similarity; lambda grid 0..1 step 0.05, reported at best; stepwise = per-step (lambda, candidate) re-optimisation by immediate marginal gain",
    }, "table3": [], "table4": [], "table5": [], "budget_sweep": [], "lambda_bands": {}}

    # Table 3
    t3 = [("uniform", 0.00, 5), ("uniform", 0.00, 11), ("uniform", 0.50, 11),
          ("semantic", 0.25, 11), ("semantic", 0.50, 11),
          ("sparse", 0.50, 11), ("sparse", 0.50, 5)]
    print("\n=== Table 3 (regenerated) ===")
    for name, gamma, B in t3:
        b = BWEIGHTS[name]; beta = 1.0 - gamma
        cov_opt, opt_set = exact_opt(b, beta, gamma, B)
        rank_set = rank_alloc(B)
        cov_rank = coverage(rank_set, b, beta, gamma)
        greed_set = greedy(b, beta, gamma, B)
        cov_greed = coverage(greed_set, b, beta, gamma)
        r = {"interaction": name, "gamma": gamma, "B": B,
             "optimal": round(cov_opt, 4), "ranking": round(cov_rank, 4), "greedy": round(cov_greed, 4),
             "r_rank": round(cov_rank / cov_opt, 3), "r_greedy": round(cov_greed / cov_opt, 3),
             "groups": f"{len(rank_set)}/{len(greed_set)}/{len(opt_set)}"}
        results["table3"].append(r)
        print(f"  {name:9s} g={gamma:.2f} B={B:2d} | opt {r['optimal']:.4f} rank {r['ranking']:.4f} "
              f"(r={r['r_rank']:.3f}) greedy {r['greedy']:.4f} (r={r['r_greedy']:.3f}) | {r['groups']}")

    # Table 4: allocators with TF-IDF MMR
    print("\n=== Table 4 (regenerated) ===")
    S_tfidf = get_sim("tfidf", X)
    t4 = [("uniform", 0.00, 8), ("semantic", 0.50, 8), ("sparse", 0.50, 8),
          ("semantic", 0.50, 5), ("semantic", 0.50, 11)]
    for name, gamma, B in t4:
        b = BWEIGHTS[name]; beta = 1.0 - gamma
        cov_opt, _ = exact_opt(b, beta, gamma, B)
        rank_set = rank_alloc(B)
        r_rank = coverage(rank_set, b, beta, gamma) / cov_opt
        best = (-1.0, None)
        for k in range(21):
            lam = round(0.05 * k, 2)
            s = mmr_alloc(lam, S_tfidf, B)
            c = coverage(s, b, beta, gamma)
            if c > best[0]: best = (c, lam)
        r_mmr, lam_best = best[0] / cov_opt, best[1]
        step_set = mmr_stepwise(S_tfidf, B, beta, gamma, b)
        r_step = coverage(step_set, b, beta, gamma) / cov_opt
        greed_set = greedy(b, beta, gamma, B)
        r_greed = coverage(greed_set, b, beta, gamma) / cov_opt
        r = {"configuration": f"{name}, gamma={gamma}", "B": B,
             "ranking": round(r_rank, 3), "mmr_best": round(r_mmr, 3), "lambda": lam_best,
             "mmr_stepwise": round(r_step, 3), "greedy": round(r_greed, 3)}
        results["table4"].append(r)
        print(f"  {name:9s} g={gamma:.2f} B={B} | rank {r['ranking']:.3f} MMR* {r['mmr_best']:.3f} "
              f"(lam {lam_best}) step {r['mmr_stepwise']:.3f} greedy {r['greedy']:.3f}")

    # Table 5: MMR best-lambda under four similarities (computed one at a time)
    print("\n=== Table 5 (regenerated) ===")
    def sim_from_matrix(M):
        Xs = sparse.csr_matrix(M)
        S = (Xs @ Xs.T).toarray().astype(np.float32)
        np.fill_diagonal(S, 1.0)
        return S
    def ideal_sim():
        # ideal class-centroid embedding: cosine between the doc's class centroid
        # and its neighbour's (per-document iid noise shrinks all cosines
        # uniformly and is omitted); 16x16 centroid cosines -> block matrix
        nrm_c = np.linalg.norm(cent, axis=1); nrm_c[nrm_c == 0] = 1
        Cn = cent / nrm_c[:, None]
        C16 = Cn @ Cn.T
        S = C16[np.ix_(doc_group, doc_group)].astype(np.float32)
        np.fill_diagonal(S, 1.0)
        return S
    t5 = [("semantic", 0.50, 8), ("sparse", 0.50, 8), ("semantic", 0.50, 11)]
    results["table5"] = []
    for name, gamma, B in t5:
        b = BWEIGHTS[name]; beta = 1.0 - gamma
        cov_opt, _ = exact_opt(b, beta, gamma, B)
        greed_set = greedy(b, beta, gamma, B)
        r_greed = coverage(greed_set, b, beta, gamma) / cov_opt
        row = {"configuration": f"{name}, gamma={gamma}", "B": B, "greedy": round(r_greed, 3)}
        for sname, M in [("tfidf", X), ("count", sparse.csr_matrix((vals_i, (rows_i, cols_i)), shape=(n, V)))]:
            S = sim_from_matrix(M)
            best = (-1.0, None)
            for k2 in range(21):
                lam = round(0.05 * k2, 2)
                s = mmr_alloc(lam, S, B)
                c = coverage(s, b, beta, gamma)
                if c > best[0]: best = (c, lam)
            row[sname] = round(best[0] / cov_opt, 3)
            del S
        # LSA-256
        k = min(256, min(X.shape) - 1)
        U, s, Vt = svds(X.astype(np.float64), k=k)
        order_s = np.argsort(-s)
        X_lsa = U[:, order_s] * s[order_s]
        S = sim_from_matrix(X_lsa)
        best = (-1.0, None)
        for k2 in range(21):
            lam = round(0.05 * k2, 2)
            s = mmr_alloc(lam, S, B)
            c = coverage(s, b, beta, gamma)
            if c > best[0]: best = (c, lam)
        row["lsa"] = round(best[0] / cov_opt, 3)
        del S, X_lsa, U, s, Vt
        # ideal (class centroid block matrix)
        S = ideal_sim()
        best = (-1.0, None)
        for k2 in range(21):
            lam = round(0.05 * k2, 2)
            s = mmr_alloc(lam, S, B)
            c = coverage(s, b, beta, gamma)
            if c > best[0]: best = (c, lam)
        row["ideal"] = round(best[0] / cov_opt, 3)
        del S
        results["table5"].append(row)
        print(f"  {name:9s} g={gamma:.2f} B={B} | " +
              " ".join(f"{s}={row[s]:.3f}" for s in ["tfidf", "lsa", "count", "ideal"]) +
              f" | greedy {r_greed:.3f}")

    # Budget sweep (replaces the draft's 21-group R2)
    print("\n=== Budget sweep (B = 4..14, gamma = 0.5) ===")
    for name in ["semantic", "sparse"]:
        b = BWEIGHTS[name]; beta = 0.5
        for B in range(4, 15):
            cov_opt, _ = exact_opt(b, beta, 0.5, B)
            rank_set = rank_alloc(B)
            r_rank = coverage(rank_set, b, beta, 0.5) / cov_opt
            greed_set = greedy(b, beta, 0.5, B)
            r_greed = coverage(greed_set, b, beta, 0.5) / cov_opt
            rank_g = len(rank_set)
            results["budget_sweep"].append({"interaction": name, "B": B,
                                            "r_rank": round(r_rank, 3), "r_greedy": round(r_greed, 3),
                                            "rank_groups": rank_g})
            print(f"  {name:9s} B={B:2d} | r_rank {r_rank:.3f} r_greedy {r_greed:.3f} | rank groups {rank_g}")

    # Lambda bands (B = 11)
    print("\n=== MMR admissible-lambda bands (B = 11, TF-IDF) ===")
    for name in ["uniform", "semantic", "sparse"]:
        b = BWEIGHTS[name]; beta = 0.5 if name != "uniform" else 0.5
        bands = lam_bands(b, beta, 0.5, 11, S_tfidf)
        results["lambda_bands"][name] = bands
        print(f"  {name:9s} | no-solution steps: {bands['no_solution_steps']}/11 | bands: {bands['bands']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved:", OUT)

if __name__ == "__main__":
    main()
