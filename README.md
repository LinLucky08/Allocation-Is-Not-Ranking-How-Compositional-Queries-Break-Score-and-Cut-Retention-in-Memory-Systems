# Code & Data Archive — "Allocation Is Not Ranking"

Repository accompanying the manuscript submitted to *Annals of the New York
Academy of Sciences*. Every table in the paper and the supplementary material
is reproducible from the scripts indexed below.

## Layout

```
code_repository/
├── README.md              <- this file (per-table index)
├── requirements.txt
├── run_all.py             <- runs every script below in order
├── exp_synth/             <- synthetic workload (U=300, G=100, m=3, B=90)
├── real/                  <- 20 Newsgroups + LongMemEval experiments
│   ├── build_corpus.py    <- corpus-reconstruction sweep (runs offline after first download)
│   ├── exp_real.py        <- Table 3 regeneration (group-level model)
│   ├── exp_real_full.py   <- full regeneration: Tables 3/4/5 + budget sweep + lambda bands (Supplementary S2.8)
│   └── exp_longmemeval.py <- LongMemEval comparison regeneration (Supplementary S2.9; near-exact reproduction of Table 7)
├── rerun/                 <- independent re-runs recorded in Supplementary S2.3
├── audit/                 <- baseline-audit scripts (Supplementary S1)
├── calibration/           <- 216-setting generator calibration (Supplementary S2.5)
└── results/               <- generated outputs: reconstruction hits, regenerated Tables 3/4/5, budget sweep, LongMemEval comparison
    ├── reconstruction_result.txt    <- every exact-count hit of the 5,732/16 sweep
    └── reconstruction_summary.md    <- the finding recorded in the manuscript's Data Availability statement
```

The raw 20 Newsgroups corpus is **not committed** (public dataset, ~7,500 files);
`real/build_corpus.py` downloads it on first run, then works offline.

## Corpus reconstruction (already executed; results archived)

`results/reconstruction_summary.md` documents the sweep behind the manuscript's
Data Availability statement: the 16-group / 5,732-document configuration is
reproducible in count — 297–355 exact-count hits per tokenisation, **all of
which drop alt.atheism and comp.graphics** — while the identity of the remaining
two dropped groups is not recoverable from the description. Re-run any time with
`python real/build_corpus.py`.


## Per-table index

Fill the right-hand column with the actual script filenames as you copy them in;
`run_all.py` reads this list.

### Main text

| Paper table | Content | Script (fill in) | Seed(s) |
|---|---|---|---|
| Table 1 | Corollary 7 perturbation bound vs measured coverage (δ sweep) | `exp_synth/TODO_cor7_sweep.py` | 5 seeds |
| Table 2 | Structural gap vs estimation error across γ | `exp_synth/TODO_gap_gamma.py` | 5 seeds |
| Table 3 | Real workload, 16 groups, exhaustive optima | `real/TODO_real_main.py` | — |
| Table 4 | Four allocators + MMR on the real workload | `real/TODO_allocators.py` | — |
| Table 5 | MMR under four similarities | `real/TODO_similarity.py` | — |
| Table 6 | Online re-allocation over 60 turns | `exp_synth/TODO_online.py` | 5 seeds |
| Table 7 | LongMemEval, 426 questions | `real/TODO_longmemeval.py` | — |

### Supplementary

| Supp table | Content | Script (fill in) |
|---|---|---|
| S1 | Constructed worst cases (clique), un-normalised objective | `exp_synth/TODO_clique.py` |
| S2 | Structural gap vs retention budget | `exp_synth/TODO_budget.py` |
| S3 | Admissible λ per step | `real/TODO_lambda_bands.py` |
| S4 | Write budget (2 swaps vs full re-solve) | `exp_synth/TODO_writebudget.py` |
| S5–S7 | Baseline-audit claims 1–3 | `audit/hedge.py`, `audit/exp7_audit.py` |
| S8 | 12 pairs × 4 budgets × parameter cells | `exp_synth/TODO_pairs.py` |
| S9 | Interaction-value scan R_b | `exp_synth/TODO_rb_scan.py` |
| S10 | Re-run of Table 2 | `rerun/TODO_gap_gamma_rerun.py` |
| S11 | Re-run of Table S2 | `rerun/TODO_budget_rerun.py` |
| S12 | Re-run of the MMR audit | `rerun/TODO_mmr_rerun.py` |
| S13 | Re-run of the online protocol | `rerun/TODO_online_rerun.py` |
| S14 | Endpoint-criteria audit on the 1,512-cell grid | `exp_synth/TODO_grid_audit.py` |
| S15 | Literal vs group-comparison greedy | `exp_synth/TODO_normalisation.py` |
| S16 | Calibration search over 216 settings | `calibration/TODO_calibrate.py` |
| S17–S19 | Rebuilt claims 1–3 (from §6.5 description) | `audit/TODO_rebuild_claim{1,2,3}.py` |

Known script names from the manuscript: `exp8_optimality.py` (Theorem 6, 100
instances; outputs `exp8_results.json`), `real/exp_real_budget.py` (the draft's
R2 budget sweep, retained as the draft's record).

## Reproduction contract

1. `python run_all.py` regenerates every JSON in `results/`; each script states
   its seed, runtime and the table it feeds in its header.
2. The 16-group / 5,732-document corpus subset is reconstructed by
   `real/build_corpus.py` (see the manuscript's Data Availability statement).
3. Common random numbers are shared across deployments within a seed; seeds are
   the integers 0–4 unless a table states otherwise.

## Archive steps (one-time)

```bash
git init && git add . && git commit -m "Initial archive"
git remote add origin <github-url> && git push -u origin main
# then: zenodo.org -> new upload from GitHub release -> reserve DOI
# finally: put the DOI in the manuscript's Data Availability statement
```
