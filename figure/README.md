# figure/ - Step 4 publication figures

## Purpose

Step 4 renders final-paper figures **offline**. It makes zero model/API calls:
it only reads the existing `results/<run_id>/` and
`trajectories/<run_id>/` artifacts written by Steps 1-3. The frozen debate
topology, prompts, scoring, RoundValue labels/thresholds, benchmark splits,
historical trajectories, and existing Step 3 plots are never modified.

Generated: 2026-08-17T23:21:40.987532+00:00

## Reproduce

```bash
python scripts/step4_paper_figures.py
```

The script auto-discovers every compatible run from its manifest/metadata
(never from directory names), recomputes all metrics, validates the
policy-comparison invariants, and writes `figure_data.json` plus the five
main-paper PNGs (and one supporting heatmap) in this directory.

## Runs used

| run_id | model | dataset | created_at |
|---|---|---|---|
| `202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d` | DeepSeek-V4-Flash (`deepseek-v4-flash`) | MMLU-Pro-50 | 2026-08-16T22:58:46.490448+00:00 |
| `202608161937_deepseek-v4-flash_HARP50_e8bbd5fd` | DeepSeek-V4-Flash (`deepseek-v4-flash`) | HARP-50 | 2026-08-17T00:37:15.520506+00:00 |
| `202608161940_deepseek-v4-flash_LogiQA50_db682c1c` | DeepSeek-V4-Flash (`deepseek-v4-flash`) | LogiQA-50 | 2026-08-17T00:40:14.950126+00:00 |
| `202608171054_gpt-5-nano_HARP50_42a8def4` | GPT-5-nano (`gpt-5-nano`) | HARP-50 | 2026-08-17T15:54:16.590829+00:00 |
| `202608171724_gpt-4o-mini_HARP50_71642e0e` | GPT-4o-mini (`gpt-4o-mini-2024-07-18`) | HARP-50 | 2026-08-17T22:24:38.138482+00:00 |

Skipped runs: - none

## Run identity

| run_id | model snapshot | temperature | reasoning | R1..R5 accuracy (%) | n_R1_wrong |
|---|---|---|---|---|---|
| `202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d` | deepseek-v4-flash | 0.2 | enabled=False, effort=high | 76 / 76 / 78 / 80 / 78 | 12 |
| `202608161937_deepseek-v4-flash_HARP50_e8bbd5fd` | deepseek-v4-flash | 0.3 | enabled=False, effort=high | 92 / 96 / 96 / 92 / 96 | 4 |
| `202608161940_deepseek-v4-flash_LogiQA50_db682c1c` | deepseek-v4-flash | 0.3 | enabled=False, effort=high | 82 / 82 / 82 / 78 / 76 | 9 |
| `202608171054_gpt-5-nano_HARP50_42a8def4` | gpt-5-nano | 1.0 | enabled=True, effort=medium | 96 / 96 / 96 / 96 / 96 | 2 |
| `202608171724_gpt-4o-mini_HARP50_71642e0e` | gpt-4o-mini-2024-07-18 | 0 | enabled=False | 38 / 42 / 42 / 42 / 44 | 31 |

Step 4 uses exactly one DeepSeek x MMLU-Pro run, `202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d`: snapshot `deepseek-v4-flash`, reasoning enabled=False (effort high), temperature 0.2, R1..R5 = 76 / 76 / 78 / 80 / 78, n_R1_wrong = 12. These values were verified independently against `scores.json`, the raw Writer checkpoints in `trajectories/`, and `analysis.json`; they match the rendered figures exactly. The previously reviewed delayed-checkpoint DeepSeek MMLU-Pro run `202608161016_MMLUPro50_79a9e682` (reasoning enabled) had R1 = 86% and n_R1_wrong = 7; it was retired when DeepSeek reasoning was switched off (git commit `2d03c15`), is not present in `results/`, and is therefore neither selected nor plotted. No metric was edited to match either run.

## What each figure answers

- **fig01_round_accuracy_dynamics.png** - how accuracy evolves from R1 to R5,
  in two panels on a shared 0-100% scale: (a) HARP-50 with one line per
  model, and (b) DeepSeek-V4-Flash with one line per benchmark. The shared
  condition DeepSeek-V4-Flash x HARP-50 is drawn with a black halo and star
  markers in both panels. All tasks (n = 50 per run).
- **fig02_continuation_opportunity.png** - the main RoundValue policy figure.
  Held-out **test tasks only** (n = 10 per run): R1 -> Oracle is the available
  continuation opportunity, R1 -> RoundValue the captured gain, and
  RoundValue -> Oracle the remaining policy regret. When R1 == RoundValue,
  the RoundValue diamond is drawn inside a black R1 ring so the equality is
  never hidden, and the row annotation says "R1 = RoundValue". The
  DeepSeek-V4-Flash x HARP-50 row is tinted and marked with a star.
- **fig03_r1_wrong_trajectories.png** - compact trajectory-pattern counts:
  identical Writer correctness trajectories among R1-wrong tasks are
  aggregated into W/C patterns with a task count per pattern, grouped by
  model x benchmark. No task is outcome-selected. The DeepSeek-V4-Flash x
  HARP-50 row is tinted and marked with a star.
- **fig03_supp_task_level_heatmap.png** - supporting/appendix figure
  preserving the full task-level R1-wrong heatmap for auditability. The
  DeepSeek-V4-Flash x HARP-50 block is tinted and marked with a star.
- **fig04_repair_mechanism.png** - (a) P(blind Stage-1 gold emerges at
  R2..R5 | R1 wrong) per condition, and (b) a mutually exclusive,
  exhaustive partition of the R1-wrong tasks crossing blind Stage-1 gold
  emergence with Writer repair (with stable/temporary repair subdivided).
  Panel (b) counts sum to n_R1_wrong. The DeepSeek-V4-Flash x HARP-50 row
  is tinted and marked with a star.
- **fig05_continuation_landscape.png** - one point per model x benchmark:
  R1 error rate versus Ever-repair rate, with constant-Oracle-headroom
  contours (headroom = error headroom x recoverability) and Wilson 95%
  confidence intervals for P(EverRepair | R1 wrong). No regression lines.
  The DeepSeek-V4-Flash x HARP-50 point is drawn with a black ring and a
  star marker.

## Split conventions

- `fig02` uses the held-out test split only, per the policy-comparison rule.
- `fig01`, `fig03`, `fig04`, and `fig05` use **all tasks** of each run
  (train + validation + test, n = 50) and are labeled as such in the figures.
- `figure_data.json` stores both an `all` record and a `test` record per
  condition, so every plotted value can be audited against its split.

## Metric definitions

- **R1..R5 accuracy** - share of tasks whose Writer checkpoint at that round
  matches the gold option under binary exact-option scoring (percent).
- **R1 wrong count** - number of tasks incorrect at round 1.
- **Ever-repair** - P(any of R2..R5 correct | R1 wrong).
- **Stable-repair** - repaired after R1 and still correct at R5.
- **Temporary-repair** - correct at some later round but wrong again at R5.
- **Late-repair** - first correct checkpoint appears at R5.
- **First repair round** - the first round in R2..R5 with a correct
  checkpoint, per repaired task (distribution recorded).
- **Blind Stage-1 gold emergence** - P(Planner/Analyst/Critic Stage-1
  candidate_answer equals gold at R2..R5 | R1 wrong), using structured
  candidate answers only.
- **Mechanism taxonomy (Figure 4)** - every R1-wrong task is assigned to
  exactly one cell crossing two independent trajectory facts: blind Stage-1
  gold emergence (yes/no) and Writer repair (yes/no). The four cells are
  "gold never emerged, no repair", "gold never emerged, repaired",
  "gold emerged, no repair", and "gold emerged, repaired"; repaired cells are
  subdivided into stable repair (correct at R5) and temporary repair (wrong
  again at R5). Counts always sum to n_R1_wrong, and a Writer can repair even
  when blind Stage-1 gold never emerged.
- **Wilson 95% CI (Figure 5)** - Wilson score interval for
  P(EverRepair | R1 wrong), using n_R1_wrong as the denominator.
- **Oracle headroom** - Oracle test accuracy minus R1 test accuracy (pp).
- **Captured gain** - RoundValue test accuracy minus R1 test accuracy (pp).
- **Oracle regret** - Oracle test accuracy minus RoundValue test accuracy (pp).

## Same-split invariant

For the policy comparison R1 / RoundValue / Oracle, all values come from the
same split (test) and identical task IDs:

```text
task_ids(R1) == task_ids(RoundValue) == task_ids(Oracle)
```

This is verified per run and must also equal the test split in `scores.json`.
The script fails loudly if it is violated. Under binary exact-option scoring
and per-task best-round Oracle, `Oracle headroom = R1 error rate x
Ever-repair rate` is additionally verified within floating-point tolerance and
stored in `figure_data.json` under `validation`. The same validation section
also verifies that Figure 4 categories sum to `n_R1_wrong`, that the Wilson
intervals use the `n_R1_wrong` denominators, and that the raw Writer
checkpoints and `analysis.json` independently reproduce the `scores.json`
round accuracies.

## Small-sample limitations

Each run has 50 tasks (10 test). Test-split percentages
therefore move in coarse percentage-point steps, and repair rates are
conditioned on small R1-wrong denominators. The mechanism figures use all
tasks of each run to mitigate this, but condition-level rates remain noisy;
treat point estimates as exploratory and report n where the figures do.

## Adding future runs

Collect a new run with Steps 1-2 (e.g. GPT-5-nano x HARP-50). Step 4
auto-discovers it from `results/<run_id>/manifest.json`,
`results/<run_id>/scores.json`, `results/<run_id>/test_policy_replay.json`,
and `trajectories/<run_id>/`; model, dataset, and split are resolved from
those artifacts, not from directory names. Re-run
`python scripts/step4_paper_figures.py` and every figure, `figure_data.json`,
and this README update without rewriting any plotting logic.

## Token / latency / cost

Token, latency, and cost remain Step-3 diagnostics and are intentionally
absent from these primary paper figures; the existing Step 3 plots are
unchanged.

## Validation results

- policy_score_consistency (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- oracle_decomposition (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- same_split_task_ids (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- mechanism_partition_consistency (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- wilson_ci_denominators (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- trajectory_scores_consistency (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- analysis_scores_consistency (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
- policy_score_consistency (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- oracle_decomposition (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- same_split_task_ids (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- mechanism_partition_consistency (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- wilson_ci_denominators (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- trajectory_scores_consistency (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- analysis_scores_consistency (202608161937_deepseek-v4-flash_HARP50_e8bbd5fd): PASS
- policy_score_consistency (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- oracle_decomposition (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- same_split_task_ids (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- mechanism_partition_consistency (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- wilson_ci_denominators (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- trajectory_scores_consistency (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- analysis_scores_consistency (202608161940_deepseek-v4-flash_LogiQA50_db682c1c): PASS
- policy_score_consistency (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- oracle_decomposition (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- same_split_task_ids (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- mechanism_partition_consistency (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- wilson_ci_denominators (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- trajectory_scores_consistency (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- analysis_scores_consistency (202608171054_gpt-5-nano_HARP50_42a8def4): PASS
- policy_score_consistency (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- oracle_decomposition (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- same_split_task_ids (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- mechanism_partition_consistency (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- wilson_ci_denominators (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- trajectory_scores_consistency (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- analysis_scores_consistency (202608171724_gpt-4o-mini_HARP50_71642e0e): PASS
- deepseek_mmlu_run_identity (202608161758_deepseek-v4-flash_MMLUPro50_5a6c654d): PASS
