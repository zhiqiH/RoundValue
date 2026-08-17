# HARP benchmarks

`HARP-500.json` and `HARP-50.json` are deterministic RoundValue benchmark
assets derived from the official HARP multiple-choice dataset
(`HARP_mcq.jsonl.zip`).

- `HARP-500`: 500 tasks, stratified by difficulty `level x subject` and split
  300/100/100 (train/validation/test).
- `HARP-50`: a strict 30/10/10 subset of `HARP-500`, reconstructed
  byte-for-byte from its parent.

Rebuild deterministically from the pinned upstream commit:

```bash
python src/build_harp.py
python src/build_harp_50.py
python src/verify_real_benchmarks.py
```

## Source and license

Upstream project: <https://github.com/aadityasingh/HARP> (pinned commit
`dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2`).

Paper: *HARP: A challenging human-annotated math reasoning benchmark*
(arXiv:2412.08819).

HARP is Copyright (c) 2024 Aaditya Singh and is released under the MIT
License. See the upstream repository `LICENSE` file for the full text.

Only the official multiple-choice asset is used. Human `solution_*` fields,
gold answers, and source identifiers are offline-only metadata and never
reach Agent-facing task views.
