# LogiQA 2.0 English MRC benchmarks

`LogiQA-500.json` and `LogiQA-50.json` are deterministic RoundValue benchmark
assets derived from the **LogiQA 2.0 English MRC** files of the official
repository (`logiqa/DATA/LOGIQA/{train,dev,test}.txt`). This is not LogiQA
v1 and not the LogiQA 2.0 NLI conversion.

- `LogiQA-500`: 300 tasks from official train, 100 from official dev, and
  100 from official test. Upstream split boundaries are preserved verbatim as
  RoundValue train/validation/test.
- `LogiQA-50`: a strict 30/10/10 subset of `LogiQA-500`, reconstructed
  byte-for-byte from its parent without crossing an official split boundary.

Rebuild deterministically from the pinned upstream commit:

```bash
python src/build_logiqa.py
python src/build_logiqa_50.py
python src/verify_real_benchmarks.py
```

## Source and license

Upstream project: <https://github.com/csitfun/LogiQA2.0> (pinned commit
`955e1d3df6c59d9bfb44d9913da1e1a27ec14e18`).

Paper: *LogiQA 2.0 — An Improved Dataset for Logical Reasoning in Natural
Language Understanding*, Liu et al., IEEE/ACM Transactions on Audio, Speech,
and Language Processing 31:2947–2962 (2023), doi:10.1109/TASLP.2023.3293046.

LogiQA 2.0 is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License
(CC BY-NC-SA 4.0), as declared in the upstream repository README. See
<https://creativecommons.org/licenses/by-nc-sa/4.0/> for the full license
terms.

Upstream reasoning-type annotation strings (including their original
spelling) are preserved as offline audit metadata. Gold answers and source
identifiers never reach Agent-facing task views.
