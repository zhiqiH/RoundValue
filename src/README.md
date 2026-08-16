# src/

底层实现与共享编排，采用扁平模块。

- 运行时模块：`config_loader`、`provider`、`debate_runner`、`single_runner`
  （自动单智能体基线）、`single_analysis`（基线聚合与配对诊断）、`scorer`、
  `policy`、`pipeline` 等
- 构建/验证工具：`build_mmlu_pro.py`、`build_mmlu_pro_50.py`、
  `build_real_benchmarks.py`（旧 MATH）、`build_math50.py`（旧 MATH）、
  `verify_real_benchmarks.py`
- `roundvalue_cli.py`：`roundvalue` 控制台命令 → 三个 step 入口的转发

普通实验流程不直接运行这里的文件（构建/验证工具除外）。
