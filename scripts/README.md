# scripts/

三个顺序用户入口是本目录仅有的实验步骤：

1. `step1_smoke.py`：小规模真实 API 验收
2. `step2_run.py`：跑主实验——收集单个数据集的 Debate 轨迹与单智能体基线，全部完成后立即做离线评分、标签、策略拟合与评估
3. `step3_visualize.py`：渲染 CSV、HTML/SVG 报告、PNG 图表与结论

也可用 `roundvalue smoke|run|visualize` 等价转发调用。

`dev_selfcheck.py`、`dev_model_selfcheck.py`、`dev_single_selfcheck.py`、
`dev_naming_selfcheck.py` 与 `dev_benchmark_selfcheck.py` 是纯离线自检工具，
不属于实验步骤。
