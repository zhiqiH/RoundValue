# scripts/

四个顺序用户入口，本目录只放这四个 Python 文件。

1. `step1_smoke.py`：小规模真实 API 验收
2. `step2_collect.py`：收集单个数据集的原始轨迹
3. `step3_analyze.py`：完全离线评分、标签、策略拟合与评估
4. `step4_visualize.py`：渲染 CSV、HTML/SVG 图表与结论

也可用 `roundvalue smoke|collect|analyze|visualize` 等价转发调用。
