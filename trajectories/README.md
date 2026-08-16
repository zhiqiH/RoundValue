# trajectories/

`step2_run` 保存的原始模型证据：每个任务同时保存 Debate 轨迹（每次 API 调用尝试、
节点输出、Writer checkpoint、token、延迟、重试与错误）与独立的单智能体基线观测。

- 目录名：`YYYYMMDDHHMM_<模型>_<数据集>_<hex>`（与同名 results 目录精确配对）
- 是原始记录，后续离线阶段不回写
- `results/` 可由这里完全重建（重跑 step3）
