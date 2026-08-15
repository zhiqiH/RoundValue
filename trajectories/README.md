# trajectories/

`step2_collect` 保存的原始模型轨迹：每次 API 调用尝试、节点输出、Writer checkpoint、token、延迟、重试与错误。

- 目录名：`YYYYMMDDHHMMSS_<数据集>_<hex>`
- 是原始记录，后续离线阶段不回写
- `results/` 可由这里完全重建（重跑 step3）
