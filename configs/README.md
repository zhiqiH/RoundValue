# configs/

三份实验配置，所有实验参数都集中在这里。

- `agents.json`：四个 Debate 角色（Planner/Analyst/Critic/Writer）与独立求解器
  `single_solver` 的提示词与输出字段（`single_solver` 是基线角色，不是拓扑）
- `model_config.json`：Provider、模型、采样、重试、价格与密钥位置
- `topology.json`：唯一冻结的 `debate` 拓扑（单智能体基线不需要拓扑条目）

run-level 只选择模型，拓扑与单智能体基线都固定：

```text
roundvalue smoke|run --model-id deepseek_flash|gpt5_nano|gpt4o_mini
```

缺省为 `deepseek_flash`。`debate` 的节点、边、packet、可见性与五轮上限保持冻结；
每次实验自动收集一个单智能体基线（每任务一次独立的 `single_solver` 逻辑调用），
它不与 Debate 共用任何可见信息，也不进入任何多轮 RoundValue 概念。

修改任一配置后需重跑 smoke，否则 collect 的门禁会拒绝开始。
