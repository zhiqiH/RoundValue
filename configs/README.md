# configs/

三份实验配置，所有实验参数都集中在这里。

- `agents.json`：四个 Debate 角色（Planner/Analyst/Critic/Writer）与独立求解器
  `single_solver` 的提示词与输出字段
- `model_config.json`：Provider、模型、采样、重试、价格与密钥位置
- `topology.json`：两个可选的具名拓扑 `debate`（缺省）与 `single`

模型与拓扑是彼此独立的 run-level 选择：

```text
roundvalue smoke|run --model-id deepseek_flash|gpt5_nano|gpt4o_mini
                     --topology debate|single
```

缺省为 `deepseek_flash` + `debate`，因此省略两个参数时行为与既有命令完全一致。
`debate` 的节点、边、packet、可见性与五轮上限保持冻结；`single` 是每任务一次逻辑
调用的独立求解器，不构造 Planner/Analyst/Critic、debate packet、历史 transcript
或任何多轮标签。

修改任一配置后需重跑 smoke，否则 collect 的门禁会拒绝开始。
