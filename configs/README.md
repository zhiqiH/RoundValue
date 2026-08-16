# configs/

三份实验配置，所有实验参数都集中在这里。

- `agents.json`：四个 Debate 角色的提示词与输出字段
- `model_config.json`：Provider、模型、采样、重试、价格与密钥位置
- `topology.json`：拓扑注册表（节点、边、packet、轮数）

模型是唯一 run-level 选择：`roundvalue smoke|run --model-id deepseek_flash|gpt5_nano`
（缺省 `deepseek_flash`）。

修改任一配置后需重跑 smoke，否则 collect 的门禁会拒绝开始。
