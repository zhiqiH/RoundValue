# RoundValue 实验架构（冻结版）

## 1. 边界与第一性原理

本仓库是论文实验的可执行部分，不保存论文正文、投稿图表或文献。研究对象是：在**固定通信拓扑**下，已完成一轮后是否值得继续一轮完整 Debate。配置、基准、轨迹和结果均为 JSON；没有伪造 Provider、假结果或虚拟运行逻辑。

一次运行只允许在完整 Writer checkpoint 后输出 `STOP` 或 `CONTINUE`，最多五轮：第 1–4 轮之后可以继续，第 5 轮是最终轮。

## 2. 冻结的一个 Debate 回合

```text
P1 / A1 / C1 并行
      │（题目 + 上一轮 Writer checkpoint {answer, reasoning_summary}）
      ▼
D1 = 确定性 JSON packet [P1, A1, C1]
      ├───────────────┬───────────────┐
      ▼               ▼               ▼
     P2              A2              C2       （并行；全部读取完整 D1）
      \               │               /
       └── W-packet = [P1,A1,C1,P2,A2,C2] ──► Writer ──► {answer, reasoning_summary}
```

`P/A/C/W` 分别表示 Planner、Analyst、Critic、Writer。`D1` 与 `W-packet` 是确定性 JSON 聚合节点：不是 Agent、不请求模型、不改变 token 或成本。Writer 看到六条角色输出的固定顺序，输出两个语义分离的字段：`answer` 是唯一可评分答案，`reasoning_summary` 是对外可读的紧凑推理摘要，只作为跨轮通信证据、不参与评分。第 `t` 轮 checkpoint `{answer_t, reasoning_summary_t}` 作为 `previous_writer_checkpoint` 进入第 `t+1` 轮全部七个逻辑节点，构成受控的跨轮信息瓶颈；除该紧凑 checkpoint 外不累积传递完整历史 transcript。

一轮恒为 7 次逻辑模型调用：`P1,A1,C1,P2,A2,C2,W`。网络重试与格式修复重试都属于附属 API 尝试，完整保存在轨迹中并另行累积；它们不改变每轮的逻辑调用数。格式修复是有界的（`agents.json` 的 `format_retries`），把具体违规反馈给模型后重新采样同一节点，绝不静默改写字段；耗尽后该节点如实失败。P/A/C 同阶段并行，阶段 2 必须等待完整 D1，Writer 必须等待完整 W-packet。角色、边、可见性、顺序和最大轮数在运行中均不可变。

## 3. 可见性与防泄漏

| 位置 | 可读取信息 |
|---|---|
| P1/A1/C1 | 公开题面、上一轮 Writer checkpoint（answer + reasoning_summary） |
| P2/A2/C2 | 上述信息、完整 D1 |
| Writer | 上述信息、完整 W-packet |
| 在线停止策略 | 题目、当前答案、可见消息、公开 verifier、已用预算 |
| 离线评分/标签 | 参考答案、隐藏测试与离线 Judge（不可回流在线） |

所有角色必须返回严格 JSON。节点可见输出预算由角色 output_schema 推导；非思考模式下它逐节点限制 `max_output_tokens`，思考模式下它是 prompt 层目标、API 上限改为模型 `max_output_tokens`（因为 DeepSeek 的 `max_tokens` 计入隐藏 reasoning token）；非法 JSON、`finish_reason=length` 截断或缺失/空字段会被检测，进入带具体违规反馈的验证-修复重试，每次尝试都被记录，非最终修复的可见预算逐级减半。最后一次修复是确定性的 **answer-only 回退**：只请求模型返回最终答案，runner 用自描述占位符补全其余字段并在轨迹 `fallback` 字段记录这次降级；Writer 回退时 `answer` 始终是模型给出的真实答案，`reasoning_summary` 被显式占位。字段的 `max_length` 是 prompt 层软目标：模型无法可靠地数字符，因此少量超出不视为致命错误，也不静默裁剪。回退仍拿不到可用答案时节点如实失败；不存在静默修正。未知 token、缓存计数、费用和延迟保持未知，不可用零填充。

Agent 可见的 `task_id` 是原 ID 的确定性匿名哈希；`public_metadata` 会剔除 `source_task_id` 与 `base_input_count`/`plus_input_count` 等可识别具体上游题目或暴露隐藏测试规模的信息。MMLU-Pro 的题目与全部选项已内嵌在公开 `prompt` 中，而 `answer_index`、`reference_answer`、原 ID 与完整标签只保存在磁盘记录和离线评分中。

## 4. 实现边界

```text
benchmark/mmlu_pro/MMLU-Pro-500.json   主实验基准（300/100/100），provenance 内嵌
benchmark/mmlu_pro/MMLU-Pro-50.json    MMLU-Pro-500 分层验证子集（30/10/10），provenance 内嵌
benchmark/math/MATH-500.json           旧数学基准，仅保留用于追溯与回归验证
benchmark/math/MATH-50.json            旧 MATH-500 子集，仅保留用于追溯与回归验证
benchmark/test/                   独立仓库验收题；不来自主基准且不用于论文结果
configs/                          agents.json, model_config.json, topology.json
scripts/step1_smoke.py            三个顺序用户入口（dev_*.py 为纯离线自检）
scripts/step2_run.py
scripts/step3_visualize.py
pyproject.toml                    运行依赖管理 + roundvalue 控制台命令注册
src/                              扁平模块 + pipeline 共享编排 + benchmark 构建/验证工具
src/roundvalue_cli.py             roundvalue 子命令 → 三个 step 入口的等价转发
trajectories/YYYYMMDDHHMM_<数据集>_<hex>/    任务级完整调用与 checkpoint 记录
results/YYYYMMDDHHMM_<数据集>_<hex>/         聚合指标、置信区间、策略报告与 manifest
.secret/model_key.json            本地密钥，永不提交
```

三份配置同时校验。`agents.json` 固定 Debate 角色提示词和字段；`topology.json` 只保留唯一且无版本号的 `debate` 通信流；`model_config.json` 只保留唯一的 `deepseek_flash` profile。Debate 模块是一个不可选的固定整体：运行时不接受模型或拓扑选择参数，任何变更都会通过 config/源码哈希与 smoke gate 强制重跑验收。默认使用 `deepseek-v4-flash`、`temperature: 0.2`、`max_output_tokens: 32768`，适配器显式发送 `thinking: {"type":"enabled"}` 与 `reasoning_effort: "high"`。

正式基准是两个 MMLU-Pro 数据集文件，每个文件内部按统一比例 60/20/20、以固定种子
确定性划分，余数计入 Test：MMLU-Pro-500（500 → 300/100/100）与
MMLU-Pro-50（50 → 30/10/10）。两者都从钉死的 MMLU-Pro `test` 分片按
category/src 分层确定性选取，不按任何实验结果筛选；评分只比较保守归一化后的
规范选项字母与金标准字母，`src/scorer.py` 是唯一的评分来源。
每个 run 通过 `--benchmark <数据集 json>` 只选取一个数据集，Train/Validation/Test
绝不跨数据集混用。每个数据集文件内嵌的 `provenance`
字段冻结来源 revision、上游校验信息、原始记录 hash、选取与划分种子、比例和全部测试 task ID；
collect 时整个文件（含 provenance）都会进入 run 的 benchmark 快照。

## 5. 实验链路

1. `step1_smoke.py`：运行独立仓库验收题、真实 API、每题完整一轮；验证密钥、配置、DAG、Writer JSON、评分、预期分数与落盘。任一题失败即停止，Smoke 数据不进论文结果。
2. `step2_run.py`：用 `--benchmark` 选择单个数据集文件，校验 smoke 通过后，只在该数据集内按原始 `task_id` 冻结 Train/Validation/Test，收集每题至多五轮原始轨迹；全部完成后在同一命令内继续完全离线评分、构建 ΔQ/V/G、Train 拟合、Validation 选阈值、Test 评估，`results/` 的产物仍只能由 trajectories 重建，收集不完整则跳过分析。
3. `step3_visualize.py`：只读 `results/` 渲染 CSV、HTML/SVG 报告、5 张 PNG 图表与简短结论。

`roundvalue smoke|run|visualize` 是这三个入口的等价转发命令，参数、门禁与退出码
完全一致，不构成第五个实验步骤；`python scripts/step*_*.py` 形式保持不变。

每个 run 保存配置快照和哈希、基准来源与哈希、Git 状态、源代码快照、命令行、模型响应名、API 尝试、checkpoint、评分、特征、标签和聚合结果。服务端即便在温度为零时仍可能变化，因此以任务级轨迹和配对 bootstrap 报告不确定性，不宣称逐 token 完全确定。

历史产物按历史语义处理，不静默迁移：旧的 3 轮 answer-only 轨迹
（checkpoint 字段为 `final_answer`、无 `reasoning_summary`）仍可离线评分与重放，
只生成 Fixed-1/2/3 比较；不会为其补造推理摘要。新 run 的 checkpoint 一律为
`answer + reasoning_summary`，旧新格式只在兼容读取层相遇，不混入同一 run。

## 6. 论文结论的最低证据

比较 Fixed-1/2/3/4/5、task-only、RoundValue 和 trajectory Oracle。默认 `λ_cost=μ_latency=0`，因此值函数 `G` 是纯质量收益、阈值选择只在质量相同时以更少 token 决胜；质量—成本与质量—延迟 Pareto、任务级配对置信区间、Oracle regret 与 Repair/Neutral/Harm/Recovery 作为独立坐标报告。Oracle 覆盖全部五个 checkpoint，只测量可达上界与后悔值，绝不可部署。
