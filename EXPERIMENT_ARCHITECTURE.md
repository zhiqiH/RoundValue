# RoundValue 实验架构（冻结版）

## 1. 边界与第一性原理

本仓库是论文实验的可执行部分，不保存论文正文、投稿图表或文献。研究对象是：在**固定通信拓扑**下，已完成一轮后是否值得继续一轮完整 Debate。配置、基准、轨迹和结果均为 JSON；没有伪造 Provider、假结果或虚拟运行逻辑。

一次运行只允许在完整 Writer checkpoint 后输出 `STOP` 或 `CONTINUE`，最多三轮。

## 2. 冻结的一个 Debate 回合

```text
P1 / A1 / C1 并行
      │（题目 + 上一轮 Writer checkpoint）
      ▼
D1 = 确定性 JSON packet [P1, A1, C1]
      ├───────────────┬───────────────┐
      ▼               ▼               ▼
     P2              A2              C2       （并行；全部读取完整 D1）
      \               │               /
       └── W-packet = [P1,A1,C1,P2,A2,C2] ──► Writer ──► final_answer
```

`P/A/C/W` 分别表示 Planner、Analyst、Critic、Writer。`D1` 与 `W-packet` 是确定性 JSON 聚合节点：不是 Agent、不请求模型、不改变 token 或成本。Writer 看到六条角色输出的固定顺序，且其 `final_answer` 是唯一可评分答案。

一轮恒为 7 次逻辑模型调用：`P1,A1,C1,P2,A2,C2,W`。网络重试属于附属 API 尝试，完整保存在轨迹中并另行累积；它不改变每轮的逻辑调用数。P/A/C 同阶段并行，阶段 2 必须等待完整 D1，Writer 必须等待完整 W-packet。角色、边、可见性、顺序和最大轮数在运行中均不可变。

## 3. 可见性与防泄漏

| 位置 | 可读取信息 |
|---|---|
| P1/A1/C1 | 公开题面、上一轮 Writer checkpoint |
| P2/A2/C2 | 上述信息、完整 D1 |
| Writer | 上述信息、完整 W-packet |
| 在线停止策略 | 题目、当前答案、可见消息、公开 verifier、已用预算 |
| 离线评分/标签 | 参考答案、隐藏测试与离线 Judge（不可回流在线） |

所有角色必须返回严格 JSON。格式错误会失败并如实记录；不得静默修正。未知 token、缓存计数、费用和延迟保持未知，不可用零填充。

Agent 可见的 `task_id` 是原 ID 的确定性匿名哈希；`public_metadata` 会剔除 `source_task_id` 与 `base_input_count`/`plus_input_count` 等可识别具体上游题目或暴露隐藏测试规模的信息。原 ID 与完整标签只保存在磁盘记录和离线评分中。

## 4. 实现边界

```text
benchmark/math/MATH-500.json      MATH-500 独立任务文件（300/100/100），provenance 内嵌
benchmark/code/HumanEvalPlus.json HumanEval+ 独立任务文件（98/32/34），provenance 内嵌
benchmark/code/MBPPPlus.json      MBPP+ 独立任务文件（226/75/77），provenance 内嵌
benchmark/test/                   独立仓库验收题；不来自主基准且不用于论文结果
configs/                          agents.json, model_config.json, topology.json
scripts/step1_smoke.py            四个顺序用户入口（scripts/ 只有这四个 Python 入口）
scripts/step2_collect.py
scripts/step3_analyze.py
scripts/step4_visualize.py
pyproject.toml                    运行依赖管理 + roundvalue 控制台命令注册
src/                              扁平模块 + pipeline 共享编排 + benchmark 构建/验证工具
src/roundvalue_cli.py             roundvalue 子命令 → 四个 step 入口的等价转发
trajectories/YYYYMMDDHHMMSS_<数据集>_<hex>/    任务级完整调用与 checkpoint 记录
results/YYYYMMDDHHMMSS_<数据集>_<hex>/         聚合指标、置信区间、策略报告与 manifest
.secret/model_key.json            本地密钥，永不提交
```

三份配置同时校验。`agents.json` 固定 Debate 角色提示词和字段；`topology.json` 是拓扑注册表，当前选择并冻结上述 Debate DAG；`model_config.json` 冻结模型与运行参数。新增拓扑时在 `topologies` 下新增 ID，并同时实现对应 runner 与校验器；不会静默改写现有 Debate。默认 DeepSeek profile 使用 `deepseek-v4-flash`、`temperature: 0.0`，适配器显式发送 `thinking: {"type":"disabled"}`。Provider 是通用接口：新增厂商只增加适配器和 JSON profile，不改变 Debate、评分或策略。

正式基准是三个各自独立的数据集文件，每个文件内部按统一比例 60/20/20、以固定种子
确定性划分，余数计入 Test：MATH-500（500 → 300/100/100）、HumanEval+ v0.1.10
（164 → 98/32/34）、MBPP+ v0.2.0（378 → 226/75/77）。每个 run 通过
`--benchmark <数据集 json>` 只选取一个数据集，Train/Validation/Test 绝不跨数据集或
跨域混用；未来新增数据集时增加一个独立 JSON 即可。每个数据集文件内嵌的 `provenance`
字段冻结来源 revision、上游校验信息、原始记录 hash、划分种子、比例和全部测试 task ID；
collect 时整个文件（含 provenance）都会进入 run 的 benchmark 快照。

代码任务使用 `evalplus_differential_v1`：在去除密钥的本地子进程中，以官方发布的 base/plus 输入、canonical oracle 与容差进行差分评分，并且不向 Agent 暴露测试输入或 oracle。所有本地评测路径对候选代码施加同一套防护：受限 builtins（`eval` 仅限算术表达式，`exec`/`compile`/`open`/`__import__` 不可达）、禁止下划线属性访问、受限语法和模块白名单（`sys` 通过只读代理暴露）；官方 canonical oracle 与受信测试程序不受该白名单限制。该适配器不是官方 EvalPlus runner、容器或 leaderboard；论文中必须将其结果标记为 RoundValue differential-adapter 结果。本地代码执行仍须显式授权；上述防护是纵深防御，不构成操作系统级沙箱。

## 5. 实验链路

1. `step1_smoke.py`：`--domain` 选择验收域（math 默认，code 需 `--allow-local-code-evaluation`），独立验收题、真实 API、每题完整一轮；验证密钥、配置、DAG、Writer JSON、评分、预期分数与落盘。任一题失败即停止，Smoke 数据不进论文结果。
2. `step2_collect.py`：用 `--benchmark` 选择单个数据集文件并从中读出 `dataset_id` 与 `domain`，校验**同域** smoke 通过后，只在该数据集内按原始 `task_id` 冻结 Train/Validation/Test，收集每题至多三轮原始轨迹。
3. `step3_analyze.py`：完全离线评分、构建 ΔQ/V/G、Train 拟合、Validation 选阈值、Test 评估；只写 `results/`，不回写 trajectories。
4. `step4_visualize.py`：只读 `results/` 渲染 CSV、HTML/SVG 图表与简短结论。

`roundvalue smoke|collect|analyze|visualize` 是这四个入口的等价转发命令，参数、门禁与退出码
完全一致，不构成第五个实验步骤；`python scripts/step*_*.py` 形式保持不变。

每个 run 保存配置快照和哈希、基准来源与哈希、Git 状态、源代码快照、命令行、模型响应名、API 尝试、checkpoint、评分、特征、标签和聚合结果。服务端即便在温度为零时仍可能变化，因此以任务级轨迹和聚类 bootstrap 报告不确定性，不宣称逐 token 完全确定。

## 6. 论文结论的最低证据

比较 Fixed-1/2/3、启发式、task-only、RoundValue、one-step Oracle 和 trajectory Oracle。启发式共识信号基于 Planner/Analyst/Critic 六条输出的 `candidate_answer`：至少 2/3 完全一致，或 Writer 答案跨轮稳定，才判定为共识。报告质量—成本 Pareto、任务级置信区间、Oracle regret 与 Repair/Neutral/Harm/Recovery。Oracle 只测量可达上界与后悔值，绝不可部署。
