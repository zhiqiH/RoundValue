# RoundValue

RoundValue 是一套为独立论文实验准备的、精简且可复现的固定 Debate 实验代码。仓库只保存代码、配置、基准、轨迹和结果；论文正文、投稿图表与文献在仓库外单独维护。

用户配置密钥后按顺序运行四个入口：

```powershell
python -m pip install httpx   # 运行时依赖；代码评分还需 numpy
python scripts/step1_smoke.py
python scripts/step2_collect.py --benchmark benchmark/formal_experiment_v1.json \
  --smoke-run-id <SMOKE_RUN_ID> --allow-local-code-evaluation
python scripts/step3_analyze.py --run-id <RUN_ID> --allow-local-code-evaluation
python scripts/step4_visualize.py --run-id <RUN_ID>
```

## 配置密钥

在仓库根目录创建 `.secret/model_key.json`：

```json
{
  "deepseek": {
    "api_key": "填入真实 API key"
  }
}
```

也可使用环境变量 `DEEPSEEK_API_KEY`。`.secret/` 已被忽略，密钥不会写入轨迹、结果或运行日志。`smoke` 是一次真实 API 冒烟实验：若密钥、网络、模型响应、JSON 输出或评分失败，命令会失败；不会退化为假调用。

## 冻结的 Debate 协议

一轮就是下列完整 DAG，四种角色固定为 Planner、Analyst、Critic、Writer：

```text
P1 / A1 / C1（并行，读取题目与上一轮 Writer checkpoint）
                     ↓
       D1 = [P1, A1, C1] 的确定性 JSON packet
                     ↓
P2 / A2 / C2（并行，三者均读取完整 D1）
                     ↓
 W-packet = [P1, A1, C1, P2, A2, C2]（固定顺序）
                     ↓
                 Writer → final_answer
```

`D1` 和 `W-packet` 只是确定性 JSON 拼接，不是额外 Agent，也不产生模型调用。每轮固定有 7 次**逻辑模型调用**，至多 3 轮；网络重试是附属 API 尝试，单独记录，不改变“7 次逻辑调用/轮”的定义。只有 Writer 的 `final_answer` 是可评分 checkpoint。格式错误会使该轮失败，不能静默修补输出。

策略只在某个完整 checkpoint 后决定 `STOP` 或 `CONTINUE`。它只能读取题目、已可见消息、公开 verifier 信号和已用预算；参考答案、隐藏测试、未来轮输出及离线 Judge 信息只用于离线评分和标签构建。

## 三份实验配置

所有实验参数均集中在 JSON 文件中：

| 文件 | 作用 |
|---|---|
| `configs/agents.json` | 四个 Debate 角色的内嵌提示词与输出字段 |
| `configs/model_config.json` | Provider、模型、采样、重试、价格和密钥位置 |
| `configs/topology.json` | 拓扑注册表：选择当前拓扑，并定义其节点、边、packet 与轮数 |

`topology.json` 的 `topologies` 是按 ID 索引的注册表。当前仅有 `debate_pac_v1`；运行时可用 `--topology-id <ID>` 选择。每个拓扑条目只保留 `runner`、`max_rounds`、`nodes`、`packets`、`edges`：前四者定义调用与确定性汇聚，`edges` 定义可见通信路径。新增拓扑还必须新增对应的 Python runner 与校验器；未实现的 runner 会在调用 API 前明确报错。

默认 profile 是 `deepseek_flash`，请求模型 `deepseek-v4-flash`，`temperature` 固定为 `0.0`。DeepSeek 适配器会显式关闭思考模式（发送 `thinking: {"type": "disabled"}`），并要求响应中不存在 reasoning 内容。Provider 接口与 Debate 执行器分离；后续接入其他公司模型时增加适配器与 JSON profile，不改变拓扑、评分或策略。

角色提示词保存在 `agents.json`，不使用散落的 prompt 文件。模型价格采用服务端返回的 token 明细计算；缺失 token、缓存计数、成本或延迟时保存为“未知”，绝不按零处理。

## 四个执行步骤

| 脚本 | 职责 | 模型 API | 写入位置 |
|---|---|---|---|
| `step1_smoke.py` | 小规模通路测试：配置、完整一轮 Debate、数学/代码评分、token、延迟、落盘 | 是 | 独立 smoke run（split 恒为 `smoke`） |
| `step2_collect.py` | 只收集第 1/2/3 轮原始轨迹（节点输出、Writer checkpoint、token、延迟、重试、错误） | 是 | `trajectories/<run_id>/` |
| `step3_analyze.py` | 完全离线：检查完整性、逐轮评分、`ΔQ/V/G`、Train 拟合、Validation 选阈值、Test 评估 | 否 | `results/<run_id>/` |
| `step4_visualize.py` | 只读 `results`，输出 CSV、自包含 HTML/SVG 图表与简短结论 | 否 | `results/<run_id>/` |

执行顺序是强制的：

1. `step1_smoke` 的每道验收题都必须得到分数 1，任一失败会以非零退出码结束，因此不能进入正式收集。Smoke 数据永远使用独立 run 与 `smoke` split，不进入论文结果。
2. `step2_collect` 必须用 `--smoke-run-id` 指向一个已通过的 smoke run，并校验该 smoke 已全部通过、且三份 config 与模型/拓扑选择未发生变化；不满足时拒绝开始。它只保存原始轨迹，不训练策略、不生成结论。给定已存在的 `--run-id` 时进入断点续跑，只重跑失败或缺失任务；失败任务 ID 输出到终端，原因写入 `results/<run>/failure_details.json`。
3. `step3_analyze` 只读 trajectories，用确定性离线评分器对每一轮评分（代码任务需 `--allow-local-code-evaluation`），构建 `ΔQ/V/G` 标签，只在 Train 拟合、只在 Validation 选阈值、只在 Test 评估，并把 `scores.json`、`labels.json`、`policy.json`、`test_policy_replay.json`、`analysis.json` 与汇总写入 results。它绝不调用模型 API，也绝不回写 trajectories。
4. `step4_visualize` 只读 `results/<run_id>/analysis.json` 与 manifest，生成 `task_level_results.csv`、`report.html`（内嵌每轮准确率、token、wall-clock、R/N/H/R、停止轮次、策略对比及质量—token/质量—延迟 SVG 图）与 `summary_conclusion.txt`。可视化不读取 trajectories，不可能反向影响评分或策略。

因此 `trajectories/` 是原始且尽量不被后续阶段修改的模型轨迹；`results/` 必须能由 trajectories 完全离线重建（重跑 step3）。`run_id` 是 step3/step4 唯一接受的实验标识，不要混用不同 run。代码任务的本地执行必须显式允许，并在去除密钥的临时子进程中进行；它不替代专用隔离执行环境。

延迟统计保留两个字段：`wall_clock_ms` 是真实的墙钟等待时间（并行的 P/A/C 请求只计一次），`api_latency_ms` 是全部 API 尝试的服务时间总和（含重试）。离线标签、报告图表与策略的时间成本使用墙钟；旧轨迹只有服务时间总和时按未知处理，不用服务时间冒充墙钟。

### 防泄漏与 Agent 可见信息

Agent 只能看到 `task_id`、`domain`、`prompt` 以及少量明确允许的公开元数据。其中 `task_id` 在送入 Agent 前会被替换为确定性的匿名哈希（原始 ID 只保留在磁盘记录中），并且会剔除 `source_task_id`、`base_input_count`、`plus_input_count` 等能暴露具体上游题号或隐藏测试规模的信息。参考答案、隐藏测试与离线 Judge 只用于离线评分和标签构建。

### 本地代码执行

所有本地代码评测路径对模型生成的候选代码施加同一套防护：受限 builtins（`eval` 仅限算术表达式，`exec`/`compile`/`open`/`__import__` 不可达）、禁止下划线属性访问、受限语法与模块白名单（`sys` 通过只读代理暴露）；官方 canonical oracle 和受信测试程序不受该白名单限制。候选进程仍使用去除密钥的环境和临时工作目录，但这只是纵深防御，不是操作系统级沙箱。

## Benchmark 边界

`benchmark/test/` 是仓库独立验收题，只用于检查 API、DAG、JSON 输出、评分与落盘是否正常；它不从论文基准抽样，也不得出现在训练、验证、测试或论文结果中。`smoke` 默认运行其中的数学题；加 `--allow-local-code-evaluation` 才会同时运行代码验收题，因为这会执行模型生成的代码。

论文主实验只使用 `benchmark/math/dataset_registry.json` 登记的 **MATH-100**，以及 `benchmark/code/dataset_registry.json` 登记的 **EvalPlus HumanEval+ / MBPP+**。registry 是来源和评测协议的清单；可直接运行的冻结任务集合是 `benchmark/formal_experiment_v1.json`。

## 正式真实数据基准（v1）

`benchmark/formal_experiment_v1.json` 固定包含 842 道真实题：

| Split | 数据 | 题数 | 用途 |
|---|---:|---:|---|
| Train | MATH 开发题 | 140 | 拟合策略 |
| Validation | MATH 开发题 | 60 | 选择阈值和偏好 |
| Test | MATH-100（来自 MATH-500） | 100 | 数学测试 |
| Test | EvalPlus HumanEval+ v0.1.10 | 164 | 代码测试 |
| Test | EvalPlus MBPP+ v0.2.0 | 378 | 代码测试 |

MATH-100 是从 MATH-500 以固定种子进行分层抽样得到的 100 道测试题。200 道数学开发题来自完整 MATH 镜像，并在抽样前排除了全部 MATH-500 题目；因此开发题与数学测试题不重叠。542 道代码题全部冻结为 Test：它们用于检验从数学开发题拟合出的跨领域策略，**不应表述为 EvalPlus 的训练/验证划分**。

代码评分器 `evalplus_differential_v1` 使用官方 EvalPlus 发布的 base/plus 输入、canonical oracle 与容差字段进行差分比较；题面之外的测试输入和 oracle 不会提供给 Agent。它是 RoundValue 的可复现适配器，**不是**官方 `evalplus.evaluate`、官方容器或 leaderboard 成绩；正式报告应标注为“RoundValue EvalPlus differential adapter”。

运行正式收集：

```powershell
python scripts/step2_collect.py --benchmark benchmark/formal_experiment_v1.json \
  --smoke-run-id <SMOKE_RUN_ID> --allow-local-code-evaluation
```

每次运行还会冻结任务文件哈希。`benchmark/formal_experiment_v1.provenance.json` 固定记录生成器版本、MATH 的 Hugging Face revision、EvalPlus 官方 release URL 与 artifact SHA-256、原始记录 SHA-256、MATH-100 抽样种子和全部测试 task ID。若需从已固定来源重新构建数据，先 `python -m pip install 'datasets>=3,<5'`，再运行 `python src/build_real_benchmarks.py` 和 `python src/verify_real_benchmarks.py`。

## 目录与产物

```text
RoundValue/
├── benchmark/{code,math,test}/
├── configs/{agents.json,model_config.json,topology.json}
├── .secret/model_key.json              # 仅本地存在
├── results/YYYY-MM-DD_HHMMSS_<run_id>/
├── scripts/step1_smoke.py              # scripts/ 只含这四个用户入口
├── scripts/step2_collect.py
├── scripts/step3_analyze.py
├── scripts/step4_visualize.py
├── src/                                # 底层模块 + pipeline 编排 + benchmark 构建/验证工具
├── trajectories/YYYYMMDD_HHMMSS_<run_id>/
├── EXPERIMENT_ARCHITECTURE.md
└── README.md
```

每个 run 冻结三份配置、基准来源与哈希、命令行、模型 profile、Git 状态、源代码快照。`trajectories/` 保存任务级原始记录（调用尝试、原始响应、checkpoint、token、延迟、重试与错误）；`results/` 保存由 step3 离线派生的评分、标签、策略、比较与汇总，可由 trajectories 重建。二者是可追溯的实验产物，默认可纳入 Git；提交前仍应确认不含密钥、个人数据或不应公开的模型输出。

正式报告应比较 Fixed-1/2/3、启发式（共识）、task-only、RoundValue、one-step Oracle 与 trajectory Oracle，并报告质量—成本 Pareto、任务级置信区间、Oracle regret、Repair/Neutral/Harm/Recovery。共识信号要求 Planner/Analyst/Critic 六条输出中至少 2/3 的 `candidate_answer` 完全一致，或 Writer 答案跨轮稳定。Oracle 仅用于诊断上界，绝不用于部署策略。
