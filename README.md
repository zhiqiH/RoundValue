# RoundValue

RoundValue 是一套为独立论文实验准备的、精简且可复现的固定 Debate 实验代码。仓库只保存代码、配置、基准、轨迹和结果；论文正文、投稿图表与文献在仓库外单独维护。

用户配置密钥后按顺序运行三个入口：

```powershell
conda create -n roundvalue python=3.11 -y   # 首次创建环境；项目要求 Python 3.11+
conda activate roundvalue
python -m pip install -e .                  # 安装 pyproject.toml 声明的依赖并注册 roundvalue 命令
roundvalue smoke
roundvalue collect-analyze --benchmark benchmark/math/MATH-500.json \
  --smoke-run-id <SMOKE_RUN_ID>
roundvalue visualize --run-id <RUN_ID>
```

`python` 必须指向 3.11+ 解释器；macOS 系统自带的 `/usr/bin/python3` 可能是 3.9，
请始终在 `conda activate roundvalue` 后运行以上命令。

`roundvalue` 只是三个 step 脚本的等价转发命令：`roundvalue smoke|collect-analyze|visualize`
与 `python scripts/step*_*.py` 的参数、门禁和退出码完全一致，不新增第四步。原脚本方式
仍然可用。运行依赖由 `pyproject.toml` 管理（`httpx`、`numpy`、`matplotlib`；重建基准另需可选组
`benchmark-build`，即 `datasets>=3,<5`）。项目要求 Python 3.11+。

每个数据集都是独立的自包含基准文件：`collect` 用 `--benchmark <数据集 json>` 选择，
run 命名直接带数据集名称，任何 run 都不会混合两个数据集。本项目只测试数学：
可用数据集是 MATH-500 全集与其验证子集 MATH-50。

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

`D1` 和 `W-packet` 只是确定性 JSON 拼接，不是额外 Agent，也不产生模型调用。每轮固定有 7 次**逻辑模型调用**，至多 3 轮；网络重试是附属 API 尝试，单独记录，不改变“7 次逻辑调用/轮”的定义。只有 Writer 的 `final_answer` 是可评分 checkpoint。节点的输出预算由该角色 output_schema 的字段上限推导（`format_budget_margin` 留余量），这是硬的有界性保证；非法 JSON、截断输出或缺失/空字段会触发有界的验证-修复重试（`format_retries`），每次重试把具体违规反馈给模型并完整记录，非最终修复的预算逐级减半。最后一次修复退化为**只问答案的回退**：只要求模型返回答案本身，runner 再用自描述占位符确定性补全 schema，并在轨迹 `fallback` 字段记录该降级——Writer 的回退不产生占位（`final_answer` 是真实答案），非 Writer 角色只有辅助字段被占位。字段的 `max_length` 是软目标而非致命校验（模型无法精确数字符数），不触发失败。不存在静默修改；回退也无答案时仍如实失败。

策略只在某个完整 checkpoint 后决定 `STOP` 或 `CONTINUE`。它只能读取题目、已可见消息、公开 verifier 信号和已用预算；参考答案、隐藏测试、未来轮输出及离线 Judge 信息只用于离线评分和标签构建。

## 三份实验配置

所有实验参数均集中在 JSON 文件中：

| 文件 | 作用 |
|---|---|
| `configs/agents.json` | 四个 Debate 角色的内嵌提示词、输出字段，以及 `format_retries` / `format_budget_margin` 修复协议 |
| `configs/model_config.json` | 唯一的 Provider/模型 profile（`deepseek_flash`）、采样、重试、价格和密钥位置 |
| `configs/topology.json` | 唯一的 Debate 通信流 `debate`：节点、边、packet 与轮数 |

Debate 模块已被固化为一个整体：`topology.json` 只保留唯一且无版本号的 `debate` 通信流，`runner`、
`max_rounds`、`nodes`、`packets`、`edges` 均不可选，运行时也不再接受模型或拓扑选择参数。

唯一 profile 是 `deepseek_flash`，请求模型 `deepseek-v4-flash`，`temperature` 固定为 `0.0`。DeepSeek 适配器会显式关闭思考模式（发送 `thinking: {"type": "disabled"}`），并要求响应中不存在 reasoning 内容。

角色提示词保存在 `agents.json`，不使用散落的 prompt 文件。模型价格采用服务端返回的 token 明细计算；缺失 token、缓存计数、成本或延迟时保存为“未知”，绝不按零处理。

## 三个执行步骤

| 脚本 | 职责 | 模型 API | 写入位置 |
|---|---|---|---|
| `step1_smoke.py` | 小规模数学通路测试：配置、完整一轮 Debate、评分、token、延迟、落盘 | 是 | 独立 smoke run（split 恒为 `smoke`） |
| `step2_collect_analyze.py` | 收集 `--benchmark` 指定单个数据集的 1/2/3 轮原始轨迹（节点输出、Writer checkpoint、token、延迟、重试、错误）；全部完成后立即检查完整性、逐轮评分、构建 `ΔQ/V/G`、Train 拟合、Validation 选阈值、Test 评估 | 收集阶段是 | `trajectories/<run_id>/` 与 `results/<run_id>/` |
| `step3_visualize.py` | 只读 `results`，输出 CSV、自包含 HTML/SVG 报告、`charts/` 下 5 张 policy-level PNG 图表（质量-token、质量-延迟、RoundValue vs baseline、自适应停止分布、oracle regret）与简短结论 | 否 | `results/<run_id>/` |

执行顺序是强制的：

1. `step1_smoke` 的每道数学验收题都必须得到分数 1，任一失败会以非零退出码结束，因此不能进入正式收集。Smoke 数据永远使用独立 run 与 `smoke` split，不进入论文结果。
2. `step2_collect_analyze` 必须用 `--smoke-run-id` 指向一个已通过的 smoke run，并校验该 smoke 已全部通过、且三份 config 与源码快照未发生变化；不满足时拒绝开始。新 run 自动命名，无需传 `--run-id`；给定一个已存在的 collect `--run-id` 时先断点续跑，只重跑失败或缺失任务。它先保存原始轨迹；若全部完成，再只用 trajectories 做确定性离线评分、构建 `ΔQ/V/G`、Train 拟合、Validation 选阈值、Test 评估，把 `scores.json`、`labels.json`、`policy.json`、`test_policy_replay.json`、`analysis.json` 与汇总写入 results；若收集不完整则跳过分析并输出失败任务 ID（原因在 `results/<run>/failure_details.json`），可用同一命令加 `--run-id` 续跑。分析阶段绝不调用模型 API，也绝不回写 trajectories。
3. `step3_visualize` 只读 `results/<run_id>/analysis.json` 与 manifest，生成 `task_level_results.csv`、`report.html`（内嵌每轮准确率、token、wall-clock、R/N/H/R、停止轮次、策略对比及质量—token/质量—延迟 SVG 图）、`summary_conclusion.txt`，以及 `charts/` 下的 5 张 policy-level PNG 图表（`chart_policy_quality_vs_tokens.png`、`chart_policy_quality_vs_latency.png`、`chart_roundvalue_vs_baselines.png`、`chart_adaptive_stop_distribution.png`、`chart_oracle_regret.png`）。可视化不读取 trajectories，不可能反向影响评分或策略。

`roundvalue smoke|collect-analyze|visualize` 分别转发到上述三个脚本的 `main`，不改变
任何参数、强制顺序或门禁；`scripts/` 目录仍然只包含这三个用户入口。

run 命名统一为 `YYYYMMDDHHMM_<数据集名称>_<hex>`：trajectories 与 results 使用
完全相同的目录名，时间取达拉斯本地时区（America/Chicago），精确到分钟。例如
MATH-500 的 collect run 是 `202608142334_MATH500_15193d5a`（数据集名中的连字符
会被去掉，run 名只含两个下划线），smoke run 用 `smoke` 作为数据集占位名。

因此 `trajectories/` 是原始且尽量不被后续阶段修改的模型轨迹；`results/` 必须能由 trajectories 完全离线重建（重跑 step3）。`run_id` 是 step3/step4 唯一接受的实验标识，不要混用不同 run。代码任务的本地执行必须显式允许，并在去除密钥的临时子进程中进行；它不替代专用隔离执行环境。

延迟统计保留两个字段：`wall_clock_ms` 是真实的墙钟等待时间（并行的 P/A/C 请求只计一次），`api_latency_ms` 是全部 API 尝试的服务时间总和（含重试）。离线标签、报告图表与策略的时间成本使用墙钟；旧轨迹只有服务时间总和时按未知处理，不用服务时间冒充墙钟。

### 防泄漏与 Agent 可见信息

Agent 只能看到 `task_id`、`domain`、`prompt` 以及少量明确允许的公开元数据。其中 `task_id` 在送入 Agent 前会被替换为确定性的匿名哈希（原始 ID 只保留在磁盘记录中），并且会剔除 `source_task_id`、`base_input_count`、`plus_input_count` 等能暴露具体上游题号或隐藏测试规模的信息。参考答案、隐藏测试与离线 Judge 只用于离线评分和标签构建。

## Benchmark 边界

`benchmark/test/` 是仓库独立数学验收题，只用于检查 API、DAG、JSON 输出、评分与落盘是否正常；它不从论文基准抽样，也不得出现在训练、验证、测试或论文结果中。

论文主实验使用两个**各自独立**的冻结数学数据集文件：`benchmark/math/MATH-500.json`
及其验证子集 `benchmark/math/MATH-50.json`。每个数据集文件自带 `dataset_id`、`domain`、
划分与内嵌 provenance，彼此不混合。

## 正式真实数据基准（v3：按数据集独立）

每个数据集内部按统一比例 60/20/20 划分（余数计入 Test），并用固定种子确定性分配：

| 数据集 | 文件 | Train | Validation | Test | 合计 |
|---|---|---:|---:|---:|---:|
| MATH-500 | `benchmark/math/MATH-500.json` | 300 | 100 | 100 | 500 |
| MATH-50 | `benchmark/math/MATH-50.json` | 30 | 10 | 10 | 50 |

MATH-500 的 500 道题只在自己的 train/validation/test 三个互不相交分区之间流动。
MATH-50 是 MATH-500 的确定性分层子集（按学科比例、难度均匀抽样，60/20/20 划分），只用于快速验证；任务 ID、题面、参考答案与原集逐字一致，可用 `python src/build_math50.py` 重建。

数学评分采用与 MATH-500 provenance 同源的 PRM800K/MATH 归一化约定（分数简写、`\sqrt`、度数、单位、`\%` 等先统一写法）做精确比较，并附一个只支持有限数值表达式与 `\frac` 的保守数值回退，数学题默认零容差。普通百分号不当作噪声删除（`5%` 不会误判成 `5`），而是经数值回退按 `/100` 解释（`50%` 与 `0.5` 等价）。该约定外的等价改写（例如 `14/3` 与 `4.666…` 互换）不会被自动判等价，以保证与公开 MATH-500 评分口径可比。

策略默认 `λ_cost=μ_latency=0`：值函数 `G` 就是纯质量收益，阈值选择只在质量相同的情况下用更少 token 作为决胜项。质量—token 与质量—延迟的 Pareto、任务级配对置信区间和 Oracle regret 作为独立坐标报告，而不是把价格系数折叠进一个不可解释的标量。

运行正式收集：

```powershell
# 先做 smoke
roundvalue smoke

# MATH-500
roundvalue collect-analyze --benchmark benchmark/math/MATH-500.json \
  --smoke-run-id <MATH_SMOKE_RUN_ID>

# MATH-50 快速验证
roundvalue collect-analyze --benchmark benchmark/math/MATH-50.json \
  --smoke-run-id <MATH_SMOKE_RUN_ID>
```

每个数据集的 provenance 都内嵌在任务文件本身的 `provenance` 字段里，固定记录生成器
版本、来源 revision/URL 与 SHA-256、原始记录 SHA-256、划分种子与比例、以及该数据集
的全部测试 task ID；collect 时整个文件（含 provenance）都会被冻结进 run 快照。若需
从已固定来源重新构建数据，先
`python -m pip install -e ".[benchmark-build]"`，再运行
`python src/build_real_benchmarks.py` 和 `python src/verify_real_benchmarks.py`。

## 目录与产物

```text
RoundValue/
├── benchmark/{math,test}/
├── configs/{agents.json,model_config.json,topology.json}
├── .secret/model_key.json              # 仅本地存在
├── pyproject.toml                      # 依赖管理 + roundvalue 控制台命令
├── benchmark/math/MATH-500.json         # 一个数据集一个文件，provenance 内嵌
├── benchmark/math/MATH-50.json          # MATH-500 的验证子集
├── results/YYYYMMDDHHMM_<数据集>_<hex>/
├── scripts/step1_smoke.py              # scripts/ 只有这三个 Python 用户入口
├── scripts/step2_collect_analyze.py
├── scripts/step3_visualize.py
├── src/                                # 底层模块 + pipeline 编排 + benchmark 构建/验证工具
│   └── roundvalue_cli.py               # roundvalue 子命令 → 三个 step 入口的转发
├── trajectories/YYYYMMDDHHMM_<数据集>_<hex>/
├── EXPERIMENT_ARCHITECTURE.md
└── README.md
```

每个 run 冻结三份配置、基准来源与哈希、命令行、模型 profile、Git 状态、源代码快照。`trajectories/` 保存任务级原始记录（调用尝试、原始响应、checkpoint、token、延迟、重试与错误）；`results/` 保存由 step3 离线派生的评分、标签、策略、比较与汇总，可由 trajectories 重建。二者是可追溯的实验产物，默认可纳入 Git；提交前仍应确认不含密钥、个人数据或不应公开的模型输出。

正式报告应比较 Fixed-1/2/3、启发式（共识）、task-only、RoundValue、one-step Oracle 与 trajectory Oracle，并报告质量—成本 Pareto、任务级置信区间、Oracle regret、Repair/Neutral/Harm/Recovery。共识信号要求 Planner/Analyst/Critic 六条输出中至少 2/3 的 `candidate_answer` 完全一致，或 Writer 答案跨轮稳定。Oracle 仅用于诊断上界，绝不用于部署策略。
