# RoundValue 实验架构（冻结版）

## 1. 边界与第一性原理

本仓库是论文实验的可执行部分，不保存论文正文、投稿图表或文献。研究对象是：在**固定通信拓扑**下，已完成一轮后是否值得继续一轮完整 Debate。配置、基准、轨迹和结果均为 JSON；没有伪造 Provider、假结果或虚拟运行逻辑。

一次运行只允许在完整 Writer checkpoint 后输出 `STOP` 或 `CONTINUE`，最多三轮。Single Agent 是独立基线，不被伪装为第 0 轮。

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

## 4. 实现边界

```text
benchmark/math/                   MATH 论文主基准的 JSON 来源登记与任务文件
benchmark/code/                   EvalPlus HumanEval+/MBPP+ 论文主基准的 JSON 来源登记与任务文件
benchmark/formal_experiment_v1.json  冻结的 842 题正式真实数据 manifest
benchmark/formal_experiment_v1.provenance.json  固定来源 revision、哈希和 MATH-100 抽样记录
benchmark/test/                   独立仓库验收题；不来自主基准且不用于论文结果
configs/                          agents.json, model_config.json, topology.json
roundvalue                         唯一用户命令（映射至 scripts/test_benchmark.py）
src/                              扁平模块：Provider、DAG、评分、标签、策略、存储、报告
trajectories/<run_id>/            任务级完整调用与 checkpoint 记录
results/<dated_run_id>/           聚合指标、置信区间、策略报告与 manifest
.secret/model_key.json            本地密钥，永不提交
```

三份配置同时校验。`agents.json` 固定角色提示词和字段；`topology.json` 是拓扑注册表，当前选择并冻结上述 Debate DAG；`model_config.json` 冻结模型与运行参数。新增拓扑时在 `topologies` 下新增 ID，并同时实现对应 runner 与校验器；不会静默改写现有 Debate。默认 DeepSeek profile 使用 `deepseek-v4-flash`、`temperature: 0.0`，适配器显式发送 `thinking: {"type":"disabled"}`。Provider 是通用接口：新增厂商只增加适配器和 JSON profile，不改变 Debate、评分或策略。

正式基准的 Train/Validation/Test 划分是：140 道 MATH 开发题 / 60 道 MATH 开发题 / 100 道 MATH-500 派生的 MATH-100、164 道 HumanEval+、378 道 MBPP+。开发题在生成时排除所有 MATH-500 题目；全部 542 道代码题都只属于 Test，因此其结果是跨领域策略评估，不是 EvalPlus 训练集结果。`formal_experiment_v1.provenance.json` 冻结每个来源的 revision、上游校验信息、原始记录 hash、抽样种子和测试 ID。

代码任务使用 `evalplus_differential_v1`：在去除密钥的本地子进程中，以官方发布的 base/plus 输入、canonical oracle 与容差进行差分评分，并且不向 Agent 暴露测试输入或 oracle。该适配器不是官方 EvalPlus runner、容器或 leaderboard；论文中必须将其结果标记为 RoundValue differential-adapter 结果。本地代码执行仍须显式授权，且不构成操作系统级沙箱。

## 5. 实验链路

1. `smoke`：独立验收题、真实 API、每题完整一轮；验证密钥、配置、DAG、Writer JSON、评分、预期分数与落盘。代码验收须显式授权本地执行。
2. `collect`：按原始 `task_id` 冻结 Train/Validation/Test，收集每题至多三轮完整轨迹。正式命令为 `roundvalue --mode collect --benchmark benchmark/formal_experiment_v1.json --allow-local-code-evaluation`。
3. `fit`：仅 Train 拟合质量增益；仅 Validation 选择阈值和资源偏好。
4. `evaluate`：只在冻结 Test 轨迹重放基线、RoundValue 与 Oracle，不发起 API 请求。
5. `reproduce`：仅由保存的 JSON 重建派生标签、指标和报告。

每个 run 保存配置快照和哈希、基准来源与哈希、Git 状态、源代码快照、命令行、模型响应名、API 尝试、checkpoint、评分、特征、标签和聚合结果。服务端即便在温度为零时仍可能变化，因此以任务级轨迹和聚类 bootstrap 报告不确定性，不宣称逐 token 完全确定。

## 6. 论文结论的最低证据

比较 Single Agent、Fixed-1/2/3、启发式、task-only、RoundValue、one-step Oracle 和 trajectory Oracle。报告质量—成本 Pareto、任务级置信区间、Oracle regret 与 Repair/Neutral/Harm/Recovery。Oracle 只测量可达上界与后悔值，绝不可部署。
