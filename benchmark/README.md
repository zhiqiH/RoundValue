# benchmark/

存放实验用的冻结基准数据，每个数据集是一个独立 JSON 文件。

- `mmlu_pro/`：当前主实验基准（MMLU-Pro-500 全集 300/100/100，以及用于快速验证的 MMLU-Pro-50 子集 30/10/10）
- `harp/`：HARP 竞赛数学多选题基准（HARP-500 全集 300/100/100，HARP-50 子集 30/10/10）
- `logiqa2/`：LogiQA 2.0 英文 MRC 逻辑推理基准（LogiQA-500 全集 300/100/100，LogiQA-50 子集 30/10/10）
- `math/`：旧数学数据集（MATH-500 / MATH-50），仅保留用于追溯与回归验证
- `test/`：仓库本地验收题，只用于 smoke，不进论文结果

每个数据集文件自带 train/validation/test 划分和内嵌 provenance。不要手工编辑；
各基准主集与子集用对应的 builder 从钉死的官方 revision 确定性重建（HARP：
`src/build_harp.py` / `src/build_harp_50.py`；LogiQA：
`src/build_logiqa.py` / `src/build_logiqa_50.py`），重建后用
`python src/verify_real_benchmarks.py` 校验。HARP 与 LogiQA 2.0 的上游来源
与许可证说明见各自目录下的 README。
