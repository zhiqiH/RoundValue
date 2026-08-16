# benchmark/

存放实验用的冻结基准数据，每个数据集是一个独立 JSON 文件。

- `mmlu_pro/`：当前主实验基准（MMLU-Pro-500 全集 300/100/100，以及用于快速验证的 MMLU-Pro-50 子集 30/10/10）
- `math/`：旧数学数据集（MATH-500 / MATH-50），仅保留用于追溯与回归验证
- `test/`：仓库本地验收题，只用于 smoke，不进论文结果

每个数据集文件自带 train/validation/test 划分和内嵌 provenance。不要手工编辑；
MMLU-Pro 主集与子集分别用 `python src/build_mmlu_pro.py` 和
`python src/build_mmlu_pro_50.py` 从钉死的 MMLU-Pro revision 确定性重建，
重建后用 `python src/verify_real_benchmarks.py` 校验。
