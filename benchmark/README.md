# benchmark/

存放实验用的冻结基准数据，每个数据集是一个独立 JSON 文件。

- `math/`：数学数据集（MATH-500 全集，以及用于快速验证的 MATH-50 子集）
- `code/`：代码数据集（HumanEval+、MBPP+）
- `test/`：仓库本地验收题，只用于 smoke，不进论文结果

每个数据集文件自带 train/validation/test 划分和内嵌 provenance。不要手工编辑；需要重建时运行 `python src/build_real_benchmarks.py`，MATH-50 子集用 `python src/build_math50.py` 从 MATH-500 确定性重建。
