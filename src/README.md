# src/

底层实现与共享编排，采用扁平模块。

- 运行时模块：`config_loader`、`provider`、`debate_runner`、`scorer`、`policy`、`pipeline` 等
- 构建/验证工具：`build_real_benchmarks.py`、`verify_real_benchmarks.py`
- `roundvalue_cli.py`：`roundvalue` 控制台命令 → 四个 step 入口的转发

普通实验流程不直接运行这里的文件（构建/验证工具除外）。
