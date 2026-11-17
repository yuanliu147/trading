# 主板极简趋势系统研究

本目录保存冻结配置下的可复现研究与负结果交付。最终裁决为 **REJECTED**；当前基线不得进入实盘观察。研究结论见 `90_研究结论.md`，失效归因和停止边界见 `92_负结果与后续边界.md`。

## 复现前提

- 在 `trading_finally/` 根目录执行下列命令，使用 Python 3 及项目已锁定的 pandas、NumPy、pyarrow、duckdb 环境。
- `config/baseline.json` 是冻结配置，不应为复现而改写。配置中的两个数据库路径只用于面板构建；基线与稳健性回测读取本地 Parquet，不连接源数据库。
- 面板构建对 DuckDB 使用 `duckdb.connect(..., read_only=True)`，对 SQLite 使用带 `?mode=ro` 的 URI。源数据库保持只读，研究输出只写入本目录的 `data/`、`results/` 和报告文件。
- 完整复现会重建同名面板和结果。需要保留旧输出时，应在运行前另行留存，不把新结果覆盖误解为旧快照仍在。

## 从面板到最终报告

按顺序执行：

```bash
# 1. 只读访问双源数据库，构建本地冻结研究面板
PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json

# 2. 运行全项目测试
PYTHONPATH=src python3 -B -m unittest discover -s tests -v

# 3. 用本地面板复现主基线
PYTHONPATH=src python3 -B scripts/run_baseline.py --config config/baseline.json

# 4. 复现稳健性、历史压力、机械裁决并生成最终研究报告
PYTHONPATH=src python3 -B scripts/run_robustness.py --config config/baseline.json
```

第 4 步生成 `90_研究结论.md`，没有另设报告脚本；该步骤耗时较长且包含完整变体矩阵与 2010—2015 单源历史压力。`92_负结果与后续边界.md` 是 REJECTED 后的人工交付，不由回测脚本改写。只做基线核对时不得用第 3 步代替或顺带重跑第 4 步。

## 大结果落盘约定

| 位置 | 内容与格式 |
| --- | --- |
| `data/` | 主面板、压力面板、公司行动、双源信号面板、指数面板等 Parquet；排除清单等 CSV；审计 JSON |
| `results/baseline/` | 账户、持仓、委托、成交、往返交易等 Parquet；年度与基准 CSV；`performance_summary.json`、`summary.md`、`run_manifest.json` |
| `results/robustness/` | 各完整账户变体结果，`variant_matrix.csv`、`annual_equity.csv`、`summary.json`、`main_freeze.json` 等 |
| `results/stress_2010_2015/` | 单源历史压力账户结果与摘要 |

大结果写入 Parquet 或 CSV，不在聊天或标准输出展开。脚本标准输出只给小型 JSON 状态；汇总指标写 JSON/Markdown，逐日、逐笔和逐变体明细留在磁盘。阅读结果时先看摘要，再按相对路径打开对应大表。

`run_manifest.json` 记录原始配置文件哈希、代码清单及哈希、输入路径、行数、代码数、数据窗口和依赖版本。相同输入复现至少核对这些字段以及 `performance_summary.json` 的期末权益、总收益、最大回撤、交易数和逐笔统计；若新增策略源码使代码清单变化，应如实记录差异，不能为匹配旧清单而删改代码或伪造哈希。

## 交付边界

本仓库没有 REJECTED 对应的实盘手册、盘后日报、持仓台账或前瞻订单协议。复现仅用于验证研究记录，不产生交易候选，也不改变既有否决裁决。
