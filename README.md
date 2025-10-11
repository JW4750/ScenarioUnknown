# HighD Scenario Mining Toolkit

本项目提供一个可在 Windows + Miniforge 环境下运行的 HighD 高速公路场景识别与覆盖度评估工具。核心能力包括：

* 逐帧生成语义标签（车辆类型、纵向/横向运动状态、周围车辆关系等）；
* 识别官方定义的 10 类高速驾驶场景并统计出现频次；
* 检测未知危险场景并计算其发生频率（公里/次）；
* 基于标签组合的覆盖度指标计算；
* 导出所有中间与统计结果为 CSV，同时生成可浏览的 HTML 报告。

## 目录结构

```
ScenarioUnknown/
├── highd_scenario_mining/        # Python 包（核心逻辑）
│   ├── __init__.py
│   ├── cli.py                    # 命令行入口
│   └── pipeline.py               # 数据处理流水线
├── scripts/
│   └── highd_scenario_mining.py  # 兼容性的可执行脚本
├── tests/
│   └── test_pipeline.py          # 合成数据的功能测试
├── README.md
├── pyproject.toml
├── requirements.txt
└── highd_scenario_mining (1).py  # 与历史脚本同名的包装器
```

## 安装

建议在 Miniforge/conda 环境中安装依赖：

```bash
conda create -n highd python=3.10 -y
conda activate highd
pip install -r requirements.txt
```

也可以使用 `pip install -e .` 在开发模式下安装该包并注册命令行入口。

## 使用方法

1. 将 HighD 数据集的三个 CSV 文件放入同一目录中，文件名需符合官方格式：
   * `<prefix>_tracks.csv`
   * `<prefix>_tracksMeta.csv`
   * `<prefix>_recordingMeta.csv`

2. 运行命令行工具：

```bash
python -m highd_scenario_mining.cli --prefix 18 --data-dir D:/highd/ --output-dir D:/highd/results/
```

或使用兼容脚本：

```bash
python scripts/highd_scenario_mining.py --prefix 18 --data-dir D:/highd/
```

所有输出（CSV + HTML 报告）会保存在 `--output-dir` 目录内，默认与数据目录相同。

### 配置覆盖

通过 `--config-json` 传入一个 JSON 文件即可覆盖默认阈值，例如：

```json
{
  "accel_threshold": 0.4,
  "ttc_crit": 1.2
}
```

## 输出文件说明

* `highd_tags.csv`：逐帧标签结果。
* `highd_scenarios.csv`：识别出的场景片段及关联车辆。
* `highd_scenario_counts.csv`：各类场景的出现次数。
* `highd_unknown_danger_events.csv`：未知危险事件片段。
* `highd_unknown_danger_stats.csv`：未知事件统计（数量、公里/次）。
* `highd_coverage_metrics.csv`：覆盖度指标列表。
* `highd_report.html`：含图表的可视化报告。

## 开发与测试

项目包含一个基于合成数据的 pytest 测试用例，可验证流水线是否能够正常运行并生成所有输出文件：

```bash
pytest
```

## 许可证

本仓库内源代码以 MIT 许可证发布，可自由复制与修改。详情见 [`LICENSE`](LICENSE)。
