# OC-RAP v48.33 全量闭环与critical可视化修改清单

## 修改文件

- `src/ocrap/simulation/closed_loop_runner.py`
  - 保存逐步物理指标与ego轨迹；
  - 支持target-key精确复跑。
- `src/ocrap/config/defaults.py`
  - 新增 `closed_loop.save_traces`；
  - 新增 `closed_loop.target_keys_file`。
- `scripts/run_ocrap_v48_trac_sr.sh`
  - near/contact独立rollout/step/target设置；
  - 支持trace保存与critical target文件。
- `tools/compare_paired_closed_loop.py`
  - 增加overlap/offroad/acceleration/deceleration/jerk/yaw-rate/route progression。

## 新增文件

- `scripts/run_v48_33_ungated_full_closed_loop.sh`
- `tools/count_closed_loop_targets.py`
- `tools/select_critical_closed_loop_scenes.py`
- `tools/visualize_closed_loop_critical.py`
- `tools/summarize_ungated_closed_loop.py`
- `tests/test_critical_closed_loop_visualization.py`
- `docs/dataset_report_summary_v48_33.json`
- `docs/dataset_report_summary_v48_33.csv`
- `docs/OC-RAP-v48.33-paper-code-data-results-review-ZH.md`
- `OC-RAP-v48.33-ungated-full-closed-loop-ZH.txt`

## 验证状态

- Python compile：通过；
- `bash -n`：通过；
- `PYTHONPATH=src pytest -q tests/test_critical_closed_loop_visualization.py tests/simulation/test_closed_loop_timing_aggregation.py`：5 passed；
- 现有precision dev-shadow near/contact结果：已生成critical JSON/CSV和HTML/PNG示例。

## 兼容性

- 原有 `save_trace_npz` 仍可使用；
- 默认不保存trace，不影响旧命令；
- 默认不读取target-key文件；
- 正式gate流程未被修改；ungated test由独立脚本显式触发。
