# OC-RAP v48.28 PROVENANCE-MARGIN-BRIDGE 变更清单

## 核心数据与 closed-loop provenance

- `src/ocrap/data/waymax_loader.py`
  - 增加官方 WOMD `scenario/id` custom loader；
  - 保留官方 ID，同时生成 legacy state-hash 迁移键；
  - 记录 WOMD source role、source pattern 和 Waymax object 配置。
- `src/ocrap/data/build/history.py`
  - 将 source scenario index、official/legacy ID 和 source provenance 写入 history metadata。
- `src/ocrap/data/build/builder.py`
  - 将 provenance 写入 manifest row 和 manifest field list。
- `src/ocrap/data/schema.py`
  - 将 provenance 字段持久化到 NPZ。
- `src/ocrap/simulation/closed_loop_runner.py`
  - official ID/legacy alias/source-index 三层匹配；
  - split role 不一致时 fail-closed；
  - source-index fallback 只允许相同 source role；
  - 改进 target mismatch 错误说明和输出 provenance。
- `scripts/run_ocrap_v48_trac_sr.sh`
  - dev-shadow 默认改为 standard validation；
  - 启用 official scenario ID；
  - 保持完整扫描和 target fail-fast。

## 模型与训练

- `src/ocrap/models/ocrap.py`
  - component-harm 默认 residual scale 从 2.0 改为 6.0；
  - 保留 prior=-2，表示范围约为 `[-8,4]`。
- `src/ocrap/models/inference.py`
  - checkpoint/inference 默认 component scale 与训练一致。
- `src/ocrap/cli/train.py`
  - 新增 `direct_factor_supervised_risk`；
  - factor checkpoint 指标显式包含实际 supervised factor loss。
- `src/ocrap/config/defaults.py`
  - 新增 official ID 与 legacy migration 默认配置；
  - component scale 默认值更新。

## v48.28 主实验与诊断脚本

- `scripts/adapt_ocrap_v48_28_provenance_margin_single_stage.sh`
- `scripts/adapt_ocrap_v48_28_provenance_margin_variant.sh`
  - 两阶段 factor→admission；
  - 两阶段都禁止 epoch-0 checkpoint；
  - 主模型使用 regression-only safe utility。
- `scripts/calibrate_v48_28_certificate_pool.sh`
  - v48.28 dev-frozen/full-certificate 协议入口。
- `scripts/run_v48_28_provenance_margin_dedicated.sh`
  - 主实验 controller；
  - model contract、factor transfer、gate failure decomposition。
- `scripts/run_v48_28_dev_shadow_closed_loop.sh`
  - shadow provenance 预审计；
  - standard validation；
  - official/legacy migration。
- `scripts/repair_v48_27_dev_shadow_with_v48_28.sh`
  - 无需重训地修复 v48.27 shadow。
- `scripts/run_v48_28_stress_if_authorized.sh`
  - 仅 RC=0 自动授权后读取 held-out stress。

## 八任务并发消融

- `scripts/run_v48_28_parallel_ablations.sh`
  - 八个任务一次启动；
  - 每张 A30 四个任务；
  - 每任务 `NUM_WORKERS=1`；
  - host BLAS/OpenMP 每任务 1 线程；
  - 四组分别验证 factor count、harm range、regression-only 和 listwise/frontier。

## 新增审计工具

- `tools/audit_v48_28_shadow_provenance.py`
- `tools/check_v48_28_factor_transfer.py`
- `tools/check_v48_28_model_contract.py`
- `tools/check_v48_28_regime_targets.py`
- `tools/summarize_v48_28_gate_failure.py`

## 测试与文档

- `tests/test_v48_28_provenance_margin_bridge.py`
  - official ID；
  - split role；
  - legacy migration；
  - harm range；
  - factor checkpoint；
  - model contract；
  - shadow fail-closed；
  - 8-task concurrency。
- `ALGORITHM_CHANGELOG.md`
  - 新增 v48.28 完整算法和协议记录。
- `README.md`
  - 增加当前开发版本入口。
- `OC-RAP-v48.27-results-audit-and-v48.28-PROVENANCE-MARGIN-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.28-run-commands-ZH.txt`
