# OC-RAP v48.32.1 RC30-INTEGRITY-HOTFIX 变更清单

## 发布性质

这是 v48.32 的工程热修版本。算法结构、数据、proposal、损失权重、训练轮数、gate 阈值和统一三-regime语义均未修改。

## 确定性崩溃修复

- `src/ocrap/models/losses.py`
  - 候选级 gap 张量改名为 `teacher_gap_vector`。
  - adaptive hardest-negative 局部标量改名为 `adaptive_teacher_gap`。
  - 增加 strict shape contract 与 exactly-one-nominal group contract。
- `src/ocrap/cli/train.py`
  - 接入 `training.direct_value_strict_shape_contract`。
- `scripts/train_ocrap_v48_trac_sr.sh`
  - 将 strict shape 配置传入训练 CLI。

## Fail-fast 与可复现性

- 新增 `tools/check_v48_32_1_multigroup_loss_contract.py`：两组proposal、factorized harm、adaptive margin、forward/backward 与变量遮蔽预检。
- `src/ocrap/algorithms/lcv.py`：deterministic CUDA 模式使用矩阵式 exclusive prefix，绕开 CUDA cumsum 非确定性路径。
- 新的 v48.32.1 训练与主控制脚本统一设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 精确 Stage-1 缓存

- 新增 `tools/materialize_v48_32_1_factor_cache.py`。
- 校验源checkpoint SHA与两个完成元数据。
- 原子复制后再次校验目标checkpoint SHA。
- 重写目标完成元数据中的checkpoint路径。
- 主控制器支持 Balanced/Precision 独立缓存目录。

## 状态与失败审计

- 新增 `tools/extract_v48_32_1_failure_signature.py`。
- 新增 `tools/check_v48_32_1_metric_calibration_contract.py`，保证发布脚本依赖闭包完整，并保持 v48.32 的 validation/calibration population identity 合同。
- variant 失败时写入 `VARIANT_STAGE_FAILED.json`。
- controller 写入 `FAILURE_SIGNATURE_balanced.json` / `precision.json`。
- `certificate_executed`、`gate_evaluated`、`certificate_exit_code`、`pipeline_exit_code` 分离。
- certificate artifact/protocol failure 不再错误标记为已评估 Natural gate。

## 新脚本

- `scripts/adapt_ocrap_v48_32_1_identity_utility_single_stage.sh`
- `scripts/adapt_ocrap_v48_32_1_identity_utility_variant.sh`
- `scripts/run_v48_32_1_rc30_integrity_hotfix_dedicated.sh`
- `scripts/calibrate_v48_32_1_certificate_pool.sh`
- `scripts/run_v48_32_1_stress_if_authorized.sh`

## 发布依赖审计

- 新增 `OC-RAP-v48.32.1-script-dependency-audit.json`；发布时扫描所有新增控制脚本引用的 `scripts/` 与 `tools/` 文件并 fail-closed。

## 测试

- 新增 `tests/test_v48_32_1_rc30_integrity_hotfix.py`。
- 更新 v48.32 adaptive-margin 测试，明确禁止重新使用危险的外层变量名。
