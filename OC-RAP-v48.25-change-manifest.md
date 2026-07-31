# OC-RAP v48.25 INTEGRITY-BRIDGE 变更清单

## 目标

v48.25 首先恢复实验完整性：修复 v48.24 的 RC 语义、模型配置丢失、validation index 错配与 all-abstain checkpoint 选择；随后用新的 dev-frozen / full-certificate verification 协议重新评价 Near/Contact safe admission。

## 修改文件

### `src/ocrap/cli/train.py`

- 将以下配置真实传入 `OCRAPModel`：
  - `direct_recovery_evidence_frontier`
  - `direct_recovery_evidence_component_prior_logit`
  - `direct_recovery_evidence_admission_bounded`
- 支持独立 `training.validation_group_index_path`。
- 缺少 dev index 时关闭错误的 validation exact stratification。
- 新增 `direct_integrity_selection_risk`，惩罚 Near/Contact hard recall shortfall 与全 abstain。
- 删除误导性的 path-name `safe_positive_fraction` 解释。

### `src/ocrap/models/ocrap.py`

- 新增 `direct_recovery_evidence_admission_bounded`。
- 支持 bounded `tanh` residual 与 unbounded zero-initialized linear residual 两种模式。
- 保持 residual=0 时的 transferred-prior identity。

### `scripts/train_ocrap_v48_trac_sr.sh`

- 传入 `VAL_GROUP_INDEX`。
- 传入 admission bounded 配置。
- 传入 integrity checkpoint metric 的阈值和权重。

### `tools/calibrate_policy_risk_v48.py`

- 新增 `--development-fit-only`。
- 新增 `--verification-only` 与 `--frozen-rule-json`。
- 写入 frozen rule SHA256 与数据 provenance。
- 支持 adaptation-dev threshold fitting 和 full certificate verification。
- 结构支持不足返回 3（Natural-gate reject），不再返回 4。
- 只有空/无效 certificate population 返回 4。
- 新增 `certificate_data_valid`、`certificate_support_feasible`、`gate_evaluated` 与 `rejection_kind`。

### `scripts/run_ocrap_v48_trac_sr.sh`

- 正式 deployment 仍严格要求有效 certificate。
- `DEV_SHADOW_DIAGNOSTIC=1` 可在 certificate 拒绝后读取 adaptation-dev frozen rule，仅用于 dev shadow。

### 新增主流程

- `scripts/adapt_ocrap_v48_25_integrity_variant.sh`
- `scripts/calibrate_v48_25_certificate_pool.sh`
- `scripts/run_v48_25_integrity_dedicated.sh`
- `scripts/run_v48_25_dev_shadow_closed_loop.sh`
- `scripts/run_v48_25_stress_if_authorized.sh`

### 新增双 A30 消融

- `scripts/run_v48_25_parallel_ablations.sh`
- 四个 wave，每个 wave Balanced/GPU0 与 Precision/GPU1 并发。
- 每张 A30 同时只运行一个任务，每张卡四个任务。

### 新增测试与目标检查

- `tests/test_v48_25_integrity_bridge.py`
- `tools/check_v48_25_regime_targets.py`

测试覆盖：

- CLI 构造参数不再丢失；
- unbounded residual 可越过 bounded ceiling；
- all-abstain checkpoint 被 integrity metric 惩罚；
- RC=20/30、dev-frozen certificate 与 shadow 合同；
- 双 A30 调度与 dev index 合同。

### 文档与审计

- `ALGORITHM_CHANGELOG.md`：根目录追加 v48.25 条目。
- `OC-RAP-v48.24-results-audit-and-v48.25-INTEGRITY-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.24-results-audit-summary.json`
- `OC-RAP-v48.24-metrics-compare-v48.23.csv`
- `OC-RAP-v48.25-run-commands-ZH.txt`

## 未修改的合同

- 不向模型暴露 Near/Contact regime ID。
- 不读取 held-out test/stress，除非 RC=0 自动授权。
- 不降低 verify min-selected、precision LCB 或 harm UCB 要求。
- 不在 v48.25 主轮事后放宽 component-veto 标签。
- 不声称真实 WOMD/Waymax gate 或闭环结果。
