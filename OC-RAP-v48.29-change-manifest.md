# OC-RAP v48.29 VETO-RANK-PHYSICS-BRIDGE 变更清单

## 核心模型与训练

- `src/ocrap/models/ocrap.py`
  - 新增 `direct_recovery_evidence_admission_prior_mode`；
  - 支持 `risk_centered` 与 `benefit_only`；
  - v48.29 主模型使用 benefit-only prior，风险保留为独立 veto。
- `src/ocrap/models/losses.py`
  - 新增 deployment-score 对齐的 safe hardest-negative group loss；
  - safe-positive group 中 teacher-best safe action 对比 nominal 与 hardest non-safe；
  - 无 safe opportunity group 中所有 recovery 均压到 nominal 以下。
- `src/ocrap/cli/train.py`
  - 新配置传递、checkpoint 持久化与 loss 参数接线。
- `src/ocrap/models/inference.py`
  - admission prior mode 的 checkpoint/inference 合同恢复与 fail-closed 检查。
- `configs/default.yaml`
  - 增加 admission prior mode 和 safe hard-negative 默认配置。
- `scripts/train_ocrap_v48_trac_sr.sh`
  - 将新增模型/loss 配置传入统一训练入口。

## Regime 与闭环工程修复

- `src/ocrap/utils/regimes.py`
  - 新增 provenance-aware canonical regime/alias 解析。
- `src/ocrap/evaluation/baselines.py`
  - selector 的所有 per-bucket/per-regime override 使用统一 alias。
- `src/ocrap/simulation/closed_loop_runner.py`
  - calibrated gamma、Contact 判定和物理指标使用统一 canonical regime；
  - scene/aggregate 输出 runtime contract 元数据；
  - Near 不再因名称包含 `contact` 被误判为 Contact。
- `tools/check_v48_29_shadow_runtime_contract.py`
  - fail-closed 检查正 gamma、canonical regime、Contact anchor/post-contact semantics；
  - 非有限值安全写为 JSON null。
- `tools/audit_v48_29_shadow_provenance.py`
  - v48.29 shadow provenance 审计。
- `scripts/run_v48_29_dev_shadow_closed_loop.sh`
  - 默认 fast physical mode；
  - Balanced/Precision 双卡并发；
  - 完成后强制 runtime-contract 检查。
- `scripts/repair_v48_28_dev_shadow_with_v48_29.sh`
  - 无需重训修复 v48.28 shadow。

## v48.29 主实验与消融

- `scripts/adapt_ocrap_v48_29_veto_rank_single_stage.sh`
- `scripts/adapt_ocrap_v48_29_veto_rank_variant.sh`
  - factor→admission 两阶段训练；
  - five-factor wide-range；
  - benefit-only bounded admission；
  - safe regression + categorical + hardest-negative。
- `scripts/calibrate_v48_29_certificate_pool.sh`
  - adaptation-dev 冻结 rule，完整 certificate 只验证。
- `scripts/run_v48_29_veto_rank_physics_dedicated.sh`
  - 主实验控制器、RC=0/20/30 分离、model contract 和 gate decomposition。
- `scripts/run_v48_29_parallel_ablations.sh`
  - 8 个消融同时启动，GPU0 四个 Balanced、GPU1 四个 Precision。
- `scripts/run_v48_29_stress_if_authorized.sh`
  - 仅在 `NEXT_COMMANDS.txt` 授权后读取 held-out stress。
- `tools/check_v48_29_model_contract.py`
- `tools/check_v48_29_factor_transfer.py`
- `tools/summarize_v48_29_gate_failure.py`
- `tools/check_v48_29_regime_targets.py`

## 测试与文档

- `tests/test_v48_29_veto_rank_physics_bridge.py`
  - 8 个新测试，覆盖 alias/gamma、Contact semantics、benefit-only prior、hard-negative objective、fast shadow、八任务并发和模型合同。
- `ALGORITHM_CHANGELOG.md`
  - 追加 v48.29 设计、归因、非声明与停止规则。
- `OC-RAP-v48.28-results-audit-and-v48.29-VETO-RANK-PHYSICS-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.29-run-commands-ZH.txt`
- 结构化审计 JSON、gate/ablation/shadow timing CSV。

## 本地验证

- `python -m compileall -q src tools tests`：通过；
- 全部 Shell `bash -n`：通过；
- `pytest`：259 passed，5 warnings；
- 当前环境无真实 WOMD/Waymax 和两张 A30，未声明 gate 或闭环性能已通过。
