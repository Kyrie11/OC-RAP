# OC-RAP v48.32 IDENTITY-UTILITY-BRIDGE 变更清单

## 1. 目标

v48.32 同时处理两类问题：

1. v48.31 主流程因合法 epoch-0 fallback 被错误判为 stage-transfer corruption，导致假 RC=30 与 `NEXT_COMMANDS.txt` 缺失；
2. v48.31 的部署 safe-utility loss 无法更新 benefit/component identity heads，造成候选级 AUC 与 proposal 内安全 top-1 脱节。

算法仍是一个统一、无 regime ID 的 candidate-vs-nominal selector。

## 2. 核心模型与训练代码

| 文件 | 修改 |
|---|---|
| `src/ocrap/models/ocrap.py` | 新增 `direct_recovery_evidence_admission_prior_detach`；Stage-2 可将 deployment safe-utility 梯度耦合到 benefit 与 supported component calibrators；推理公式不变 |
| `src/ocrap/models/losses.py` | 新增连续 teacher-gap hardest-negative margin；no-safe group 使用连续 no-op depth |
| `src/ocrap/cli/train.py` | 传递、保存新模型与 loss 配置 |
| `src/ocrap/models/inference.py` | 恢复并审计 checkpoint 中的新合同字段 |
| `scripts/train_ocrap_v48_trac_sr.sh` | 暴露 prior detach 和 adaptive margin 环境变量 |
| `scripts/adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh` | 为旧入口补充新 loss 参数的默认透传，保持向后兼容 |

## 3. v48.32 训练与控制器

| 文件 | 修改 |
|---|---|
| `scripts/adapt_ocrap_v48_32_identity_utility_single_stage.sh` | 写入完整 Stage 架构合同，显式传递 gradient-detach 与 adaptive-margin 参数 |
| `scripts/adapt_ocrap_v48_32_identity_utility_variant.sh` | Stage-1 factor、Stage-2 joint identity、Stage-3 admission-only；no-final 完整元数据复制；exact factor cache |
| `scripts/run_v48_32_identity_utility_bridge_dedicated.sh` | 清理 stale state；分离 pipeline/certificate RC；显式 NEXT 状态；train/dev index 合同复用 |
| `scripts/calibrate_v48_32_certificate_pool.sh` | RC=0/20/30 分别生成 generated/blocked 状态；校验 certificate artifact 与 metric contract |
| `scripts/run_v48_32_identity_utility_bridge_ablations.sh` | 新 A/B/C/D；所有失败结构化；A factor 被 B/C/D 复用；可进一步复用主实验 factor |
| `scripts/run_v48_32_dev_shadow_closed_loop.sh` | Waymax 前 fail-closed preflight；无效 pipeline 不启动仿真 |
| `scripts/run_v48_32_stress_if_authorized.sh` | 只有 RC=0、gate_passed 与 NEXT generated 同时满足时授权 held-out stress |

## 4. 新增 fail-closed 工具

| 文件 | 用途 |
|---|---|
| `tools/check_v48_32_stage_transfer.py` | 检查 factor→identity→final 参数变化；Stage-2/3 合法 no-op 均接受 |
| `tools/check_v48_32_training_contract.py` | 审计三阶段自然采样、exact eligibility、gradient coupling、adaptive margin、cache 与 support 合同 |
| `tools/check_v48_32_model_contract.py` | 审计训练/推理模型构造一致性 |
| `tools/check_v48_32_metric_calibration_contract.py` | 审计 checkpoint exact population 与 adaptation-dev calibration population 一致性 |
| `tools/manage_v48_32_factor_cache.py` | 通过 source/index/support/variant/hyperparameter SHA 合同创建或验证 Stage-1 缓存 |
| `tools/build_v48_32_factor_support_contract.py` | 生成 v48.32 标识的全局连续物理坐标支持合同 |
| `tools/audit_v48_32_shadow_provenance.py` | 审计 shadow 场景与目标来源 |
| `tools/check_v48_32_shadow_runtime_contract.py` | 审计 Waymax runtime 与 policy 配置 |
| `tools/check_v48_32_physical_target_support.py` | 审计物理指标目标的非空支持 |
| `tools/check_v48_32_regime_targets.py` | 审计 Near/Contact 报告层目标，不向模型暴露 regime |

## 5. 回归测试

| 文件 | 覆盖范围 |
|---|---|
| `tests/test_v48_32_identity_utility_bridge.py` | coupled/detached 实际梯度；adaptive margin；Stage-2/3 no-op；冻结参数保护；cache relocation/mismatch；NEXT 状态；shadow preflight；消融缓存 |
| `tests/test_v48_29_veto_rank_physics_bridge.py` | 更新 benefit-only prior 断言以兼容可配置 gradient detach |
| `tests/test_v48_30_slack_rank_bridge.py` | 更新 safety-slack prior 断言以兼容可配置 gradient detach |

## 6. 分析、执行与审计文件

- `OC-RAP-v48.31-results-audit-and-v48.32-IDENTITY-UTILITY-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.32-run-commands-ZH.txt`
- `OC-RAP-v48.32-ALGORITHM_CHANGELOG.md`
- `OC-RAP-v48.31-RC-and-engineering-audit.json`
- `OC-RAP-v48.31-ablation-execution-audit.csv`
- `OC-RAP-v48.31-valid-certificate-metrics.csv`
- `OC-RAP-v48.31-training-time-audit.csv`
- `OC-RAP-v48.31-speed-audit.json`
- `OC-RAP-v48.32-script-dependency-audit.json`

## 7. 速度变化

- v48.31 八个消融重复训练 8 次 factor stage，累计约 3.14 小时；
- v48.32 独立消融最多训练 2 次，估计节省约 2.38 小时；
- 按推荐顺序复用主实验 factor 后，消融额外 factor training 为 0，估计节省约 3.14 小时；
- certificate population 与 Waymax rollout 不缩减。

## 8. 本地验证

- pytest：285 passed，5 warnings；
- compileall：PASS；
- 全部 shell `bash -n`：PASS；
- v48.32 shell-to-tool dependencies：PASS；
- patch clean-apply：PASS；最终 ZIP integrity：PASS；SHA256 清单已生成。
