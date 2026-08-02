# OC-RAP v48.31 CONTRACT-SLACK-RANK 变更清单

## 版本目标

v48.31 修复 v48.30 中会混淆算法判断的训练/验证/证书合同偏差，并把优化目标从“识别存在恢复机会的场景”转为“在同一 scene-time proposal 内稳定识别可安全执行的动作”。模型不接收 regime ID，也没有 Safe / Near-contact / Contact 路由。

## 修改的核心代码

| 文件 | 变更 |
|---|---|
| `src/ocrap/models/data.py` | 计算与 certificate 一致的 candidate-vs-nominal prefix deviation，并随 batch 传递。 |
| `src/ocrap/cli/train.py` | validation 应用 feasible / hard-rule / minimum-deviation 精确可执行过滤；修复 all-abstain；增加 safe top-1、valid-safe admission 与 `direct_contract_safe_rank_risk`。 |
| `src/ocrap/models/ocrap.py` | 增加全局 component support reliability；不受支持的连续风险坐标向非伤害先验收缩；保留独立 measured hard veto。 |
| `src/ocrap/models/losses.py` | component BCE 与 signed-margin regression 使用与推理一致的 support reliability。 |
| `src/ocrap/models/inference.py` | checkpoint 持久化、恢复并核验 reliability 与 slack 合同。 |
| `scripts/train_ocrap_v48_trac_sr.sh` | 转发 v48.31 的 reliability、精确 policy metric 与新 checkpoint objective 参数。 |
| `ALGORITHM_CHANGELOG.md` | 在根目录新增 v48.31 结果归因、算法修复、消融与决策规则。 |

## 新增训练与评估脚本

| 文件 | 用途 |
|---|---|
| `scripts/adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh` | 单阶段训练封装。 |
| `scripts/adapt_ocrap_v48_31_contract_slack_rank_variant.sh` | 三阶段训练：自然 factor、自然 admission、低学习率 joint refinement。 |
| `scripts/run_v48_31_contract_slack_rank_dedicated.sh` | Balanced / Precision 双卡主实验与三值返回码控制器。 |
| `scripts/calibrate_v48_31_certificate_pool.sh` | adaptation-dev 冻结规则、metric/calibration 合同核验、certificate verification-only。 |
| `scripts/run_v48_31_parallel_ablations.sh` | 四个 wave、每次最多两项并发的 8 个消融任务。 |
| `scripts/run_v48_31_dev_shadow_closed_loop.sh` | 修复后的非空、同源、配对 adaptation-dev Waymax physical shadow。 |
| `scripts/run_v48_31_stress_if_authorized.sh` | 仅在 Natural gate 自动授权后运行 held-out stress。 |

## 新增 fail-closed 工具

| 文件 | 用途 |
|---|---|
| `tools/build_v48_31_factor_support_contract.py` | 基于全局自然 population 统计五个连续物理坐标的可学习支持度；真实运行要求 NPZ 可读。 |
| `tools/check_v48_31_metric_calibration_contract.py` | 核对 checkpoint validation 与 adaptation-dev rule fitting 的精确 group / safe-opportunity 总体。 |
| `tools/check_v48_31_model_contract.py` | 核对训练 checkpoint 与推理的 component、prior、bounded admission、slack 与 reliability。 |
| `tools/check_v48_31_training_contract.py` | 核对三个训练阶段均为自然无放回，并检查可训练参数范围与 checkpoint metric。 |
| `tools/check_v48_31_stage_transfer.py` | 检查 Stage 2 不改变 factor heads，Stage 3 只改变注册的三个 calibrator。 |
| `tools/audit_v48_31_shadow_provenance.py` | 核验 shadow 目标与 WOMD validation 来源、禁止 test/stress 混入。 |
| `tools/check_v48_31_shadow_runtime_contract.py` | 核验闭环输出非空、gamma_rec、场景配对与 runtime metric 合同。 |
| `tools/check_v48_31_physical_target_support.py` | 标记无支持、常量或饱和的 Near / Contact 物理指标，避免把结构性零值解释成增益。 |
| `tools/check_v48_31_regime_targets.py` | Near / Contact 仅作为报告分层，检查各自预注册物理目标，不参与模型路由。 |

## 新增回归测试

- `tests/test_v48_31_contract_slack_rank.py`
  - exact eligibility；
  - safe all-abstain；
  - support-weighted logits/loss；
  - inference contract；
  - stage transfer；
  - metric/calibration group equality；
  - v48.31 shell 引用工具完整性。

## 随包审计与执行材料

- `OC-RAP-v48.30-results-audit-and-v48.31-CONTRACT-SLACK-RANK-plan-ZH.md`
- `OC-RAP-v48.31-run-commands-ZH.txt`
- `OC-RAP-v48.31-ALGORITHM_CHANGELOG.md`
- `OC-RAP-v48.30-main-gate-metrics-audit.csv`
- `OC-RAP-v48.30-ablation-gate-metrics-audit.csv`
- `OC-RAP-v48.30-admission-training-trajectory-audit.csv`
- `OC-RAP-dataset-reports-concise-audit.csv`
- `OC-RAP-v48.30-engineering-contract-audit.json`

## 明确保留的设计

- frozen top-3 proposal；
- observation-consistent recoverability / OC-MERO / CRISP 主线；
- raw recoverability benefit 与 safe admission 分离；
- nominal-relative 连续非退化物理余量；
- independent measured hard veto；
- bounded nominal + top-k one-action execution；
- adaptation-dev 冻结规则、certificate verification-only、Natural gate；
- 三种 regime 统一 selector，不使用 regime-conditioned policy。

## 明确替换的设计

- Stage 1 分层有放回采样 → 所有训练阶段自然无放回；
- 只训练 admission residual 且冻结错误 factor identity → 增加低学习率 joint calibrator refinement；
- raw admission 的 all-abstain → valid-safe admission 的 all-abstain；
- 训练 validation 与 certificate 不同 eligible population → 精确合同一致；
- 对无支持的 hard-rule / harm-proxy learned coordinates 赋同等权重 → 全局支持度收缩，同时保留 measured veto；
- 以 candidate AUC 作为主要进展判断 → 以 proposal 内 safe top-1、harmful mass、regret 与 certificate 泛化作为核心判断。
