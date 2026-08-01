# OC-RAP v48.27 FACTOR-PHYSICS-BRIDGE 变更清单

基线：v48.26 EXECUTION-PHYSICS-BRIDGE  
目标：v48.27 FACTOR-PHYSICS-BRIDGE

## 核心模型与训练

| 文件 | 主要修改 |
|---|---|
| `src/ocrap/models/ocrap.py` | component harm head 从固定 3 维改为可持久化的 3–5 维；v48.27 使用 DRS、deployability、gap、hard-rule、harm-proxy 五因子。 |
| `src/ocrap/models/losses.py` | 五因子动态监督；safe-utility listwise/frontier 改用真实部署分数；取消隐藏的 `:3` 截断。 |
| `src/ocrap/models/inference.py` | 推理构造、配置回填和 checkpoint 合同持久化 component count。 |
| `src/ocrap/cli/train.py` | 训练构造与 checkpoint 持久化 component count；新增 factor-only checkpoint metric；sampler 日志区分 legacy sample 与 safe-positive group。 |
| `scripts/train_ocrap_v48_trac_sr.sh` | 将 `EVIDENCE_COMPONENT_COUNT` 传入模型配置。 |
| `scripts/adapt_ocrap_v48_27_factor_physics_single_stage.sh` | v48.27 单阶段统一训练入口，明确 raw-benefit、五因子、admission 与 gate positive 合同。 |
| `scripts/adapt_ocrap_v48_27_factor_physics_variant.sh` | 新增两阶段训练：Stage 1 只训练 benefit+5 harm factors；Stage 2 冻结 factors，只训练 bounded admission。Stage 1 关闭 setwise/selective admission 梯度。 |

## Certificate 与 gate

| 文件 | 主要修改 |
|---|---|
| `tools/calibrate_policy_risk_v48.py` | 新增 `--gate-positive-mode`；support/precision/recall 使用 safe-benefit；区分 development-rule、certificate-verification、structural-support rejection。 |
| `scripts/calibrate_v48_27_certificate_pool.sh` | adaptation-dev 冻结规则 + 完整 certificate verification；raw-benefit 模型语义与 safe-benefit gate ground truth 分离。 |
| `tools/check_v48_27_model_contract.py` | fail-closed 检查 frontier、bounded admission、semantic prior 和五因子配置。 |
| `scripts/run_v48_27_factor_physics_dedicated.sh` | 双 A30 主实验、两阶段训练、model-contract、RC 0/20/30 与 v48.27 工件。 |

## dev-shadow 与物理指标

| 文件 | 主要修改 |
|---|---|
| `src/ocrap/simulation/closed_loop_runner.py` | canonical WOMD scene ID；优先 original scenario ID；空结果 `metrics_valid=false`；关键空指标为 null；完整扫描后 0 target match 时 fail-fast。 |
| `scripts/run_ocrap_v48_trac_sr.sh` | `DEV_SHADOW_WOMD_SOURCE`、完整 raw scan、`require_bucket_targets=true`、结果有效性检查。 |
| `scripts/run_v48_27_dev_shadow_closed_loop.sh` | Balanced/Precision 双卡运行；Near/Contact 顺序执行；要求非空 paired 输出。 |
| `scripts/repair_v48_26_dev_shadow_with_v48_27.sh` | 不重训，使用新 runner 修复并重跑旧 v48.26 development shadow。 |
| `tools/check_v48_27_regime_targets.py` | 检查 Near clearance/TTC/exposure 与 Contact re-contact/overlap/free-space/escape/stable-stop 目标。 |
| `scripts/run_v48_27_stress_if_authorized.sh` | 只有自动生成授权文件时允许 held-out stress。 |

## 消融与测试

| 文件 | 主要修改 |
|---|---|
| `scripts/run_v48_27_parallel_ablations.sh` | A 三因子联合、B 五因子联合、C 五因子两阶段 regression、D 完整模型；每个 wave Balanced GPU0、Precision GPU1。 |
| `tests/test_v48_27_factor_physics_bridge.py` | 五因子 shape/prior、训练/推理持久化、部署分数尺度、raw/safe 语义分离、scene ID、空指标、shadow fail-fast、两阶段和双卡消融测试。 |

## 文档与审计

- `ALGORITHM_CHANGELOG.md` 与 `OC-RAP-v48.27-ALGORITHM_CHANGELOG.md`
- `OC-RAP-v48.26-results-audit-and-v48.27-FACTOR-PHYSICS-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.26-results-audit-summary.json`
- `OC-RAP-v48.26-ablation-metrics.csv`
- `OC-RAP-v48.27-run-commands-ZH.txt`
- `README.md`

## 本地验证

- pytest：242 passed，5 warnings；
- Python compileall：通过；
- 全部 shell `bash -n`：通过；
- 当前环境无真实 WOMD/Waymax 与两张 A30，未声称 v48.27 已通过 gate 或取得闭环改善。
