# OC-RAP v48.25 → v48.26 EXECUTION-PHYSICS-BRIDGE 变更清单

## 目标

v48.26 首先修复使 v48.25 `RC=30` 且妨碍算法归因的工程故障，然后把 checkpoint、推理、certificate 和 closed-loop 使用的策略合同统一为同一套 proposal-contained safe-positive / executable safe-utility 定义。该版本不降低 Natural gate，也不删除 certificate。

## 直接修复的工程故障

- `tools/calibrate_policy_risk_v48.py`
  - 修复 `pathlib.PosixPath` 导致的最终 JSON 序列化崩溃；
  - 对 Path、NumPy scalar/array、tuple、set 和嵌套容器递归 JSON-safe；
  - 结构支持或 learned selector 合法拒绝返回 worker 3，空 population/协议错误返回 worker 4；
  - 写入 dev-frozen rule 的约束满足状态与 SHA256 provenance。

- `src/ocrap/models/inference.py`
  - 恢复 `direct_recovery_evidence_frontier`；
  - 恢复 `direct_recovery_evidence_component_prior_logit`；
  - 恢复 `direct_recovery_evidence_admission_bounded`；
  - 推理后执行 expected/actual model-contract fail-closed 检查。

- `src/ocrap/cli/calibrate.py`
  - 增加 `calibration.exact_split_ids`；
  - Safe calibration、adaptation-dev 与 certificate 不再通过语义 alias 交叉展开 split。

- `scripts/run_v48_26_execution_physics_dedicated.sh`
  - certificate 前检查模型构造合同；
  - 保持 0/20/30 三态；
  - 独立构建 adaptation-dev teacher index；
  - Balanced/Precision 分别固定到两张 A30。

## 算法合同修复

- `src/ocrap/cli/train.py`
  - checkpoint 保存完整 frontier/prior/admission 配置；
  - checkpoint opportunity 改为冻结 proposal 内的 safe-positive；
  - harmful action 不得计为 positive admission；
  - 新增 safe-positive recall、safe admission precision、invalid admission、safe top-1 accuracy/regret；
  - checkpoint 风险直接惩罚 safe recall shortfall、precision shortfall、invalid admission 与 safe regret。

- `src/ocrap/models/losses.py`
  - safe-utility 训练改为运行时精确分数 `sigmoid(admission_delta)-0.5`；
  - teacher target 裁剪到同一 `[-0.5,0.5]` 执行范围。

- `scripts/adapt_ocrap_v48_26_execution_physics_variant.sh`
  - top-3 frozen proposal；
  - semantic low-risk prior；
  - centred/unbounded admission；
  - nominal+top-k categorical one-action objective；
  - continuous safe-utility；
  - legacy Noisy-OR 关闭；
  - exact safe-positive group batching。

## Certificate 与授权

- `scripts/calibrate_v48_26_certificate_pool.sh`
  - rule 只在 `evidence_adapt_dev` 拟合并冻结；
  - 完整 `certificate_pool` 只做 verification；
  - certificate 标签不参与调参；
  - 输出 `SAFE_REGIME_STATUS.json`，明确 Safe 当前没有独立 policy Natural gate；
  - 只有 Near 与 Contact 均通过时才自动生成 `NEXT_COMMANDS.txt`。

- `scripts/repair_v48_25_certificate_with_v48_26.sh`
  - 无需重训，使用修复后的 inference/serialization 重新评估服务器上的 v48.25 checkpoint；
  - 该结果只用于诊断，不代表 v48.26 训练结果。

- `scripts/run_v48_26_stress_if_authorized.sh`
  - 缺少自动生成的 `NEXT_COMMANDS.txt` 时拒绝读取 test/stress。

## Near-contact 物理指标

- `src/ocrap/simulation/closed_loop_runner.py`
  - near/critical exposure episode count；
  - longest continuous exposure run；
  - time-to-min clearance/TTC；
  - terminal clearance/TTC；
  - clearance/TTC recovery gain；
  - clearance/TTC deficit AUC；
  - acceleration p95/max、maximum deceleration、jerk max、yaw-rate max。

## Contact 物理指标

- 使用 post-contact target 的显式 step-0 causal anchor；
- secondary/re-contact event、episode、scene rate；
- overlap episode count、duration、longest run；
- normalized free-space AUC、clearance-deficit AUC；
- terminal clearance、clearance gain、time-to-peak；
- sustained escape、time-to-escape；
- stable-stop quality 与 time-to-quality-stable-stop；
- stable stop 同时约束速度、overlap、offroad、yaw-rate 和持续时间；
- 精确 bucket alias，防止 `near_contact` 被误识别为 Contact。

## 比较与目标检查

- `tools/compare_paired_closed_loop.py`
  - 增加新增物理指标的 paired delta、bootstrap CI 和方向定义。

- `tools/check_v48_26_regime_targets.py`
  - 检查 Near/Contact 的投稿目标与 non-inferiority 项；
  - dev shadow 输出仅供开发诊断。

- `scripts/run_v48_26_dev_shadow_closed_loop.sh`
  - 只读取 `evidence_adapt_dev`；
  - Balanced/Precision 双卡并发，每卡 Near/Contact 顺序运行；
  - 不读取 certificate/test/stress。

## 消融与双卡调度

- `scripts/run_v48_26_parallel_ablations.sh`
  - `A_engineering_contract_only`；
  - `B_add_safe_checkpoint_contract`；
  - `C_add_execution_exact_safe_utility`；
  - `D_full_execution_physics_bridge`；
  - 四个 wave；每个 wave Balanced→GPU0、Precision→GPU1；最大并发 2；每张 A30 四个任务。

## 新增测试

- `tests/test_v48_26_execution_physics_bridge.py`
  - certificate JSON-safe；
  - model-contract parity；
  - safe-positive checkpoint 定义；
  - execution-exact safe utility；
  - exact split IDs；
  - Near/Contact bucket 区分；
  - Near/Contact 新物理字段与方向。

## 文档与审计工件

- `OC-RAP-v48.25-results-audit-and-v48.26-EXECUTION-PHYSICS-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.25-results-audit-summary.json`
- `OC-RAP-v48.25-training-metrics-v48.24-comparison.csv`
- `OC-RAP-v48.26-run-commands-ZH.txt`
- `OC-RAP-v48.26-ALGORITHM_CHANGELOG.md`
- 根目录 `ALGORITHM_CHANGELOG.md` 已同步新增 v48.26 记录。

## 未做的修改

- 未降低 Natural gate 的 precision/support/harm 要求；
- 未删除 certificate；
- 未重建数据集；
- 未扩大 proposal 到 top-8；
- 未在缺少 candidate-level physical teacher 标签时伪造 Contact/Near 物理辅助训练目标；
- 未声称本地环境已经验证 RC=0 或 CCF-A 闭环目标。
