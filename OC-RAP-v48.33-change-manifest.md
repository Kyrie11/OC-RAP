# OC-RAP v48.33 ELIGIBLE-SET-POLICY 代码变更清单

## 发布目标

v48.33 在不引入 regime ID、不降低 gate 阈值、不重构数据集的前提下，修复 v48.32.1 中训练、checkpoint、开发集规则拟合与运行时策略之间的选择合同错位，并将统一 proposal 从 top-3 扩展为 top-5，使严格预注册 Near fit 合同具备必要的 proposal 支持度。

## 核心算法变更

### 1. Eligible-set policy objective

文件：

- `src/ocrap/models/losses.py`
- `src/ocrap/cli/train.py`
- `scripts/train_ocrap_v48_trac_sr.sh`

变更：

- 在统一 frozen top-5 proposal 内，先由连续 opportunity/harm 头形成 differentiable soft eligibility；
- student categorical score 使用 `admission evidence + log soft eligibility`；
- nominal 保持显式 abstention 类；
- teacher distribution 使用连续 safe utility；
- eligible-set KL 同时向 benefit、受支持的 safety-slack component 和 admission head 传播梯度；
- hard validation 与 runtime 统一为 `rank top-k -> eligibility filter -> evidence rerank -> one action or nominal`。

### 2. 统一 top-5 proposal

文件：

- `scripts/run_v48_33_eligible_set_policy_dedicated.sh`
- `scripts/adapt_ocrap_v48_33_eligible_set_policy_variant.sh`
- `scripts/adapt_ocrap_v48_33_eligible_set_policy_single_stage.sh`
- `scripts/calibrate_v48_33_certificate_pool.sh`
- v48.33 contract checkers

变更：

- 所有 regime、variant、训练阶段、checkpoint metric、规则拟合、certificate 和 runtime metadata 均使用 `proposal_top_k=5`；
- 不按 Safe/Near/Contact 分支设置不同 top-k；
- top-k 纳入训练合同和缓存身份，因此 v48.32.1 的 top-3 Stage-1 checkpoint 禁止复用。

### 3. 两阶段默认训练

变更：

- Stage-1：自然总体、无放回，学习 raw benefit 与连续 signed safety margin；
- Stage-2：联合学习 exact eligible-set one-action policy；
- 关闭默认 adaptive teacher-gap margin；
- 关闭默认 admission-only Stage-3，因为 v48.32 消融未显示净收益且多次选择 epoch 0。

默认主实验参数：

- identity epochs：24
- patience：6
- identity learning rate：`6e-5`
- parameter anchor：`0.25`
- eligible-policy weight：`1.25`
- safe-utility listwise weight：`0.50`
- fixed hardest-negative weight：`1.0`
- frontier weight：`0.25`

## 工程与协议修复

### 1. 严格 fit/verify 阈值分离

文件：

- `scripts/calibrate_v48_33_certificate_pool.sh`
- `tools/check_v48_33_metric_calibration_contract.py`

变更：

- development rule fitting 使用 `GATE_SPEC.json` 中预注册的 strict fit 阈值；
- certificate verification 阈值不再被错误复用于 fit；
- certificate 访问前核对阈值、group 数、safe-opportunity 数、top-k、rerank 和 selection semantics。

### 2. checkpoint 与部署动作顺序一致

文件：`src/ocrap/cli/train.py`

变更：

- hard checkpoint metric 不再先取 evidence top-1 再检查安全；
- 先构造 eligible candidate set，再在其中按 evidence 选一次动作；
- 没有 eligible candidate 时显式回退 nominal；
- soft early-stopping categorical mass 同样纳入 opportunity/harm eligibility。

### 3. fail-closed contract 工具

新增：

- `tools/check_v48_33_metric_calibration_contract.py`
- `tools/check_v48_33_multigroup_loss_contract.py`
- `tools/check_v48_33_training_contract.py`
- `tools/extract_v48_33_failure_signature.py`
- `tools/materialize_v48_33_factor_cache.py`

检查范围：

- 多 group forward/backward、有限 loss、各头梯度；
- strict fit 阈值与 GATE_SPEC 一致；
- proposal top-k 全链一致；
- selection order 与 evidence rerank 一致；
- train/dev index、support contract、checkpoint SHA 与缓存身份一致；
- failure stage 与异常签名结构化输出。

## 新增运行脚本

- `scripts/run_v48_33_eligible_set_policy_dedicated.sh`
- `scripts/adapt_ocrap_v48_33_eligible_set_policy_variant.sh`
- `scripts/adapt_ocrap_v48_33_eligible_set_policy_single_stage.sh`
- `scripts/calibrate_v48_33_certificate_pool.sh`
- `scripts/run_v48_33_eligible_set_policy_ablations.sh`
- `scripts/run_v48_33_dev_shadow_closed_loop.sh`
- `scripts/run_v48_33_stress_if_authorized.sh`

运行授权：

- `RC=30`：停止，不运行消融、shadow、test、stress；
- `RC=20`：只允许 v48.33 消融和 adaptation-dev physical shadow；
- `RC=0`：只执行自动生成的 `NEXT_COMMANDS.txt`。

## v48.33 消融设计

仅在有效主实验 `RC=20` 后运行，所有组统一 top-5、关闭 adaptive margin 和 Stage-3：

1. A：admission-only，无 eligible-policy；
2. B：joint coupled，无 eligible-policy；
3. C：admission-only，有 eligible-policy；
4. D：joint coupled，有 eligible-policy（完整模型）。

每组均包含 Balanced 和 Precision，共八个任务。Stage-1 必须精确复用同一主实验 top-5 factor checkpoint，避免重复训练与合同漂移。

## 新增测试与交付审计

- `tests/test_v48_33_eligible_set_policy.py`
- `OC-RAP-v48.32.1-main-gate-audit.csv`
- `OC-RAP-v48.32-ablation-audit.csv`
- `OC-RAP-v48.32.1-engineering-and-protocol-audit.json`
- `OC-RAP-v48.33-script-dependency-audit.json`
- `OC-RAP-v48.32.1-results-audit-and-v48.33-ELIGIBLE-SET-POLICY-plan-ZH.md`
- `OC-RAP-v48.33-run-commands-ZH.txt`

## 保持不变的设计

- 不输入 regime ID；
- 不为三个 regime 建立独立 planner 或规则；
- Natural population、无放回训练；
- candidate-vs-nominal 连续物理余量；
- support reliability 原则；
- measured hard veto 独立且不可补偿；
- bounded admission；
- raw recoverability benefit 与安全准入语义分离；
- scene-disjoint certificate；
- test/stress roots 在 gate 授权前保持封闭。
