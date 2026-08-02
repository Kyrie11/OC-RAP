# OC-RAP v48.30 SLACK-RANK-BRIDGE 变更清单

## 版本目标

修复 v48.29 admission stage 的 population-prior shift，并用一个不依赖 regime ID 的连续物理非退化语义统一 Safe、Near-contact 与 Contact 的 recovery ranking。

## 核心模型与损失

### `src/ocrap/models/ocrap.py`

- 新增 `direct_recovery_evidence_admission_prior_mode=safety_slack`。
- 新增：
  - `direct_recovery_evidence_slack_temperature`；
  - `direct_recovery_evidence_slack_penalty`。
- 将五个 component logits 投影为 candidate-vs-nominal signed safety margins。
- 统一 admission prior：`benefit - penalty * relu(max_component_margin)`。
- 保留五因子独立 hard veto、bounded admission 和 nominal identity。
- 输出 component margins、worst slack 与 slack barrier 诊断字段。

### `src/ocrap/models/losses.py`

- 新增五因子 signed-margin SmoothL1 regression。
- 保留 component BCE，用于边界方向识别。
- hardest-negative 仍只约束 best safe action、nominal 与 hardest non-safe action。
- 默认主模型不启用 safe-utility listwise/frontier。

### `src/ocrap/cli/train.py`

- 新增 `direct_population_safe_rank_risk` checkpoint metric。
- 指标在自然 adaptation-dev population 上联合惩罚：
  - safe top-1 regret；
  - harmful mass；
  - false admission；
  - safe recall shortfall；
  - safe mass shortfall。
- 持久化 slack temperature/penalty 和 component-margin regression 配置。

### `src/ocrap/models/inference.py`

- checkpoint → inference 完整恢复 safety-slack 参数。
- inference bundle 与运行时配置写回新增参数，防止 silent default fallback。

### `configs/default.yaml`

- 注册 safety-slack 参数与 component-margin regression 权重。

## 训练与执行脚本

### `scripts/train_ocrap_v48_trac_sr.sh`

- `GROUP_BATCHING_REPLACEMENT` 不再硬编码为 true。
- 传递 safety-slack 与 component-margin regression 参数。

### `scripts/adapt_ocrap_v48_30_slack_rank_single_stage.sh`

- 统一单阶段训练入口。
- 写出完整 `STAGE_ARCHITECTURE.json`，包括 sampler、replacement、slack、margin regression 与 checkpoint metric。

### `scripts/adapt_ocrap_v48_30_slack_rank_variant.sh`

- Stage 1：raw benefit + five factor signed margins，无 admission 梯度。
- Stage 2：自然 population、无 replacement，只训练 bounded admission。
- 默认 safe-utility regression + hardest-negative；关闭 listwise/frontier。
- 完成后生成 factor transfer integrity。

### `scripts/run_v48_30_slack_rank_bridge_dedicated.sh`

- 双 A30 Balanced/Precision 并行主实验。
- 固定 top-3、五因子、scale=6、safety-slack、两阶段 epoch 配置。
- 显式将 listwise/frontier 固定为 0，避免环境变量污染主实验。
- certificate 前执行 model/inference contract 与 training contract。
- RC=0/20/30 语义保持不变。

### `scripts/calibrate_v48_30_certificate_pool.sh`

- adaptation-dev 拟合冻结 rule；完整 certificate 仅验证。
- gate 与阈值不修改。

### `scripts/run_v48_30_dev_shadow_closed_loop.sh`

- fast physical shadow 默认关闭在线 OC-MERO relabel。
- 保留真实 policy、Waymax rollout 与物理指标。
- 输出 provenance、runtime contract 和 physical target support。

### `scripts/run_v48_30_parallel_ablations.sh`

- 8 个任务同时启动，GPU0 四个 Balanced、GPU1 四个 Precision。
- 消融：自然 population、signed margin、safety slack、full hardest-negative。
- 每任务 1 worker/1 host thread，降低 CPU/文件系统争抢。

### `scripts/run_v48_30_stress_if_authorized.sh`

- 仅在自动生成 `NEXT_COMMANDS.txt` 后允许读取 held-out stress。

## 新增 fail-closed 工具

- `tools/check_v48_30_model_contract.py`
- `tools/check_v48_30_factor_transfer.py`
- `tools/check_v48_30_training_contract.py`
- `tools/check_v48_30_physical_target_support.py`
- `tools/summarize_v48_30_gate_failure.py`

其中 training contract 强制检查：

- 无 regime routing；
- Stage 2 自然 population；
- Stage 2 无 replacement；
- safety-slack prior；
- bounded admission；
- population checkpoint metric 有限且跨 epoch 变化；
- Stage 1 signed-margin regression；
- factor checkpoint 非 epoch 0；
- factor transfer 有效；
- 五个 harm factors；
- legacy Noisy-OR 关闭。

## 测试

新增 `tests/test_v48_30_slack_rank_bridge.py`，覆盖：

- safety-slack 参数持久化；
- 无 regime routing；
- signed-margin hinge 语义；
- margin regression；
- population checkpoint metric；
- natural population/no replacement；
- 8 任务并发消融配置。

全量结果：

```text
265 passed, 5 warnings
compileall PASS
all shell bash -n PASS
```

## 审计与运行文档

代码包根目录新增：

- `OC-RAP-v48.29-results-audit-and-v48.30-SLACK-RANK-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.30-run-commands-ZH.txt`
- `OC-RAP-v48.29-main-gate-metrics.csv`
- `OC-RAP-v48.29-ablation-metrics.csv`
- `OC-RAP-v48.29-dev-shadow-metrics.csv`
- `OC-RAP-v48.29-admission-training-trajectory.csv`
- `OC-RAP-v48.28-v48.29-gate-comparison.csv`
- `OC-RAP-v48.29-results-audit-summary.json`
- 更新后的 `ALGORITHM_CHANGELOG.md`
