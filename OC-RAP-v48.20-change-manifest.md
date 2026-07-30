# OC-RAP v48.20 UNISON-BRIDGE 代码变更清单

## 版本目标

v48.20 不放宽 v48.19 Natural gate、不读取 test、不重建三个 regime 数据集。它修复 v48.19 的训练—部署错位，并以一个不接收 regime ID 的统一模型同时处理 Safe、Near-contact 和 Contact。

## 修改文件

### `configs/default.yaml`

新增统一专家、component harm heads、component scale、component auxiliary、global balance 和 safe-set temperature 配置；默认关闭以保证旧 checkpoint 兼容。

### `scripts/train_ocrap_v48_trac_sr.sh`

把新增模型与 loss 配置透传到训练 CLI；`balanced_replaces_erm` 默认保持 false。

### `src/ocrap/cli/train.py`

- 把 component harm logits 传入统一 loss；
- harmful membership 从 `>= 0` 修复为严格 `> 0`；
- 增加 `direct_unison_selection_risk` 作为不参与推理路由的 worst-regime checkpoint 指标；
- 保存/恢复 UNISON 模型结构参数。

### `src/ocrap/models/ocrap.py`

- 新增 bucket-invariant unified expert evidence 分支；
- 每个候选同时使用两套冻结 source expert 输出；
- benefit 使用精确 expert `min` 加有界共享 residual；
- harm 使用零初始化 DRS/DEP/GAP component heads，精确 `max` 聚合；
- nominal pinning 与旧 DUET/FACET checkpoint 兼容；
- 不向新分支暴露 regime/bucket ID。

### `src/ocrap/models/losses.py`

- 新增 component-specific targets 与 BCE auxiliary；
- 新增 global bucket-agnostic balancing；
- 候选级 balance 不再替换 group ERM；
- group safe set 定义为 `beneficial AND not component-harmful`；
- safe-set、selective harmful mass 和 coverage 使用与部署完全一致的冻结 top-k 和 `sigmoid(benefit)-sigmoid(harm)`；
- top-k 外候选不接收 safe-set group gradient；
- signed PCD 三分类不再污染 component hard mask/intragroup harm。

### `src/ocrap/models/inference.py`

加载并回填 UNISON checkpoint 模型参数，保证训练/证书/闭环结构一致。

### `src/ocrap/external_baselines/train.py`

使用 float64 聚合梯度范数再裁剪，避免有限 float32 梯度在 norm reduction 中溢出并把更新静默清零。

## 新增脚本

- `scripts/adapt_ocrap_v48_20_unison_variant.sh`
- `scripts/calibrate_v48_20_certificate_pool.sh`
- `scripts/calibrate_v48_20_learning_gate.sh`
- `scripts/run_v48_20_unison_dedicated.sh`
- `scripts/run_v48_20_parallel_ablations.sh`
- `scripts/run_v48_20_stress_if_authorized.sh`

主实验 Balanced/Precision 分别使用 GPU0/GPU1。消融每波四任务并发、每张 A30 两任务，Balanced 与 Precision 分两波执行。

## 新增审计与测试

### `tools/audit_external_baseline_artifacts_v48_20.py`

审计 closed-loop progress、summary 和 scene journal 的完成状态与数量一致性；不完整工件不得用于论文。

### `tests/test_v48_20_unison_bridge.py`

覆盖：

- bucket-invariant 输出与排序；
- nominal pinning 和 bounded residual；
- 精确 conservative min/max envelope；
- harm semantic reset；
- factorized component labels/masks；
- group ERM 不被 candidate balance 替换；
- deployment-exact top-k score；
- top-k 外候选无 safe-set gradient；
- legacy v48.18/v48.19 identity 兼容。

## 协议与返回码

- Gate 规格沿用可行且已绑定 manifest 的 v48.19 独立协议；没有通过降低 LCB、提高 UCB 或减少最小选择数获得结果。
- `RC=0`：有效 pass，允许生成并执行 `NEXT_COMMANDS.txt`。
- `RC=20`：有效算法拒绝，运行新消融。
- `RC=30`：工程/协议/工件失败，只排查工程。

## 本地验证

- pytest：196 passed，5 warnings；
- compileall：通过；
- 全部 Shell `bash -n`：通过；
- 未在交付环境运行真实 WOMD/Waymax/A30 实验。
