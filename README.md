# OC-RAP 完整实现说明

本仓库实现论文中的 **Observation-Consistent Recovery-Affordance Planner, OC-RAP**，并用 `论文细节.md` 中的 implementation spec 修正论文正文里的概念化描述。代码把 recoverability 作为 candidate-prefix admission criterion，而不是普通 motion prediction：每个样本以 `(scene, time, candidate prefix)` 为单位，显式保存 `M_star / Y_obs / C_star / root_probs / R_orc_star / R_dep_star / I_art_star / regime_label`，用于训练 observation-consistent recovery admission。

实现重点：

1. **Dataset construction**：从 WOMD TFRecord 或 synthetic smoke 数据构造 scene-prefix 样本；每个 prefix 生成 replay、reactive、targeted perturbation 三类 counterfactual futures。
2. **Recovery teacher**：对每个 future 和 recovery option rollout，使用 active-mask corrected margin，分别计算 clearance、stop、control、route、harm、stability、secondary collision slacks。
3. **Root clustering**：按 recovery signature 聚类，不按 raw trajectory 距离聚类。
4. **Observation compatibility**：用 post-prefix observation renderer 生成 `Y_obs` 和 `C_star`，不把 pairwise threshold 当作严格 equivalence relation。
5. **OC-MERO**：使用 `w_ij = normalize(C_ij * p_j)`，并用 weighted lower-tail LCVaR 计算 oracle/deployable recoverability。
6. **CRISP selector**：先保留 admissible nominal；否则在 admissible set 内按 nominal utility 选；若无 admissible action，用 lexicographic fallback，而不是直接最大化 `R_dep - lambda * D`。
7. **Calibration**：只使用 calibration split 中 `R_dep_star < 0` 的 negative deployability samples 估计 threshold。
8. **Metrics**：同时报告 candidate-level FRA、executed-action FRA、ODG/ODG+、DRS、nominal regret、bounded NUP、intervention、collision/harm proxies，并支持 regime-wise 和 artifact subset 报告。
9. **Ablation switches**：所有 Experiments 中的 w/o 设置都可以通过 CLI 开关运行。

---

## 1. 安装

```bash
cd ocrap_impl
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

用于本地快速验证只需要基础依赖：`numpy`, `torch`, `PyYAML`, `tqdm`。

WOMD TFRecord 解析需要额外依赖：

```bash
pip install -e '.[womd]'
```

Waymax 场景加载/闭环仿真适配需要额外依赖：

```bash
pip install -e '.[waymax]'
```

当前代码中 `womd.py` 提供 WOMD Scenario proto parser；`iter_waymax_scenarios` 使用 Waymax 官方 dataloader 入口，便于把本仓库的数据构造和 Waymax simulator state 接入同一实验环境。WOMD/Waymax 的数据访问权限和 Google Cloud 配置需要由本地环境完成。

---

## 2. 仓库结构

```text
ocrap_impl/
  configs/
    default.yaml                         # 默认实验参数
    ablations/                           # 每个 w/o 实验的配置片段
  src/ocrap/
    cli.py                               # 统一命令入口
    data/                                # 数据构造/加载/诊断 namespace re-export
    models/                              # 模型 namespace re-export
    training/                            # 训练与 loss namespace re-export
    evaluation/                          # 评估/校准/指标 namespace re-export
    planning/                            # prefix 选择/部署 namespace re-export
    schema.py                            # RawScenario / SceneHistory / DatasetSample schema
    womd.py                              # WOMD TFRecord parser 与 Waymax loader adapter
    synth.py                             # synthetic smoke-test scenes
    dataset_builder.py                   # Algorithm 1: dataset construction
    prefix_generation.py                 # candidate prefix generation + utility
    futures.py                           # replay / reactive / targeted futures
    teacher.py                           # Algorithm 2: recovery teacher margins
    observation.py                       # post-prefix observation renderer + compatibility labels
    root_clustering.py                   # recovery-signature clustering
    lcv.py                               # weighted lower-tail LCVaR + calibration quantile
    ocmero.py                            # Algorithm 3: corrected OC-MERO
    model.py                             # PyTorch OC-RAP model
    losses.py                            # assign/sig/IB/obs/margin/anti-oracle/utility losses
    train.py                             # training loop
    calibrate.py                         # calibration threshold estimation
    selector.py                          # CRISP selector
    metrics.py                           # FRA/ODG/DRS/NUP/intervention metrics
    evaluate.py                          # baseline and OC-RAP evaluation
    deploy.py                            # deployment-time candidate selection
    diagnose.py                          # dataset consistency checks
  tests/test_core.py                     # unit tests for core formulas/operators
```

---

## 3. 一键 smoke test

Synthetic 数据只用于验证代码链路，不替代论文主实验。它会构造小规模 scene-prefix dataset，跑 diagnose、训练、评估和部署。

```bash
# 1) 构造最小 dataset
PYTHONPATH=ocrap python -m ocrap.cli \
  --set data_source=synthetic \
  --set num_synthetic_scenarios=1 \
  --set split_ratios.train=1.0 \
  --set split_ratios.val=0.0 \
  --set split_ratios.calibration=0.0 \
  --set split_ratios.test=0.0 \
  --set num_candidate_prefixes=2 \
  --set num_reactive_futures=1 \
  --set num_targeted_futures=1 \
  --set num_roots=4 \
  --set max_times_per_scenario=1 \
  --set max_biased_times_per_scenario=0 \
  --set bev_resolution_m=8.0 \
  --set max_agents=8 \
  build-dataset --output runs/smoke_dataset

# 2) 诊断 dataset schema、source coverage、split leakage、compatibility labels
PYTHONPATH=ocrap python -m ocrap.cli \
  diagnose --dataset runs/smoke_dataset --output runs/smoke_dataset/diagnose.json

# 3) 训练 1 epoch
PYTHONPATH=ocrap python -m ocrap.cli \
  --set training.epochs=1 \
  --set training.batch_size=1 \
  --set num_roots=4 \
  train --dataset runs/smoke_dataset --output runs/smoke_train

# 4) 在 train split 上做流程检查
PYTHONPATH=ocrap python -m ocrap.cli \
  --set num_roots=4 \
  evaluate --dataset runs/smoke_dataset \
  --checkpoint runs/smoke_train/best.pt \
  --split train \
  --output runs/smoke_train/eval_train.json

# 5) 部署式选择某个 scene-time 的 candidate prefix
PYTHONPATH=ocrap python -m ocrap.cli \
  --set num_roots=4 \
  deploy --dataset runs/smoke_dataset \
  --checkpoint runs/smoke_train/best.pt \
  --scene-id synthetic_000000 \
  --time-index 10 \
  --output runs/smoke_train/deploy.json
```

运行单元测试：

```bash
PYTHONPATH=ocrap pytest -q
```

---

## 4. WOMD 数据集构造

WOMD 模式使用 `waymo_open_dataset.protos.scenario_pb2.Scenario` 解析 Scenario proto。代码读取：

- `scenario_id`
- `timestamps_seconds`
- `sdc_track_index`
- `tracks[*].states`：position、velocity、heading、dimensions、valid flag、object type
- `dynamic_map_states`
- `map_features`：lane / road line / road edge / crosswalk / speed bump / stop sign / driveway
- `sdc_paths`，若存在；不存在时退化为 SDC logged route proxy

构造命令：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --set data_source=womd \
  --set womd_patterns='/path/to/womd/training/*.tfrecord*' \
  --set max_scenarios=1000 \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set local_radius_m=80.0 \
  --set bev_resolution_m=0.5 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=4 \
  --set num_targeted_futures=8 \
  --set num_roots=8 \
  build-dataset --output data/ocrap_womd
```

输出目录：

```text
data/ocrap_womd/
  manifest.csv
  dataset_summary.json
  samples/*.npz
```

每个 `.npz` 是一个 `(scene_id, time_index, candidate_index)` 样本，核心字段包括：

```text
agent_history               [T_h, A, F_agent]
agent_valid                 [T_h, A]
map_polylines               [P, Q, F_map]
map_valid                   [P, Q]
dynamic_map                 [T_h, B, F_signal]
route                       [R, F_route]
bev_occ                     [C_occ, H_bev, W_bev]
prefix_states              [T_p, F_ego]
prefix_controls            [T_p-1, F_ctrl]
prefix_macro_id            scalar
prefix_param                [F_param]
future_probs                [J]
root_assignments            [J]
root_probs                  [K]
root_signature              [K, D_sig]
root_future_signature       [K, D_future_sig]
root_valid                  [K]
future_to_root_weight       [J, K]
y_obs                       [K, K]
c_star                      [K, K]
m_star                      [K, L]
option_valid                [L]
r_orc_star                  scalar
r_dep_star                  scalar
oracle_gap_star             scalar
i_art_star                  scalar bool
regime_label                dict
split_id                    train / val / calibration / test
```

---

## 5. Dataset diagnose

构造后必须先诊断：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  diagnose --dataset data/ocrap_womd \
  --output data/ocrap_womd/diagnose.json

# 更贴近论文 idea 的快速检查：oracle artifact、ODG、regime 分布和观测一致性是否足够支撑实验
PYTHONPATH=ocrap python -m ocrap.cli \
  papercheck --dataset data/ocrap_womd \
  --output data/ocrap_womd/papercheck.json
```

诊断会检查：

- 必要 label 是否存在：`M_star / Y_obs / C_star / R_orc_star / R_dep_star / I_art_star / root_probs`
- scenario-level split 是否泄漏
- 每个样本是否包含 replay、reactive、targeted 三类 future source
- `root_probs` 是否归一化
- `C_star` 是否为方阵且对角为 1
- `Y_obs` 是否对称
- hidden emergence 是否标记 `from_unknown_mask=True`
- artifact fraction 与 regime distribution

---

## 6. 模型训练

完整训练：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --config configs/default.yaml \
  train --dataset data/ocrap_womd \
  --output runs/ocrap_full
```

常用训练 override：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --set training.epochs=20 \
  --set training.batch_size=32 \
  --set training.lr=0.0005 \
  --set training.artifact_sampler_weight=0.30 \
  train --dataset data/ocrap_womd \
  --output runs/ocrap_full
```

训练输出：

```text
runs/ocrap_full/
  config.yaml
  train_history.json
  best.pt
  last.pt
```

训练 objective：

```text
L = lambda_assign * L_assign
  + lambda_sig    * L_sig
  + lambda_ib     * L_ib
  + lambda_obs    * L_obs
  + lambda_margin * L_margin
  + lambda_art    * L_art
  + lambda_util   * L_util
```

其中：

- `L_assign = - sum_j p_j log p_hat[root_assignment_j]`
- `L_sig = sum_k p_k ||s_hat_k - s_k_star||`
- `L_obs` 使用 pairwise class-imbalance weighted BCE
- `L_margin` 使用 Huber regression，按 `root_probs` 和 `option_valid` 加权
- `L_art` 只在 `I_art_star=1` 的 oracle artifact 样本上惩罚过高 `R_dep_pred`
- `L_ib` 是 root posterior 到 prefix-conditioned prior 的 KL

---

## 7. Calibration

使用 calibration split 中 teacher deployability negative 的样本：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  calibrate --dataset data/ocrap_womd \
  --checkpoint runs/ocrap_full/best.pt \
  --output runs/ocrap_full/calibration.json
```

输出示例：

```json
{
  "num_calibration": 12000,
  "num_negative": 2500,
  "thresholds": {
    "0.01": 0.83,
    "0.05": 0.41,
    "0.1": 0.22
  },
  "strict_finite_sample": true
}
```

Finite-sample threshold 实现：

```python
scores = sorted(scores_neg)
k = ceil((n + 1) * (1 - delta))
if k > n:
    gamma = +inf
else:
    gamma = scores[k - 1]
```

---

## 8. Evaluation

主评估：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  evaluate --dataset data/ocrap_womd \
  --checkpoint runs/ocrap_full/best.pt \
  --calibration runs/ocrap_full/calibration.json \
  --split test \
  --output runs/ocrap_full/test_metrics.json
```

默认评估方法：

```text
nominal
risk_aware
backup_filter
contingency
oracle_filter
ocrap
```

输出包括：

```text
FRA_cand                 candidate-level false recoverability admission
FRA_exec                 selected-action false recoverability admission
DRS                      root-probability weighted deployable recovery success
ODG                      R_orc_star - R_dep_star
ODG_pos                  max(0, R_orc_star - R_dep_star)
nominal_regret           U(a0) - U(a*)
NUP                      exp(-max(0, regret)/sigma_U)
intervention_rate        selected action != nominal
collision_rate           selected prefix hard violation proxy
hard_violation           hard-rule violation score
harm_proxy               severity proxy
artifact_selection_rate  selected action is oracle artifact
```

Regime-wise 结果保存在：

```json
{
  "methods": {...},
  "regime": {
    "ocrap": {
      "normal": {...},
      "low_headroom": {...},
      "occluded": {...},
      "near_contact": {...},
      "post_contact": {...},
      "oracle_artifact": {...}
    }
  }
}
```

---

## 9. Deployment / online prefix selection

对一个 scene-time 的 candidate set 执行 CRISP selection：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  deploy --dataset data/ocrap_womd \
  --checkpoint runs/ocrap_full/best.pt \
  --calibration runs/ocrap_full/calibration.json \
  --scene-id '<scenario_id>' \
  --time-index 42 \
  --delta 0.05 \
  --output runs/ocrap_full/deploy_scene42.json
```

输出：

```json
{
  "selected_candidate_index": 0,
  "admitted_indices": [0, 3, 7],
  "reason": "nominal_admitted",
  "candidates": [
    {
      "candidate_index": 0,
      "macro_name": "nominal",
      "utility": 8.4,
      "r_dep_pred": 0.72,
      "r_orc_pred": 0.91,
      "hard_violation": 0.0,
      "harm_proxy": 0.0,
      "feasible": true
    }
  ]
}
```

CRISP admission set：

```text
A_adm = { a : R_dep_pred(a) >= gamma_rec,
              H(a) <= gamma_H,
              D(a) <= gamma_D,
              feasible(a) = 1 }
```

Fallback 顺序：

```text
1. 删除动态不可行 prefix
2. 最小 hard violation H(a)
3. 最小 harm D(a)
4. 最大 R_dep_pred(a)
5. 最大 U(a)
```

---

## 10. Experiments 中 w/o 消融开关

所有消融都可用全局 CLI switch，不需要改代码。

### 10.1 OC-RAP w/o observation kernel

Roots branch-wise evaluation，不使用 post-prefix observation compatibility：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --without-observation-kernel \
  train --dataset data/ocrap_womd \
  --output runs/wo_observation_kernel
```

评估同样加开关，确保 checkpoint flags 和 evaluation flags 一致：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --without-observation-kernel \
  evaluate --dataset data/ocrap_womd \
  --checkpoint runs/wo_observation_kernel/best.pt \
  --split test \
  --output runs/wo_observation_kernel/test_metrics.json
```

### 10.2 OC-RAP w/o lower-tail aggregation

用 weighted mean 替代 LCVaR：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --without-lower-tail \
  train --dataset data/ocrap_womd \
  --output runs/wo_lower_tail
```

### 10.3 OC-RAP w/o calibration

使用 `selection.fixed_gamma_rec`，不读 calibration JSON：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --without-calibration \
  --set selection.fixed_gamma_rec=0.0 \
  evaluate --dataset data/ocrap_womd \
  --checkpoint runs/ocrap_full/best.pt \
  --split test \
  --output runs/wo_calibration/test_metrics.json
```

### 10.4 OC-RAP w/o anti-oracle loss

训练时去掉 `L_art`：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --without-anti-oracle \
  train --dataset data/ocrap_womd \
  --output runs/wo_anti_oracle
```

### 10.5 Full-future root model

Root signature head 改为 future-trajectory signature target，而不是 recovery signature：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --full-future-roots \
  train --dataset data/ocrap_womd \
  --output runs/full_future_roots
```

### 10.6 No occlusion BEV

模型忽略 BEV occlusion/unknown-space input：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --no-occlusion-bev \
  train --dataset data/ocrap_womd \
  --output runs/no_occlusion_bev
```

也可以使用配置片段：

```bash
PYTHONPATH=ocrap python -m ocrap.cli \
  --config configs/ablations/no_occlusion_bev.yaml \
  train --dataset data/ocrap_womd \
  --output runs/no_occlusion_bev
```

配置片段只包含消融字段；完整实验建议优先用 `configs/default.yaml` 加 CLI switch。

---

## 11. 关键实现对应关系

| 论文/细节对象 | 代码位置 |
|---|---|
| scene-prefix dataset schema | `schema.py`, `dataset_builder.py` |
| scenario-level split | `split.py` |
| candidate prefixes | `prefix_generation.py` |
| replay/reactive/targeted futures | `futures.py` |
| recovery options `g_l=(r_l, theta_l)` | `recovery_options.py` |
| teacher rollout and active-mask margins | `teacher.py` |
| post-prefix observation renderer | `observation.py` |
| pairwise compatibility labels | `observation.py::compatibility_labels` |
| recovery-signature root clustering | `root_clustering.py` |
| weighted lower-tail LCVaR | `lcv.py::weighted_lcvar` |
| corrected OC-MERO | `ocmero.py::oc_mero` |
| anti-oracle loss | `losses.py::anti_oracle_loss` |
| CRISP selector | `selector.py::crisp_select` |
| calibration threshold | `calibrate.py`, `lcv.py::finite_sample_upper_quantile` |
| FRA/ODG/DRS/NUP metrics | `metrics.py`, `evaluate.py` |
| deployment selection | `deploy.py` |

---

## 12. WOMD/Waymax 与 post-contact surrogate 边界

WOMD/Waymax 输入是 tracked bounding boxes 和 map，不是 raw-sensor perception。代码中的 occlusion 是 synthetic/map/FOV/dynamic-box observation model。WOMD/Waymax 上的 contact、impact、secondary-collision、post-contact yaw/stability 都按 bounding-box + kinematic surrogate 计算。物理级 post-impact validation 应在 MetaDrive/CARLA/custom physics stress simulator 中单独报告，并且只有显式纳入 calibration 的 distribution 才能声明 calibration false-admission control。

---

## 13. 常用参数

默认参数在 `configs/default.yaml`：

```yaml
sample_rate_hz: 10
history_horizon_s: 1.0
prefix_horizon_s: 1.0
recovery_horizon_s: 4.0
local_radius_m: 80.0
bev_resolution_m: 1.0
num_candidate_prefixes: 24
num_reactive_futures: 4
num_targeted_futures: 8
num_roots: 8
num_recovery_options: 24
ocmero:
  alpha: 0.2
  beta: 0.2
artifact:
  gamma_orc: 0.0
  gamma_dep: 0.0
calibration:
  deltas: [0.01, 0.05, 0.10]
```

提高构造速度可临时调大 `bev_resolution_m`、降低 `max_agents`、`num_candidate_prefixes`、`num_reactive_futures`、`num_targeted_futures`；论文主实验应使用默认或 appendix 中声明的正式参数。

---

## 14. 已验证命令

在当前环境中完成了以下检查：

```bash
python -m compileall -q ocrap
PYTHONPATH=ocrap pytest -q
PYTHONPATH=ocrap python -m ocrap.cli ... build-dataset
PYTHONPATH=ocrap python -m ocrap.cli diagnose ...
PYTHONPATH=ocrap python -m ocrap.cli train ...
PYTHONPATH=ocrap python -m ocrap.cli evaluate ...
PYTHONPATH=ocrap python -m ocrap.cli deploy ...
```
