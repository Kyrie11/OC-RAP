# OC-RAP：Observation-Consistent Recovery-Affordance Planner

本仓库是对论文 `post-collision.tex` 中 OC-RAP 方法的可运行实现。实现以论文的 `abstract -> introduction -> method -> appendix -> experiments` 逻辑为主线，并按照 `代码完善指令.md` 对原始代码进行了重构、补全和修复：数据构造、candidate prefix 生成、counterfactual future mining、observation-consistent root 聚类、recovery teacher、OC-MERO、CRISP selector、训练、校准、评估和 papercheck 都已落到 `src/ocrap` 包结构中。

核心目标不是做普通 motion prediction，而是为每个 `(scene, time, candidate_prefix)` 判断：在 prefix 执行后、只基于 post-prefix observation 可区分的未来簇中，是否存在可恢复的 recovery option；以及是否存在 oracle recoverable 但 deployable 不可恢复的 observation aliasing artifact。

## 已修复 / 已补全的关键点

- **代码结构**：重构为 `src/ocrap/{cli,config,data,planning,simulation,roots,models,algorithms,evaluation,utils}`，并保留少量顶层兼容 shim，避免旧扁平包结构遮蔽新实现。
- **WOMD 读取**：新增纯 Python / PyTorch 友好的 TFRecord reader，包含 length/data CRC 校验、gzip 读取、glob、多 shard 迭代和 resume 入口；不再使用 TensorFlow runtime parser。
- **SDC-first schema**：WOMD parser 会把原始 SDC track 放到 agent index `0`，并在 metadata 中记录 `original_sdc_track_index` 和 `agent_index_map`。
- **agent / map / dynamic map 特征**：agent feature 扩展到 `[x,y,z,vx,vy,ax,ay,heading,sin,cos,length,width,height,type,valid,confidence]`；map feature 覆盖 route flag、speed limit、traffic control、valid mask；dynamic map 支持多 lane state。
- **candidate prefix**：用 route-local lattice 和 bicycle-style rollout 生成多宏动作 prefix，controls 为 `[accel, steer, jerk, steer_rate]`，并包含 route / collision / control feasibility 检查。
- **counterfactual futures**：实现 replay、reactive route-following IDM surrogate、targeted perturbation；hidden spawn 只从 `history.occ_mask[unknown]` 中采样，且 `hidden_start >= T_p + delay`。
- **BEV observation**：7 通道 mask：`visible_free, occupied_visible, unknown, occluder, route, drivable, confidence`，包含动态遮挡 shadow；occluded regime 基于 route corridor 内 unknown 比例，不泄漏 hidden metadata。
- **root 聚类与聚合**：按 recovery signature 聚类；root aggregation 默认使用 lower-tail LCVaR，而不是 mean。
- **teacher**：补齐 recovery options、controllers 和 active-mask corrected margins；option 参数会真实改变 rollout 和 margin。
- **OC-MERO**：修正为 `w_ij = normalize(C_ij * p_j)`，支持 invalid mask，不再假设 `R_orc >= R_dep` 恒成立。
- **papercheck**：新增 `python -m ocrap.cli papercheck` 和 `ocrap.data.build.papercheck`，检查 artifact fraction、hidden spawn 合法性、candidate coverage、observation aliasing 非退化等关键字段。
- **可复现性**：稳定随机种子使用 SHA1，不依赖 Python `hash()`。

## 安装

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

开发 / 测试依赖：

```bash
pip install -e '.[dev]'
```

基础依赖只有 `numpy`, `torch`, `PyYAML`。WOMD TFRecord reader 本身不依赖 TensorFlow；如需从真实 WOMD `Scenario` bytes 解析 proto，需要本地环境提供 `waymo_open_dataset.protos.scenario_pb2` 或等价生成的 proto module。

## 仓库结构

```text
configs/
  default.yaml                    # 默认配置
src/ocrap/
  cli/                            # build-dataset / papercheck / train / evaluate 等入口
  config/                         # YAML 与 dotted override
  data/
    schema.py                     # RawScenario / SceneHistory / DatasetSample 等 schema
    womd/                         # 纯 Python TFRecord reader + WOMD parser
    build/                        # synthetic/WOMD dataset builder, diagnose, papercheck
  planning/                       # route lattice, prefix generation, CRISP selector
  simulation/
    observation/                  # BEV, visibility, renderer, compatibility
    futures/                      # replay / reactive / targeted future mining
    teacher/                      # recovery options, controllers, margins, rollout
  roots/                          # signature clustering, LCVaR aggregation
  algorithms/                     # LCVaR, OC-MERO, calibration
  models/                         # encoder, OC-RAP torch model, losses
  evaluation/                     # metrics, baselines, evaluator
  utils/                          # geometry, deterministic seed
  *.py                            # backward-compatible import shims
tests/                            # unit tests and smoke tests
```

## 快速验收命令

在仓库根目录运行：

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
```

构造论文 artifact fixture，并检查 papercheck：

```bash
PYTHONPATH=src python -m ocrap.cli build-dataset \
  --set data_source=synthetic_artifact \
  --set num_synthetic_scenarios=4 \
  --set num_candidate_prefixes=8 \
  --output runs/artifact_fixture

```

期望 `papercheck.json` 中：

```json
{
  "failures": []
}
```

并且 artifact / negative deployability / oracle recoverability 为正、hidden spawn 数量一致、candidate count 至少为 2、`Y_obs` off-diagonal 非退化。

## Synthetic artifact 数据构造逻辑

`data_source=synthetic_artifact` 会构造专门验证论文 idea 的场景：

1. SDC 沿 route 前进，前方存在遮挡物或 occluder。
2. BEV unknown channel 在 route corridor 内形成动态遮挡 shadow。
3. targeted future 只允许从 unknown 且 drivable 的 cell 中生成 hidden agent。
4. hidden agent 的出现时间满足 `hidden_start >= T_p + hidden_emergence_delay_steps`。
5. 同一 post-prefix observation 下构造至少一对 recovery 签名不同的 futures，使 oracle recoverable 和 deployable recoverable 之间出现可检测 gap。
6. papercheck 会验证 artifact mining 不是靠 metadata 泄漏，而是通过 observation compatibility 与 recovery signature 产生。

## Natural WOMD + Waymax数据集构造(作为论文benchmark)

真实 WOMD 构造使用 `data_source=womd`：

```bash
python -m ocrap.cli build-dataset \
 --set data_source=womd \
 --set simulation_backend=waymax_closed_loop \   
 --set womd_patterns='/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/scenario/training/*.tfrecord*' \   
 --set max_scenarios=1000 \   
 --set max_agents=64 \   
 --set max_map_polylines=256 \   
 --set max_polyline_points=64 \   
 --set max_times_per_scenario=4 \   
 --set max_biased_times_per_scenario=4 \   
 --set num_candidate_prefixes=24 \   
 --set num_reactive_futures=4 \   
 --set num_targeted_futures=8 \   
 --set waymax.prefix_dynamics=invertible_bicycle \   
 --set waymax.allow_new_objects_after_warmup=true \   
 --set waymax.enable_augmented_hidden_roots=false \   
 --set artifact.use_margin_override=false \   
 --set progress=true \   
 --output data/ocrap_womd 

```

每个waymax future写入
```python
{
  "runtime_backend": "waymax_closed_loop",
  "waymax_runtime": true,
  "waymax_version": "...",
  "dynamics_model": "InvertibleBicycleModel",
  "sim_agent_policy": "log_playback",
  "scenario_augmented": false,
  "allow_object_injection": false,
  "controlled_object_ids": ["sdc"],
  "action_dim": 2,
  "rollout_start_timestep": 37,
  "prefix_steps": 10,
  "recovery_steps": 40,
  "rng_seed": 123,
  "waymax_metrics": {}
}
```

## Waymax augmented stress数据集构造

```bash
python -m ocrap.cli build-dataset \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns='/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/scenario/training/*.tfrecord*' \
  --set max_scenarios=1000 \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set max_times_per_scenario=4 \
  --set max_biased_times_per_scenario=4 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=4 \
  --set num_targeted_futures=8 \
  --set waymax.prefix_dynamics=invertible_bicycle \
  --set waymax.allow_new_objects_after_warmup=true \
  --set waymax.enable_augmented_hidden_roots=true \
  --set artifact.force_mine=true \
  --set artifact.use_margin_override=true \
  --set progress=true \
  --output data/ocrap_womd_waymax_stress_1k
```

augmented hidden stress future额外写入
```python
{
  "scenario_augmented": true,
  "hidden_emergence": true,
  "hidden_from_unknown": true,
  "hidden_visible_free_spawn": false,
  "artifact_branch": "yield|accelerate"
}
```
输出目录：

```text
output_dir/
  manifest.csv
  dataset_summary.json
  samples/*.npz
```

每个 `.npz` 对应一个 `(scene_id, time_index, candidate_index)` 样本，核心字段包括：

```text
agent_history, agent_valid, map_polylines, map_valid, dynamic_map, route,
bev_occ, prefix_states, prefix_controls, prefix_macro_id, prefix_param,
future_probs, root_assignments, root_probs, root_signature,
root_future_signature, root_valid, future_to_root_weight,
y_obs, c_star, m_star, option_valid,
r_orc_star, r_dep_star, oracle_gap_star, i_art_star,
regime_label, sample_metadata, split_id
```


## Diagnose 数据集

`diagnose` 用于检查生成数据集是否能支撑论文实验，而不只是检查文件能否读取。它会输出 schema/shape、split 泄漏、candidate coverage、future source coverage、hidden spawn 合法性、observation equivalence 退化、aliasing/incompatible recovery pair、OC-MERO label 可复算性、FRA/ODG/DRS 相关标签覆盖、regime 覆盖和 calibration/test split 可用性。

```bash
PYTHONPATH=src python -m ocrap.cli diagnose \
  --dataset data/ocrap_womd \
  --output data/ocrap_womd/diagnose.json
```[dff1.patch](../dff1.patch)

对于 smoke test fixture：

```bash
PYTHONPATH=src python -m ocrap.cli diagnose \
  --dataset runs/artifact_fixture \
  --output runs/artifact_fixture/diagnose.json
```

重点查看 `paper_support`、`failures`、`warnings`、`roots_and_observation.incompatible_alias_pair_fraction`、`recovery_labels.artifact_fraction`、`future_generation.hidden_from_unknown_count` 和 `splits`。

## 训练 / 校准 / 评估 smoke test

```bash
PYTHONPATH=src python -m ocrap.cli train \
  --dataset runs/artifact_fixture \
  --output runs/train_smoke

PYTHONPATH=src python -m ocrap.cli calibrate \
  --dataset runs/artifact_fixture \
  --checkpoint runs/train_smoke/best.pt \
  --output runs/train_smoke/calibration.json

PYTHONPATH=src python -m ocrap.cli evaluate \
  --dataset runs/artifact_fixture \
  --checkpoint runs/train_smoke/best.pt \
  --split train \
  --output runs/train_smoke/eval_train.json
```

## Ablation switches

CLI 支持论文实验中的主要 ablation：

```bash
--without-observation-kernel
--without-lower-tail
--without-calibration
--without-anti-oracle
--full-future-roots
--no-occlusion-bev
```

也可通过 dotted override 修改任意配置：

```bash
PYTHONPATH=src python -m ocrap.cli build-dataset \
  --set ocmero.alpha=0.1 \
  --set root_margin_aggregation=mean \
  --output runs/ablation_mean
```

## 已验证内容

本仓库已通过以下本地检查：

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
PYTHONPATH=src python -m ocrap.cli build-dataset --set data_source=synthetic_artifact --set num_synthetic_scenarios=4 --set num_candidate_prefixes=8 --output runs/artifact_fixture
PYTHONPATH=src python -m ocrap.cli papercheck --dataset runs/artifact_fixture --output runs/artifact_fixture/papercheck.json
```

当前对话未提供真实 WOMD TFRecord，因此没有在真实 WOMD 文件上跑端到端构造；代码已包含 TFRecord reader/parser 单测和 synthetic artifact fixture 验证。
