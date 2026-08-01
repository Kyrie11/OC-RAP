# OC-RAP：Observation-Consistent Recovery-Affordance Planner

本仓库是对论文 `post-collision.tex` 中 OC-RAP 方法的可运行实现。实现以论文的 `abstract -> introduction -> method -> appendix -> experiments` 逻辑为主线，并按照 `代码完善指令.md` 对原始代码进行了重构、补全和修复：数据构造、candidate prefix 生成、counterfactual future mining、observation-consistent root 聚类、recovery teacher、OC-MERO、CRISP selector、训练、校准、评估和 papercheck 都已落到 `src/ocrap` 包结构中。

核心目标不是做普通 motion prediction，而是为每个 `(scene, time, candidate_prefix)` 判断：在 prefix 执行后、只基于 post-prefix observation 可区分的未来簇中，是否存在可恢复的 recovery option；以及是否存在 oracle recoverable 但 deployable 不可恢复的 observation aliasing artifact。

## 当前实验版本 v48.27 FACTOR-PHYSICS-BRIDGE

本版本针对 v48.26 的有效 `RC=20` 与空 adaptation-dev shadow 做了两类修复：

- **工程完整性**：完整扫描 sparse WOMD targets、canonical scenario ID、strict target match、空物理指标 fail-closed、development-rule 与 certificate rejection 分型；
- **算法合同**：raw-benefit 与 safe admission 分解、五个不可补偿 harm heads、factor→admission 两阶段训练，以及 regression/listwise/frontier 全部使用真实部署分数 `sigmoid(logit)-0.5`。

主入口：

```bash
bash scripts/run_v48_27_factor_physics_dedicated.sh
```

修复并重跑旧 v48.26 development shadow：

```bash
OUT=runs/ocrap_v48_26_execution_physics_dedicated_4826 \
DEV_SHADOW_WOMD_SOURCE="$WOMD_VAL_INTERACTIVE" \
DEV_SHADOW_RAW_MAX_SCENARIOS=0 \
bash scripts/repair_v48_26_dev_shadow_with_v48_27.sh
```

详细协议、双 A30 命令与结果判读见 `OC-RAP-v48.27-run-commands-ZH.txt`。

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
  --skip-existing \
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

# 所有指令集 
```bash
export WOMD_TRAIN=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord
export WOMD_VAL=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord
export WOMD_TEST=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing_interactive/testing_interactive_tfexample.tfrecord
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
```
## 构建Train Set(三个互补训练子集)

测速后建议默认采用 screened hybrid teacher：`teacher_rollout_top_k_options=2`、`teacher_metrics_stride=0`、`use_jit_scan_rollouts=true`。它保留全量 structural recovery option 标签，同时只对最有希望的 recovery option 做 Waymax 闭环验证，训练集速度和标签质量折中最好。下面的 sharded 命令默认后台并行执行；如果只有一张显存较小的 GPU，可以去掉行尾 `&` 分批运行。
### Natural/normal train set
这个子集用于补强自然 WOMD 场景，避免模型只学到 stress case。
```bash
for i in 0 1 2 3; do
python -m ocrap.cli build-dataset \
  --skip-existing \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns=${WOMD_TRAIN}@1000 \
  --set max_scenarios=1200 \
  --set scenario_stride=4 \
  --set scenario_worker_index=${i} \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set max_times_per_scenario=4 \
  --set max_biased_times_per_scenario=0 \
  --set dataset_quality.min_uniform_times_per_scenario=2 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=2 \
  --set num_targeted_futures=2 \
  --set num_recovery_options=12 \
  --set waymax.compute_future_metrics=false \
  --set waymax.detect_natural_hidden_emergence=false \
  --set waymax.enable_augmented_hidden_roots=false \
  --set waymax.enable_visible_perturbation_roots=false \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=2 \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  --set waymax.cache_env_objects=true \
  --set waymax.cache_postprefix_rollouts=true \
  --set waymax.cache_teacher_metric_rollouts=true \
  --set profiling.enabled=true \
  --set artifact.force_mine=false \
  --set artifact.mine_probability=0.0 \
  --set artifact.use_margin_override=false \
  --set dataset_quality.balanced_two_pass=false \
  --set dataset_quality.artifact_pass_use_margin_override=false \
  --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
  --set dataset_quality.min_artifact_prefixes_per_scene_time=0 \
  --set dataset_quality.max_artifact_prefixes_per_scene_time=1 \
  --set dataset_quality.min_nonartifact_prefixes_per_scene_time=6 \
  --set dataset_quality.max_nonartifact_prefixes_per_scene_time=8 \
  --set split_ratios.train=0.75 \
  --set split_ratios.val=0.10 \
  --set split_ratios.calibration=0.05 \
  --set split_ratios.test=0.10 \
  --set progress=true \
  --output ${OCRAP_ROOT}/train_natural_v1_w${i} &
done
wait
```

### Strict oracle-artifact train set
这个子集用于验证 “oracle recoverable ≠ deployable recoverable”。
关键是关闭 override，尽量让 teacher label 来自真实 rollout / hybrid rollout，而不是人为 margin override。
```bash
for i in 0 1 2 3; do
  nohup python -m ocrap.cli build-dataset \
    --skip-existing \
    --set data_source=womd \
    --set simulation_backend=waymax_closed_loop \
    --set womd_patterns=${WOMD_TRAIN}@1000 \
    --set max_scenarios=800 \
    --set scenario_stride=4 \
    --set scenario_worker_index=${i} \
    --set max_agents=64 \
    --set max_map_polylines=256 \
    --set max_polyline_points=64 \
    --set max_times_per_scenario=3 \
    --set max_biased_times_per_scenario=2 \
    --set dataset_quality.min_uniform_times_per_scenario=1 \
    --set num_candidate_prefixes=24 \
    --set num_reactive_futures=2 \
    --set num_targeted_futures=6 \
    --set num_recovery_options=12 \
    --set waymax.compute_future_metrics=true \
    --set waymax.detect_natural_hidden_emergence=true \
    --set waymax.enable_augmented_hidden_roots=true \
    --set waymax.enable_visible_perturbation_roots=true \
    --set waymax.teacher_backend=hybrid \
    --set waymax.teacher_rollout_top_k_options=2 \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    --set profiling.enabled=true \
    --set waymax.apply_artifact_override_to_screened_options=false \
    --set waymax.skip_waymax_rollout_for_augmented_override=false \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.20 \
    --set artifact.use_margin_override=false \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.balanced_rotate_prefix_order=true \
    --set dataset_quality.balanced_keep_nominal_nonartifact=true \
    --set dataset_quality.artifact_pass_use_margin_override=false \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=0 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6 \
    --set split_ratios.train=0.75 \
    --set split_ratios.val=0.10 \
    --set split_ratios.calibration=0.05 \
    --set split_ratios.test=0.10 \
    --set progress=true \
    --output ~/code/OC-RAP/dataset/train_strict_artifact_v1_w${i} \
    > build_strict_train_w${i}.log 2>&1 &
done
```

### Post-contact / near-contact stress train set
这个子集用于补强论文里 post-contact、secondary collision、near-contact 的叙事。

```bash
 
for i in 0 1 2 3; do
  nohup python -m ocrap.cli build-dataset \
    --skip-existing \
    --set data_source=womd \
    --set simulation_backend=waymax_closed_loop \
    --set womd_patterns=${WOMD_TRAIN}@1000 \
    --set scenario_stride=4 \
    --set scenario_worker_index=${i} \
    --set max_scenarios=800 \
    --set max_agents=64 \
    --set max_map_polylines=256 \
    --set max_polyline_points=64 \
    --set max_times_per_scenario=4 \
    --set max_biased_times_per_scenario=4 \
    --set dataset_quality.min_uniform_times_per_scenario=0 \
    --set num_candidate_prefixes=24 \
    --set num_reactive_futures=2 \
    --set num_targeted_futures=8 \
    --set 'targeted_future_kinds=[contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
    --set num_recovery_options=12 \
    --set waymax.compute_future_metrics=true \
    --set waymax.detect_natural_hidden_emergence=true \
    --set waymax.enable_augmented_hidden_roots=true \
    --set waymax.enable_visible_perturbation_roots=true \
    --set waymax.teacher_backend=hybrid \
    --set waymax.teacher_rollout_top_k_options=2 \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    --set 'waymax.teacher_rollout_option_modes=[post_contact_stabilize,avoid_secondary]' \
    --set waymax.apply_artifact_override_to_screened_options=false \
    --set waymax.skip_waymax_rollout_for_augmented_override=false \
    --set artifact.force_mine=false \
    --set artifact.mine_probability=0.0 \
    --set artifact.use_margin_override=false \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=10 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=0 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=8 \
    --set split_ratios.train=0.75 \
    --set split_ratios.val=0.10 \
    --set split_ratios.calibration=0.05 \
    --set split_ratios.test=0.10 \
    --set progress=true \
    --output ~/code/OC-RAP/dataset/train_post_contact_v1_w${i} \
    > build_stress_train_w${i}.log 2>&1 &
done
```
### Proof artifact train dataset
```bash
for i in 0 1 2 3; do
  python -m ocrap.cli build-dataset \
    --output ~/code/OC-RAP/dataset/train_proof_artifact_v3_w${i} \
    --skip-existing \
    --set data_source=womd \
    --set womd_patterns="${WOMD_TRAIN}@1000" \
    --set simulation_backend=waymax_closed_loop \
    --set max_scenarios=800 \
    --set scenario_stride=4 \
    --set scenario_worker_index=$i \
    --set max_times_per_scenario=2 \
    --set max_biased_times_per_scenario=2 \
    --set num_candidate_prefixes=16 \
    --set num_reactive_futures=1 \
    --set num_targeted_futures=4 \
    --set num_recovery_options=8 \
    --set num_roots=6 \
    --set waymax.enable_augmented_hidden_roots=true \
    --set waymax.detect_natural_hidden_emergence=false \
    --set waymax.enable_visible_perturbation_roots=false \
    --set waymax.teacher_backend=hybrid \
    --set waymax.teacher_rollout_top_k_options=1 \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    --set waymax.skip_waymax_rollout_for_augmented_override=true \
    --set waymax.compute_future_metrics=false \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.40 \
    --set artifact.use_margin_override=true \
    --set artifact.compatible_margin=1.2 \
    --set artifact.incompatible_margin=-6.0 \
    --set dataset_quality.artifact_pair_mode=balanced \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.artifact_pass_use_margin_override=true \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=6 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=3 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=6 \
    --set dataset_quality.max_nonartifact_attempts_per_scene_time=4 \
    --set split_ratios.train=0.55 \
    --set split_ratios.val=0.10 \
    --set split_ratios.calibration=0.25 \
    --set split_ratios.test=0.10
done
```


## 构建Val set
### Natural Validation Set
```bash
python -m ocrap.cli build-dataset \
  --skip-existing \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns=${WOMD_VAL}@150 \
  --set max_scenarios=400 \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set max_times_per_scenario=4 \
  --set max_biased_times_per_scenario=0 \
  --set dataset_quality.min_uniform_times_per_scenario=2 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=2 \
  --set num_targeted_futures=2 \
  --set num_recovery_options=12 \
  --set waymax.compute_future_metrics=false \
  --set waymax.detect_natural_hidden_emergence=false \
  --set waymax.enable_augmented_hidden_roots=false \
  --set waymax.enable_visible_perturbation_roots=false \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=2 \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  --set waymax.cache_env_objects=true \
  --set waymax.cache_postprefix_rollouts=true \
  --set waymax.cache_teacher_metric_rollouts=true \
  --set profiling.enabled=true \
  --set artifact.force_mine=false \
  --set artifact.mine_probability=0.0 \
  --set artifact.use_margin_override=false \
  --set dataset_quality.balanced_two_pass=false \
  --set split_ratios.train=0.0 \
  --set split_ratios.val=1.0 \
  --set split_ratios.calibration=0.0 \
  --set split_ratios.test=0.0 \
  --set progress=true \
  --output ${OCRAP_ROOT}/val_natural_v1
```

### Strict artifact validation set
```bash
nohup python -m ocrap.cli build-dataset \
  --skip-existing \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns=${WOMD_VAL}@150 \
  --set max_scenarios=300 \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set max_times_per_scenario=3 \
  --set max_biased_times_per_scenario=2 \
  --set dataset_quality.min_uniform_times_per_scenario=1 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=2 \
  --set num_targeted_futures=6 \
  --set num_recovery_options=12 \
  --set waymax.compute_future_metrics=true \
  --set waymax.detect_natural_hidden_emergence=true \
  --set waymax.enable_augmented_hidden_roots=true \
  --set waymax.enable_visible_perturbation_roots=true \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=2 \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  --set profiling.enabled=true \
  --set waymax.apply_artifact_override_to_screened_options=false \
  --set artifact.force_mine=true \
  --set artifact.mine_probability=0.15 \
  --set artifact.use_margin_override=false \
  --set dataset_quality.artifact_pass_use_margin_override=false \
  --set dataset_quality.artifact_quota_uses_label=true \
  --set split_ratios.train=0.0 \
  --set split_ratios.val=1.0 \
  --set split_ratios.calibration=0.0 \
  --set split_ratios.test=0.0 \
  --set progress=true \
  --output ~/code/OC-RAP/dataset/val_strict_artifact_v1 \
  > build_strict_val.log 2>&1 &
```

### Post-contact validation set
```bash
nohup python -m ocrap.cli build-dataset \
  --skip-existing \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns=${WOMD_VAL}@150 \
  --set max_scenarios=300 \
  --set max_agents=64 \
  --set max_map_polylines=256 \
  --set max_polyline_points=64 \
  --set max_times_per_scenario=4 \
  --set max_biased_times_per_scenario=4 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=2 \
  --set num_targeted_futures=8 \
  --set 'targeted_future_kinds=[contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
  --set num_recovery_options=12 \
  --set waymax.compute_future_metrics=true \
  --set waymax.detect_natural_hidden_emergence=true \
  --set waymax.enable_augmented_hidden_roots=true \
  --set waymax.enable_visible_perturbation_roots=true \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=2 \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  --set profiling.enabled=true \
  --set 'waymax.teacher_rollout_option_modes=[post_contact_stabilize,avoid_secondary]' \
  --set artifact.force_mine=false \
  --set artifact.mine_probability=0.0 \
  --set artifact.use_margin_override=false \
  --set split_ratios.train=0.0 \
  --set split_ratios.val=1.0 \
  --set split_ratios.calibration=0.0 \
  --set split_ratios.test=0.0 \
  --set progress=true \
  --output ~/code/OC-RAP/dataset/val_post_contact_v1 \
  > build_stress_val.log 2>&1 &
 
```

### proof artifact val dataset

```bash
python -m ocrap.cli build-dataset \
  --output ~/code/OC-RAP/dataset/val_proof_artifact_v3 \
  --skip-existing \
  --set data_source=womd \
  --set womd_patterns="${WOMD_VAL}@150" \
  --set simulation_backend=waymax_closed_loop \
  --set max_scenarios=150 \
  --set scenario_stride=4 \
  --set scenario_worker_index=$i \
  --set max_times_per_scenario=2 \
  --set max_biased_times_per_scenario=2 \
  --set num_candidate_prefixes=16 \
  --set num_reactive_futures=1 \
  --set num_targeted_futures=4 \
  --set num_recovery_options=8 \
  --set num_roots=6 \
  --set waymax.enable_augmented_hidden_roots=true \
  --set waymax.detect_natural_hidden_emergence=false \
  --set waymax.enable_visible_perturbation_roots=false \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=1 \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  --set waymax.skip_waymax_rollout_for_augmented_override=true \
  --set waymax.compute_future_metrics=false \
  --set artifact.force_mine=true \
  --set artifact.mine_probability=0.40 \
  --set artifact.use_margin_override=true \
  --set artifact.compatible_margin=1.2 \
  --set artifact.incompatible_margin=-6.0 \
  --set dataset_quality.artifact_pair_mode=balanced \
  --set dataset_quality.balanced_two_pass=true \
  --set dataset_quality.artifact_pass_use_margin_override=true \
  --set dataset_quality.artifact_quota_uses_label=true \
  --set dataset_quality.max_accepted_prefixes_per_scene_time=6 \
  --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
  --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
  --set dataset_quality.min_nonartifact_prefixes_per_scene_time=3 \
  --set dataset_quality.max_nonartifact_prefixes_per_scene_time=4 \
  --set dataset_quality.max_artifact_attempts_per_scene_time=6 \
  --set dataset_quality.max_nonartifact_attempts_per_scene_time=4 \
  --set split_ratios.train=0.55 \
  --set split_ratios.val=0.10 \
  --set split_ratios.calibration=0.25

```

## 数据检查与可视化

### 训练集拼接
```bash
export TRAIN_MIX="${OCRAP_ROOT}/train_natural_v1_w0,${OCRAP_ROOT}/train_natural_v1_w1,${OCRAP_ROOT}/train_natural_v1_w2,${OCRAP_ROOT}/train_natural_v1_w3,${OCRAP_ROOT}/train_strict_artifact_v1_w0,${OCRAP_ROOT}/train_strict_artifact_v1_w1,${OCRAP_ROOT}/train_strict_artifact_v1_w2,${OCRAP_ROOT}/train_strict_artifact_v1_w3,${OCRAP_ROOT}/train_post_contact_v1_w0,${OCRAP_ROOT}/train_post_contact_v1_w1,${OCRAP_ROOT}/train_post_contact_v1_w2,${OCRAP_ROOT}/train_post_contact_v1_w3"
```

### 验证集拼接
```bash
export VAL_MIX="${OCRAP_ROOT}/val_natural_v1,${OCRAP_ROOT}/val_strict_artifact_v1,${OCRAP_ROOT}/val_post_contact_v1"
```

### 检查训练集
```bash
python -m ocrap.cli diagnose \
  --dataset "$TRAIN_MIX" \
  --output ${OCRAP_ROOT}/reports/diagnose_train_mix_v1.json

python -m ocrap.cli papercheck \
  --dataset "$TRAIN_MIX" \
  --output ${OCRAP_ROOT}/reports/papercheck_train_mix_v1.json

python -m ocrap.cli analyze-dataset \
  --dataset "$TRAIN_MIX" \
  --output ${OCRAP_ROOT}/analysis/train_mix_v1
```

### 检查验证集
```bash
python -m ocrap.cli diagnose \
  --dataset "$VAL_MIX" \
  --output ${OCRAP_ROOT}/reports/diagnose_val_mix_v1.json

python -m ocrap.cli papercheck \
  --dataset "$VAL_MIX" \
  --output ${OCRAP_ROOT}/reports/papercheck_val_mix_v1.json

python -m ocrap.cli analyze-dataset \
  --dataset "$VAL_MIX" \
  --output ${OCRAP_ROOT}/analysis/val_mix_v1
```

## 模型训练
```bash
CUDA_VISIBLE_DEVICES=0 python -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1 \
  --set training.device=cuda:0 \
  --set training.require_cuda=true \
  --set training.progress=true \
  --set training.epochs=40 \
  --set training.batch_size=128 \
  --set training.lr=3e-4 \
  --set training.weight_decay=1e-4 \
  --set training.num_workers=8 \
  --set training.artifact_sampler_weight=1.0 \
  --set model.encoder_type=structured_transformer \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.d_model=256 \
  --set model.d_obs=64 \
  --set model.tau_obs=1.0
```

## Calibration

```bash
python -m ocrap.cli calibrate \
  --dataset "$TRAIN_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/calibration.json
```

## 内部test split评估
```bash
python -m ocrap.cli evaluate \
  --dataset "$TRAIN_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --calibration ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/calibration.json \
  --split test \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/eval_internal_test.json \
  --set 'evaluation.methods=[nominal,risk_aware,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]'
```

## 外部validation set评估

```bash
python -m ocrap.cli evaluate \
  --dataset "$VAL_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --calibration ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/calibration.json \
  --split val \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/eval_external_val.json \
  --set 'evaluation.methods=[nominal,risk_aware,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]'
```

## 消融实验

### 无 observation kernel

```bash
python -m ocrap.cli evaluate \
  --dataset "$VAL_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --calibration ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/calibration.json \
  --split val \
  --without-observation-kernel \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/ablation_without_obs_kernel.json \
  --set 'evaluation.methods=[ocrap]'
```
对应论文 OC-RAP w/o observation kernel 

### 无lower-tail aggregation 
```bash
python -m ocrap.cli evaluate \
  --dataset "$VAL_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --calibration ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/calibration.json \
  --split val \
  --without-lower-tail \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/ablation_without_lower_tail.json \
  --set 'evaluation.methods=[ocrap]'
```
### 无calibration
```bash
python -m ocrap.cli evaluate \
  --dataset "$VAL_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/best.pt \
  --split val \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_mix_v1/ablation_without_calibration.json \
  --set selection.gamma_rec=0.0 \
  --set 'evaluation.methods=[ocrap]'
```

### 无anti-oracle loss，需要重新训练
```bash
CUDA_VISIBLE_DEVICES=0 python -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1 \
  --set training.device=cuda:0 \
  --set training.require_cuda=true \
  --set training.progress=true \
  --set training.epochs=40 \
  --set training.batch_size=128 \
  --set training.lr=3e-4 \
  --set training.weight_decay=1e-4 \
  --set training.num_workers=8 \
  --set training.artifact_sampler_weight=1.0 \
  --set loss_weights.anti_oracle=0.0 \
  --set model.encoder_type=structured_transformer \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.d_model=128 \
  --set model.d_obs=64 \
  --set model.tau_obs=1.0
```

```bash
python -m ocrap.cli calibrate \
  --dataset "$TRAIN_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1/best.pt \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1/calibration.json

python -m ocrap.cli evaluate \
  --dataset "$VAL_MIX" \
  --checkpoint ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1/best.pt \
  --calibration ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1/calibration.json \
  --split val \
  --output ${OCRAP_ROOT}/runs/ocrap_structured_no_anti_oracle_v1/eval_external_val.json \
  --set 'evaluation.methods=[ocrap]'
```
