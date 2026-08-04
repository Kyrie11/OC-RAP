# OC-RAP v50.4：外部 baseline 对比、critical scene 选择与视频生成修复报告

## 1. 结论

当前“CSV 能生成、video 完全没有生成”不是 Matplotlib、ffmpeg 或 Waymax 渲染故障。流程在 selective trace rerun 之前的 target-key preflight 就退出了，因此没有任何 rollout trace，也不可能进入视频编码。

故障由三个相互叠加的问题造成：

1. **历史 full-run journal 的 target key 与当前重建 dataset 的 scene-id 命名空间不同。**
   selection 中是 `test_near_contact:waymax_<old_hash>:t37`，当前 bucket 样本则是另一套 `waymax_<new_hash>__wx########`。直接做字符串相等匹配必然失败。
2. **preflight 与 closed-loop runner 的 bucket label 契约不一致。**
   runner 使用目录名 `test_near_contact` / `test_contact`；旧 preflight 却归一化成 `near_contact` / `contact`。即使 scene id 一致，合法 key 仍会被 preflight 错误拒绝。
3. **`--num-failure 0` 的选择器仍追加一条 failure。**
   旧循环先 `append` 再判断数量，因此 Near/Contact 都选择了 5 positive + 1 failure；下游却要求严格 5 + 5 个正向视频。

本补丁同时修复 target key 迁移、bucket 契约、选择器、可视化语义，并加入回归测试。

---

## 2. 对论文与当前实现的理解

### 2.1 论文核心 idea

论文把 recoverability 从“发生危险后才调用的 emergency behavior”提升为规划阶段的 action-admission primitive。关键不是每个 latent future 是否各自存在某个 recovery，而是：执行 candidate prefix 后，在 ego 无法区分的 observation-equivalent latent roots 中，是否存在一个可由同一 post-prefix observation 选择的共享 recovery option。

核心链路是：

1. **Recovery-sufficient latent roots**：只建模影响 recovery affordance 的未来因素，不追求完整未来生成。
2. **Post-prefix observation equivalence**：根据 candidate 执行后的可观测结果，把不可区分 roots 分组。
3. **Affordance-conditioned signed recovery margins**：对 semantic recovery mode 与连续参数计算带约束 slack 的 margin。
4. **OC-MERO**：先在 observation-equivalent roots 内对同一个 recovery option 做 lower-tail aggregation，再对 option 取最大；同时计算 branch-wise oracle recovery 与 oracle-to-deployable gap。
5. **CRISP**：把 deployable recovery 作为 admission constraint，在 nominal candidate 可接受时保持 nominal，否则选择满足 recovery/hard-rule/harm constraint 的最高 utility candidate。

这使算法天然适合 Near-contact 和 Contact：前者关注在危险尚未接触时保留制动、让行、横向逃逸或 route-rejoin 余量；后者关注初始接触后是否能脱离 overlap、避免二次接触、稳定运动或安全停车。

### 2.2 当前代码相对论文的演化

当前实现保留了论文主干：scene/prefix encoding、root queries、root probability、observation embedding、recovery option margin、observation-consistent aggregation和 calibrated admission。但工程实现已明显扩展：

- candidate lattice 包含 `nominal / keep / brake / yield / lane_shift / merge / pull_over / stabilize / perturb_nominal` 等 macro，并进行 route-local rollout、动态/道路可行性筛选；
- 模型加入 direct recovery value/frontier、set tournament、relative features、experts、evidence calibrator、continuous safety-capped frontier 等多轮实验演化模块；
- closed-loop runner 已计算 clearance、TTC、collision/overlap、offroad、critical exposure、post-contact escape、overlap duration、recontact、stable-stop quality、speed/acceleration/jerk/yaw-rate、route progression、intervention/NUP 等指标。

因此，论文和代码不是逐行对应，但算法的 observation-consistent recoverability 主线仍然一致。

---

## 3. 数据集性质与它对视频选择的含义

上传的 test reports 显示：

| Regime | samples | scenes | scene-time groups | candidates/group mean | roots | recovery options | oracle artifact | negative deployable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe | 3216 | 175 | 402 | 8.00 | 8 | 12 | 0.00% | 6.90% |
| Near-contact | 4723 | 250 | 595 | 7.94 | 8 | 12 | 24.41% | 48.80% |
| Contact | 6687 | 209 | 747 | 8.95 | 8 | 12 | 21.80% | 44.40% |

这与论文定位一致：Near/Contact 中 oracle artifact 和 negative deployability 足够多，确实是展示 OC-RAP 作用的主要 regime；Safe 更适合用 population table 证明 nominal utility preservation，而不是做大量视频。

---

## 4. 现有外部 baseline 结果应如何解读

### 4.1 Safe（175 scenes）

OC-RAP 的 collision rate 为 1.71%，offroad 2.29%，bounded NUP 0.982，intervention 13.14%。它和 nominal replay 的 collision 相同，offroad 略低；与部分学习 baseline 比较没有“所有指标绝对最佳”。Safe 的主要论文叙事应是：**保持高 nominal utility、低干预，并未为了 recovery 过度改变正常驾驶**。

### 4.2 Near-contact（250 scenes）

被 aggregate selector 选中的外部 comparator 是 DRO-CVaR：

| Metric | OC-RAP | DRO-CVaR |
|---|---:|---:|
| collision rate ↓ | 7.60% | 3.60% |
| offroad rate ↓ | 2.00% | 2.80% |
| TTC p05 ↑ | 0.441 s | 0.516 s |
| terminal clearance ↑ | 4.253 m | 4.418 m |
| critical TTC exposure ↓ | 1.827 s | 1.772 s |
| bounded NUP ↑ | **0.986** | 0.852 |
| intervention rate | **10.34%** | 75.32% |

因此 Near-contact 不能声称 OC-RAP 在 population collision/TTC 上全面优于 DRO-CVaR。更准确的结论是：**OC-RAP 以显著更低的干预率保留更高 nominal utility，同时在一部分 observation-consistency 关键场景中避免不必要的保守动作或保留更好的 recovery headroom。** 视频必须和 population table 一起呈现，明确属于 post-hoc qualitative evidence。

### 4.3 Contact（209 scenes）

被选中的 comparator 是 post-collision restoration：

| Metric | OC-RAP | Restoration |
|---|---:|---:|
| terminal clearance ↑ | **4.368 m** | 4.323 m |
| free-space AUC ↑ | 4.248 m | **4.284 m** |
| escape rate ↑ | 77.03% | 77.03% |
| recontact rate ↓ | **8.13%** | 10.05% |
| overlap duration ↓ | **0.0746 s** | 0.0837 s |
| stable-stop-quality ↑ | 3.83% | **5.26%** |

Contact 的 strongest population story 是：**更高 terminal separation、更低 recontact、更短 overlap duration**；不是“稳定停车全面更好”。如果视频展示“碰撞后继续稳定驾驶”，必须额外确认 yaw-rate、jerk、offroad、recontact 和 route progression，而不能仅用 `new_stable_stop_quality_event` 代替 controlled continuation。

---

## 5. 视频为何没有生成：完整执行链分析

执行链为：

1. `recover_v50_1_full_pipeline_no_retrain.sh`
2. `build_external_comparison_artifacts.sh`
3. 生成三张表、选 best external baseline
4. 从 metric-only scene journals 选择 critical targets
5. `run_selected_recovery_video_traces.sh`
6. preflight target/WOMD 支持
7. 只针对选中 targets 重跑 OC-RAP 与 comparator，开启 `render_trace=true`
8. `render_critical_scenes_v48_34.py`
9. Matplotlib + ffmpeg 生成 MP4

日志停止在第 6 步。Near selection 请求 6 个 key，但扫描当前 Near bucket 的 4723 行、595 个 unique scene-time targets 后，匹配数为 0，`target_keys_valid=false`。由于 shell 使用 `set -euo pipefail`，第一个 preflight 返回 3 后立即退出：Contact preflight、两个方法的 selective rerun、renderer 和 ffmpeg 都没有执行。

### 5.1 旧/新 scene-id 命名空间

旧 selection 例如：

```text
test_near_contact:waymax_45a289bb2ca6f3b4:t37
```

selection item 同时保留了：

```text
scene_id = 366c5e430006a7b__wx00000669
```

当前重建 bucket 的 scene hash 可能不同，但 `__wx00000669` 仍编码 stable `source_scenario_index=669`。因此正确的迁移身份应是：

```text
(dataset bucket, source_scenario_index, target_time_index)
```

并且必须验证映射唯一，不能静默猜测。

### 5.2 preflight bucket label 不一致

旧 preflight 构造可用 key 时使用 `near_contact:<scene>:t...`；runner 构造 key 时使用 `test_near_contact:<scene>:t...`。这是一处独立的 false negative。补丁后 preflight 使用与 runner 相同的 `_dataset_label` 语义，同时单独输出 human-readable `regime_counts_after_split`。

### 5.3 0 failure 的 off-by-one

旧代码：

```python
failure.append(row)
if len(failure) >= max(args.num_failure, 0):
    break
```

当 `num_failure=0` 时第一次 append 后 `1 >= 0`，所以仍得到一条 failure。补丁改为只有 `num_failure > 0` 才构造 failure list；selective video resolver 也显式只接受 `positive_toy_example`，最多 5 条，以兼容已经产生的 v50.3 selection artifact。

---

## 6. 旧 critical-scene 选择逻辑的问题

### 6.1 Near：未截断 TTC 会被 horizon/sentinel 主导

旧 score 直接线性使用秒数：

```text
1.5 * ΔTTC_p05 + 0.5 * Δterminal_TTC + ...
```

已选择的 Near 第一名：

- `ΔTTC_p05 ≈ +10.68 s`
- `Δterminal_TTC ≈ +94.04 s`
- `Δterminal_clearance ≈ -3.19 m`
- old score ≈ 61.37

它因为超大 terminal TTC 排第一，却有明显 terminal clearance 回退；视频很可能看起来并不支持“保持更大安全距离”。这通常来自 finite-horizon/sentinel TTC，而非 94 秒真实可比较的 maneuver benefit。

另一条 selected scene 的 `Δterminal_TTC ≈ +64.89 s`，同样说明 raw TTC 对排序影响过大。

### 6.2 Contact：TTC recovery 同样可能反向误导

旧 Contact score 对 `Δttc_recovery_gain_s` 使用 0.35 的无界线性权重。原 top scene 含约 `+19.48 s` TTC gain；而被误加进来的“failure” scene 实际有约：

- `Δterminal clearance ≈ +1.91 m`
- `Δfree-space AUC ≈ +0.61 m`
- `Δclearance gain ≈ +1.92 m`
- 但 `ΔTTC recovery ≈ -72.46 s`

所以 old score 为 -22.42，被当成 failure。对 post-contact recovery 来说，这种分类显然不能可靠代表视觉质量。

### 6.3 仅检查“至少一个改善”，缺少关键维度回退 guard

旧 Near guard 没有约束 terminal clearance、terminal TTC、near-zero exposure；旧 Contact 没有检查 yaw-rate、jerk、route progress。因此一个场景可以靠某一项巨大改善入选，同时另一项视觉上更重要的指标明显恶化。

### 6.4 没有 mechanism diversity

旧逻辑只限制每个 scene 最多一个 target，但 top-5 可能全部是相同类型的“terminal clearance 增加”。这不利于论文 qualitative figure/video 的覆盖面。

---

## 7. 新选择逻辑

### 7.1 bounded dimensionless score

所有连续 delta 先除以有物理含义的 scale，再 clip 到有限区间：

```python
clip(delta / scale, -limit, limit)
```

这样 +94 s 和 +10 s TTC 都不会无限抬高分数。Near 更重视 p05 clearance、terminal clearance、critical exposure；Contact 更重视 terminal separation、free-space AUC、overlap duration、escape/recontact，并将 TTC 降为弱辅助项。

### 7.2 explicit non-regression guards

Near 新增：

- collision/overlap、offroad 不得恶化；
- TTC p05、terminal TTC 不得超过阈值回退；
- clearance p05、terminal clearance 不得超过阈值回退；
- critical TTC exposure、near-zero clearance exposure 不得恶化。

Contact 新增：

- offroad、recontact 不得恶化；
- terminal clearance、free-space AUC、overlap duration 不得实质恶化；
- journal 存在时，yaw-rate p95、jerk p95、route progression 也纳入 guard。

### 7.3 evidence-profile diversity

Near profiles：

- `ttc_margin`
- `geometric_clearance`
- `reduced_exposure`

Contact profiles：

- `separation_recovery`
- `overlap_escape`
- `recontact_avoidance`
- `stable_stop`
- `controlled_continuation`

选择先尝试每个 profile 取一个最高分，再按总分补足 5 个。这样 Near 至少覆盖“距离、TTC、暴露时长”中的多类现象；Contact 尽量同时覆盖 separation 和 secondary-contact avoidance，而不是五条相似视频。

### 7.4 仍需声明 post-hoc qualitative

新 selection JSON 继续写入：

```json
"exploratory_qualitative_only": true,
"paper_population_claim_allowed": false,
"not_population_level_evidence": true
```

视频不能替代全量 paired table，也不能用于声称总体统计显著性。

---

## 8. 可视化逻辑修正

### 8.1 固定共享 world-frame camera

旧代码每一帧用两个 ego 的平均位置作为 moving center。它会：

- 掩盖绝对位移和 route progress；
- 让策略之间的 divergence 看起来变小；
- 产生相机运动造成的“稳定/不稳定”视觉错觉。

新 renderer 默认计算两个完整 trace 的共同 fixed bounding view。两侧 panel 在所有帧使用同一 center/radius，可直接比较轨迹偏离、停车、逃逸和进展。仍保留 `--camera-mode dynamic` 作为调试选项。

### 8.2 Contact anchor 语义

如果 trace 中没有观测到 overlap，但 target 属于 Contact，旧代码把第一帧标为 `causal contact anchor`。这会把 rollout start 错当成真实碰撞坐标。新标签是：

```text
post-contact rollout start
```

只有 trace 实际观测到 overlap 时才标 `observed overlap`。

### 8.3 去掉误导性的 clearance circle

`min_clearance_m` 是 oriented boxes 之间的边缘距离，不是以 ego 为圆心的 free-space radius。旧 clearance circle 的几何语义错误。新 renderer 改为：

- 找到当前最近的其他 agent center；
- 画一条辅助连接线；
- 文本标注 `reported box clearance`；
- 不宣称圆周上的任一点具有同样 clearance。

### 8.4 增加可解释动态信息

每帧 panel 现在显示：

- TTC
- box clearance
- ego speed
- yaw rate
- overlap/offroad
- selected macro/candidate/reason

底部加入同步 clearance timeline，Near 显示 2 m near-contact threshold，Contact/ Near 都标出 overlap 区间。这样能够更可靠地支持：

- Near：OC-RAP 是否维持更大 minimum/terminal clearance、减少 critical exposure；
- Contact：是否更快脱离 overlap、避免 recontact、保持平稳速度/航向变化。

---

## 9. 修改文件

| 文件 | 修改 |
|---|---|
| `tools/resolve_selected_targets_v50.py` | 新增：旧 selection key 到当前 bucket key 的唯一迁移；支持 exact / alias+time / source-index+time；输出审计报告 |
| `tools/check_closed_loop_dataset_support.py` | target-key bucket label 与 runner 对齐；新增 regime count |
| `tools/select_critical_scenes_v48_34.py` | 修复 zero-failure；bounded score；新 regression guards；evidence-profile diversity；保存 score components/reasons |
| `scripts/run_selected_recovery_video_traces.sh` | selective rerun 前先解析 key；只取 5 个 positive；再 preflight |
| `tools/render_critical_scenes_v48_34.py` | fixed camera、正确 contact label、去 clearance circle、动态指标、timeline |
| `scripts/build_top10_recovery_videos.sh` | 支持 `CAMERA_MODE=fixed` |
| `tests/test_v50_full_regime_runtime.py` | 添加 bucket contract、legacy key migration、zero-failure、contact label 回归测试 |

---

## 10. 验证结果

已完成：

1. `pytest` 针对 full-regime runtime 和 selector 相关测试：**17 passed**。
2. 用合成成对 closed-loop trace 运行新 renderer，并通过本机 ffmpeg 生成有效 MP4。
3. shell syntax 与 Python compile 检查通过。

由于当前分析环境没有你服务器上的 `/data0/...` WOMD、OC-RAP buckets、训练 checkpoint 和 GPU，无法在这里实际重跑 10 个真实 selected targets。补丁通过 unit/contract test 验证了导致 0-match 的两层 key 问题和视频编码链路；服务器运行后应重点检查 `target_resolution.json` 和 `preflight.json`。

---

## 11. 推荐运行方式

覆盖补丁后，不需要重跑 full metric：

```bash
cd /home/senzeyu2/code/OC-RAP
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

export OCRAP_ROOT="/data0/senzeyu2/dataset/OCRAP"
export CUDA_DEVICES="0,1"
export EXTERNAL_RESULTS_ROOT="runs/all_regime_external_baselines_v50_1_full"
export OCRAP_RESULTS_ROOT="runs/ocrap_three_regime_closed_loop_v50_1_full"
export COMPARISON_OUT="runs/external_comparison_v50_4_video_fix"
export EXTERNAL_CHECKPOINT_ROOT="/data0/senzeyu2/checkpoints/ocrap_external_baselines_v49"
export OCRAP_MODEL_RUN="runs/ocrap_v48_34_barrier_crossfit_dedicated_4834"
export MODEL_VARIANT="balanced"
export ALLOW_DIAGNOSTIC_RC20=1

export RECOVER_EXTERNAL=false
export RECOVER_OCRAP=false
export BUILD_COMPARISON=true
export BUILD_VIDEOS=true
export FPS=10

bash scripts/recover_v50_1_full_pipeline_no_retrain.sh
```

预期关键日志顺序：

```text
v50_critical_scene_selection ... selected=5
v50_selected_target_resolution ... num_resolved=5
v50_closed_loop_dataset_support ... num_matching_requested_target_keys=5 ... target_keys_valid=true
... selective OC-RAP/external trace reruns ...
critical_scene_recovery_videos_v50 ... num_videos=5
critical_scene_recovery_videos_v50 ... num_videos=5
top10_recovery_videos_v50 ... num_videos=10
```

重点输出：

```text
$COMPARISON_OUT/selective_traces/near/target_resolution.json
$COMPARISON_OUT/selective_traces/contact/target_resolution.json
$COMPARISON_OUT/selective_traces/near/preflight.json
$COMPARISON_OUT/selective_traces/contact/preflight.json
$COMPARISON_OUT/selective_traces/videos/TOP10_VIDEO_INDEX.json
$COMPARISON_OUT/selective_traces/videos/near/*.mp4
$COMPARISON_OUT/selective_traces/videos/contact/*.mp4
```

---

## 12. 论文呈现建议

推荐把 qualitative videos 分成以下叙事：

### Near-contact

1. **Geometric clearance preservation**：同一交通互动中，OC-RAP 的 p05/terminal clearance 更高，且不引入 overlap/offroad。
2. **Reduced critical exposure**：最低距离可能接近，但 OC-RAP 更快退出 TTC < 3 s 或 clearance < 2 m 的危险区间。
3. **Observation-consistent intervention**：baseline 因 robust risk 频繁强制干预，而 OC-RAP 在可恢复时保留 nominal；在 oracle artifact 场景才介入。

### Contact

1. **Separation recovery**：接触后更快拉开 box clearance，terminal clearance 更高。
2. **Secondary-contact avoidance**：baseline 出现 recontact/secondary overlap，OC-RAP 避免。
3. **Controlled continuation**：没有 offroad/recontact，yaw-rate 与 jerk 不恶化，并保持合理 route progression。
4. **Stable stop**：仅在该 scene 的 stable-stop-quality 确实改善时使用，不要把所有 post-contact recovery 都描述为停车稳定。

每个视频标题应显示 selection profile、关键 paired deltas、post-hoc qualitative disclaimer；正文同时引用完整 population table，避免 cherry-picking 质疑。
