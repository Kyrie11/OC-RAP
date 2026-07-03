# OC-RAP 当前修复包说明

## 1. 代码修复：balanced artifact pass 的预检加速

修改文件：

- `src/ocrap/data/build/builder.py`

问题定位：

- `dataset_quality.artifact_pair_mode=balanced` 时，第二个 mined/artifact pass 会通过 `_cfg_with_artifact_mining(..., enable=True)` 把 `dataset_quality.require_artifact_pairs` 打开。
- 但原来的 `materialize()` 只在 `mode == "filter"` 时调用 `_artifact_pair_attempt_is_possible()`。
- 因此 balanced near/contact 构建仍可能对无法构造 hidden yield/accelerate artifact pair 的 prefix 执行完整 Waymax future generation 与 teacher rollout，然后再丢弃，造成无效耗时。

修复逻辑：

- 改成检查 `local_cfg` 中的 `require_artifact_pairs`。
- non-artifact pass 不受影响，因为 `_cfg_with_artifact_mining(..., enable=False)` 会把该 flag 置为 False。
- artifact/mined pass 会先做 cheap preflight，不可构造 artifact pair 时直接跳过该 prefix。

## 2. test_safe 空目录的建议命令

`validation_interactive` 更适合 near/contact 的 artifact-recovery test，不适合作为 clean safe/normal test 的唯一来源。建议 test_safe 使用普通 validation shard 构造，并沿用 val_safe 的未来分支设置，再强制 split 为 test。

```bash
python -m ocrap.cli build-dataset \
  --set data_source=womd \
  --set simulation_backend=waymax_closed_loop \
  --set womd_patterns="${WOMD_VAL}@150" \
  --set max_scenarios=400 \
  --set split.force_id=test \
  --set max_times_per_scenario=3 \
  --set max_biased_times_per_scenario=1 \
  --set dataset_quality.min_uniform_times_per_scenario=2 \
  --set num_candidate_prefixes=24 \
  --set num_reactive_futures=2 \
  --set num_targeted_futures=4 \
  --set num_roots=8 \
  --set num_recovery_options=12 \
  --set waymax.dataloader_include_sdc_paths=false \
  --set 'waymax.metrics_to_run=[log_divergence,overlap,offroad,kinematic_infeasibility]' \
  --set waymax.compute_future_metrics=false \
  --set waymax.teacher_backend=hybrid \
  --set waymax.teacher_rollout_top_k_options=4 \
  --set waymax.enable_augmented_hidden_roots=false \
  --set waymax.enable_visible_perturbation_roots=true \
  --set artifact.force_mine=false \
  --set artifact.mine_probability=0.0 \
  --set artifact.use_margin_override=false \
  --set dataset_quality.balanced_two_pass=false \
  --set dataset_quality.artifact_pair_mode=tag \
  --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
  --set dataset_quality.require_nominal_per_scene_time=true \
  --set dataset_quality.keep_nominal_even_if_quality_fails=true \
  --set dataset_quality.min_accepted_prefixes_per_scene_time=2 \
  --set 'dataset_quality.require_nominal_regimes=[normal]' \
  --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact]' \
  --set 'dataset_quality.forbid_any_regimes=[oracle_artifact]' \
  --set regime_thresholds.tau_occ=0.75 \
  --set regime_thresholds.tau_normal_occ=0.90 \
  --set regime_thresholds.include_prefix_collision_in_near=false \
  --set regime_thresholds.include_prefix_contact_in_post=false \
  --set regime_thresholds.use_paper_regime_definitions=true \
  --set io.compress_npz=false \
  --set io.fsync_npz=false \
  --output "$OCRAP_ROOT/test_safe"
```

如果仍然样本少，先不要放宽到 `validation_interactive`，优先增大 `max_scenarios` 或改用普通 `validation` 的更多 shard。

## 3. near/contact 快速构建可选项

在保持当前 proof-artifact 逻辑的前提下，可尝试增加：

```bash
--set dataset_quality.artifact_pass_structural_teacher=true
```

它会让 artifact/mined pass 使用 structural teacher，速度更快；但这会让 mined pass 不再是完整 hybrid Waymax teacher，最终论文主实验需要如实说明，或只用于 debug/proof 数据。
