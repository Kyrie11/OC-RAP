# OC-RAP v48.56 strict-teacher shadow：8 小时长运行问题诊断与 runtime hotfix r1

## 1. 结论

这次 `bash scripts/run_v48_56_strict_teacher_shadow.sh` 长时间不结束的首要原因不是算法死循环，也不是 GPU 互锁，而是 **Waymax source partition 被放在了昂贵 WOMD/Waymax 前处理之后**，在 `scenario_start_index=11000` 的 strict-shadow 配置下产生了严重的 source-scan amplification。

原实现为了在 builder 层得到全局索引

```text
global_index = start + worker + n * stride
```

会先在 source config 中把 `scenario_start_index/scenario_stride/scenario_worker_index` 归零，然后把 source `max_scenarios` 扩张到目标 global index。Waymax loader 随后用 `get_data_generator()` 对这些记录先做 WOMD TFExample parse、`SimulatorState` construction，并进一步转换为 `RawScenario`；builder 最后才丢掉不属于当前 worker residue 的记录。

默认 strict-shadow 六 worker 的旧 expensive-source budget 是：

| shard | worker | 真正需要 | 旧 source expensive preprocess budget |
|---|---:|---:|---:|
| Safe | 0 | 600 | 14,595 |
| Safe | 1 | 600 | 14,596 |
| Near | 2 | 700 | 15,197 |
| Near | 3 | 700 | 15,198 |
| Contact | 4 | 700 | 15,199 |
| Contact | 5 | 700 | 15,200 |
| **总计** | | **4,000** | **89,985** |

因此旧代码对昂贵的 WOMD parse / Waymax state / RawScenario construction 的调用量约被放大 **22.50×**。注意这表示昂贵前处理次数的下降，不应机械理解为 wall-clock 必然缩短 22.50×；raw TFRecord 顺序流仍需走过 start 之前的 serialized records，而且 Near/Contact exact teacher 本身仍然很重。

## 2. 为什么看起来像“卡死”

原 `build_v48_56_strict_teacher_calibration_shadow.sh` 把两个 worker 的 stdout/stderr 全部重定向到文件，然后父 shell 直接 `wait`。在 worker 长时间做 source conversion 或 exact teacher 时，父终端没有进度信息，因此表现为几个小时“没有反应”。

代码中没有找到会让本链条无限迭代的算法 loop：Waymax DatasetConfig 使用 `repeat=1`；builder 的主循环在 source iterator Exhaust 后结束；父脚本的 `wait` 是在等待真实 worker。真正的问题是 **有限但被异常放大的工作量 + 不透明的运行状态**。

## 3. 第二个真实成本：Near/Contact exact teacher

Near 与 Contact strict shadow 明确使用：

```text
waymax.teacher_rollout_top_k_options=0
```

这是 all-valid-option exact Waymax teacher。它不能为了提速直接改成 `top_k=4`，否则 strict shadow 的 teacher semantics 会变化，correctness audit 不再是同一个实验。

因此 runtime hotfix 保留 `top_k_options=0`。优化 source amplification 后，如果 heartbeat 中 `teacher_margins` 成为主要 stage，这是合理的剩余计算成本，而不是继续把它误判成 source 卡死。

另外发现一个语义等价的无效计算：历史代码即便 `option_valid=false`，仍先执行完整 4 s Waymax recovery rollout，最后才把 margin 强制成 `-1e9`。hotfix 现在直接跳过这种 invalid-option exact rollout；最终 teacher margin 不变。

## 4. 已落地修复

### 4.1 Raw TFRecord prefilter

新增：

```text
waymax.prefilter_source_scan_controls=true
waymax.require_source_scan_prefilter=true   # strict-shadow only
```

Waymax fast path现在按如下顺序：

```text
serialized TFRecord stream
  -> skip(scenario_start_index)
  -> shard(scenario_stride, scenario_worker_index)
  -> take(max_scenarios)
  -> WOMD parse
  -> Waymax SimulatorState
  -> RawScenario
```

所以 expensive parse/state conversion 只发生在当前 worker 真正需要的记录上。global index、scene id、worker residue、scene-disjoint partition 不变。

strict shadow 设为 `require_source_scan_prefilter=true`：若安装的 Waymax/TensorFlow 版本不兼容，直接 fail-closed，不允许静默退回旧慢路径。

### 4.2 Resume compatibility

`prefilter_source_scan_controls` 与 `require_source_scan_prefilter` 只改变 transport/runtime，不改变 sample generation semantics，已经从 semantic resume fingerprint 中排除。因此用旧代码运行 8 小时后留下的 partial shard，只要其原始 teacher/config 相同，可以在新代码中继续 `RESUME=1`，无需仅因为本 hotfix 删除有效 sample。

### 4.3 Heartbeat / live profile

每个 worker 开启 runtime profiling，controller 周期输出：

- `dataset_status.json`
- `build_stage_profile.json`
- `raw_scenarios_seen`
- `scene_time_groups`
- `new_samples_written`
- `samples_per_hour`
- `stage_totals_s`
- worker log tail / log age

因此下一次可以直接区分：source scan、planning-time selection、future generation、teacher margins、NPZ I/O 中谁是真正的 dominant runtime。

### 4.4 Completed-shard reuse

`RESUME=1` 时，如果某一 safe/near/contact worker shard 已经同时存在非空 `manifest.csv` 和 `dataset_summary.json`，该 shard不重新运行。partial shard继续走 dataset builder 的原有 atomic/resume contract。

### 4.5 删除 semantic audit 的第三次全量 NPZ read

`build_teacher_pcd_index_v48.py` 本来已经加载每个 NPZ 并用 `m_star/root_probs/c_star` fresh recompute OC-MERO。旧 strict runner 构建 train index、dev index后，semantic audit 又把 train+dev NPZ 全量打开一次，仅为了重复做同一 source consistency check。

现在 index 直接保存：

- fresh `R_dep/R_orc/GAP`
- fresh-vs-cached absolute error
- cached `r_dep_star/r_orc_star` 是否真实存在

最终 semantic audit复用 index 中的 fresh fields。若 cached label 缺失仍然 fail-closed，因此不是通过放松 correctness 换速度。

### 4.6 Runtime scan contract

build 一开始输出并写入：

```text
V48_56_STRICT_SHADOW_RUNTIME_SCAN_CONTRACT.json
```

默认应记录：

```text
legacy_preprocessed_record_budget = 89985
optimized_preprocessed_record_budget = 4000
avoided_expensive_preprocess_records = 85985
legacy_to_optimized_preprocess_ratio = 22.49625
```

## 5. 下一次运行时如何判断是否正常

首先停止旧 controller 及其仍存活 worker，避免新旧进程同时写同一个 `SHADOW_ROOT`。然后用同一个 shadow root、`RESUME=1` 运行新代码。

启动后应先看到：

```text
event=v48_56_waymax_raw_prefilter_preflight
status=pass
```

以及 runtime scan contract。worker log 中应看到：

```text
event=waymax_raw_source_prefilter
```

controller 默认每 60 秒打印一次 `v48_56_worker_heartbeat`。

如果 `raw_scenarios_seen` 持续增加而 `stage_totals_s.teacher_margins` 占主导，说明程序没有挂死，而是正在执行正式 all-option exact teacher。若 source fast path不可用，新版会直接报错退出，不再允许它无提示地回退到旧的数小时扫描。

## 6. 验证范围

本交付环境没有用户训练机上的 TensorFlow、Waymax、WOMD `/data0/...` 和对应 GPU 环境，因此无法在这里声称完成真实 full-shadow wall-clock benchmark。已完成的是：

- raw/global-index partition arithmetic regression；
- optimized fast path / legacy fallback / required-fast fail-closed / mid-stream duplicate prevention；
- builder source-delegation contract；
- old-partial/new-fast resume fingerprint equivalence；
- invalid-option exact rollout elimination；
- inline fresh OC-MERO + cached-label-presence correctness；
- v48.47–v48.56 relevant changed-path regression；
- Python compileall；
- 全部当前 `scripts/*.sh` 的 `bash -n`。

完整历史 pytest 的第一个失败仍来自上传代码包缺失早期历史文件 `scripts/train_ocrap_v48_12_trident.sh`，属于 package history debt，不是本次 runtime hotfix 引入；没有通过补造历史脚本去伪造全套通过。
