# OC-RAP v50.3 闭环恢复与无重训续跑修复

## 1. 输入材料与结论边界

本次审计使用了：

- V50.2 源码包 `OC-RAP(7).zip`；
- `all_regime_external_baselines_v50_1_full.zip`；
- 该结果包中保留的 launcher log、部分 result、progress 和 scene journal。

用户没有打包 `runs/ocrap_three_regime_closed_loop_v50_1_full` 下的 OC-RAP launcher log、progress 和 scene journal，因此无法从上传文件恢复 OC-RAP 进程最后一条 Python traceback 或系统 OOM 日志。本文对 OC-RAP 的判断分为：

1. **从代码可以直接证明的缺陷**；
2. **由这些缺陷形成的最可能故障链**。

不会把推断写成已观察到的 traceback。

## 2. 外部 baseline 的已确认错误

### 2.1 Safe：报告名称被错误地作为 runtime method

`safe/closed_loop_nominal_replay.log` 中的明确异常为：

```text
ValueError: Unknown evaluation method 'nominal_replay'; valid methods:
['nominal', 'log_replay', 'idm_proxy', 'mpc_proxy', 'risk_aware',
 'backup_filter', 'contingency', 'oracle_filter', 'ocrap', 'ocrap_teacher']
```

`nominal_replay` 是对比表中的报告名称；closed-loop runner 的合法方法名是 `nominal`。V50.2 将前者直接传给了 `closed_loop.method`。

修复：

- 输出文件仍使用 `closed_loop_nominal_replay.json`；
- 表格仍显示 `nominal_replay`；
- 真正传给 runner 的方法改为 `nominal`。

### 2.2 Near-contact：错误假设 Bash 支持 `wait -p`

`near.launcher.log` 中的明确异常为：

```text
wait: -p: invalid option
wait: usage: wait [-fn] [id ...]
PID_GPU: bad array subscript
```

V50.2 仅依据 `BASH_VERSINFO[0] >= 5` 选择动态调度器。但 Bash 5.0 虽支持 `wait -n`，并不一定支持 `wait -p`。`done_pid` 未被写入后，随后访问 `PID_GPU[$done_pid]` 又产生数组下标错误。

修复：

- 使用 `help wait` 进行功能探测，而不是只看主版本号；
- 不支持 `wait -p` 时自动使用 portable fixed-batch 调度；
- 无重训恢复命令显式设置 `USE_DYNAMIC_SCHEDULER=false`。

### 2.3 当前 external 结果的完成情况

上传的 `EXTERNAL_BASELINE_RUN_INDEX.json` 表明：

- Safe：`wayformer_bc` 已完成 175/175；`nominal_replay` 在第一个 scene 失败；GameFormer 和 BeTop 尚未进入闭环；
- Near：`gameformer_lite` 和 `marc_lite` 已完成 250/250；其余五个方法未启动；
- Contact：四个方法均完成 209/209。

因此恢复时只需要计算：

- Safe：nominal、GameFormer、BeTop；
- Near：RACP、predictive safety filter、DRO-CVaR、CVaR、expected-risk；
- 已完成的 Wayformer、Near GameFormer、MARC 和全部 Contact 直接跳过。

## 3. OC-RAP metric-only 的代码级故障

V50.2 中 `render_trace=false` 并不等于轻量结果：

1. 每个 scene 的 40 步 `decisions` 始终保留在 `scene_results`；
2. 每个 scene journal 行仍写入全部 decisions；
3. partial JSON 周期性嵌入此前全部 scenes；
4. 最终 JSON 嵌入全部 scenes；
5. CLI 完成后又执行 `print(result)`，把整个对象再次写入 launcher log。

上传的 external 结果可直接观察这一行为：

- 单个完整 result 为约 20–29 MB；
- 单个 scene journal 为约 15–22 MB；
- 日志末尾出现完整的逐步 decision 字典，而不是一个完成摘要。

对三个已完成 journal 的首条 scene 做同字段压缩后：

| Regime | 完整 scene | metric-only scene | 减少 |
|---|---:|---:|---:|
| Safe | 83,274 B | 6,081 B | 92.7% |
| Near | 84,587 B | 6,334 B | 92.5% |
| Contact | 85,612 B | 6,480 B | 92.4% |

OC-RAP 的 decision 结构比外部 baseline 更复杂，并且 Safe 与 Near 默认并行，因此 V50.2 很容易在最终聚合、反复序列化或日志打印阶段触发 RAM 峰值、磁盘配额、日志限制或系统 SIGKILL。SIGKILL 不执行 Bash `EXIT` trap，这能解释“scene 可能已跑完，但 `OCRAP_THREE_REGIME_RUN_INDEX.json` 完全不存在”的现象。

由于没有上传 OC-RAP 运行目录，本次不能判定服务器最终是 OOM killer、磁盘满还是调度器 kill；但上述 metric-only 实现缺陷是确定存在的，并且必须修复。

## 4. V50.3 修复

### 4.1 真正的 metric-only 持久化

新增 closed-loop 配置：

- `closed_loop.result_scene_detail=metrics|full`
- `closed_loop.scene_journal_detail=metrics|full`
- `closed_loop.memory_scene_detail=metrics|full`
- `closed_loop.include_scenes_in_result=false|true`
- `closed_loop.include_scenes_in_partial=false|true`

完整 benchmark 默认：

```text
result_scene_detail=metrics
scene_journal_detail=metrics
memory_scene_detail=metrics
include_scenes_in_result=false
include_scenes_in_partial=false
```

只保留 target identity、scene scalar metrics、macro counts、intervention summary 和 timing。10 个选择性视频场景重跑时才使用 `full` journal。

这些参数被定义为 persistence-only，不改变 run fingerprint，因此可以从 V50.2 的旧 journal 继续恢复，不需要 `resume_force=true`。

### 4.2 不再把完整 result 打印到日志

closed-loop CLI 现在只打印：

- method/source；
- num_scenes/num_decisions；
- bucket_target_count；
- fingerprint；
- scene storage detail。

完整结果只写 JSON 和 journal。

### 4.3 从 journal 原地重建结果

新增：

```text
tools/finalize_closed_loop_from_journal.py
```

它会：

- 流式读取旧 `.scenes.jsonl`；
- 忽略被 kill 时可能留下的最后一条破损 JSON；
- 按 target key 去重；
- 丢弃旧 decisions/render payload；
- 重新聚合所有指标；
- 重建 `closed_loop_ocrap.json`；
- 将 progress 标为 `complete`。

若 journal 已包含 Safe 175、Near 250、Contact 209 条，无需重新运行任何 Waymax rollout。

为防止误把半截 journal 当成完整结果，在 progress 缺少 `requested_rollouts` 时必须显式传 `--expected-count`。

### 4.4 可恢复的总索引

OC-RAP 三-regime 启动器现在：

- 启动任何 worker 前先写一个 incomplete index；
- 每个 regime 结束后刷新 index；
- 捕获 INT/TERM/HUP；
- result + complete progress + journal 的完整性优先于过期 phase 文件；
- 自动 finalize 完整 journal；
- 自动跳过已经完成的 regime；
- 只续跑缺失 target。

external 总索引也改成由 artifact completeness 判定，过期的 `phase=failed` 不再覆盖已经完整的结果。

### 4.5 跳过已完成方法

Safe、Near、Contact runner 新增：

```text
SKIP_COMPLETE_METHODS=true
```

已完成的 result/progress/journal 三元组不会再次扫描 WOMD。Near 的 GameFormer 如果闭环结果已经完整且不做 offline eval，也不会为了运行其余非学习方法而强制加载 checkpoint。

### 4.6 MP4 逻辑修正

完整 benchmark 只保存 metric journal。可视化过程为：

1. 对所有方法验证完全相同的 target-key 集合；
2. 选择 aggregate 指标最好的 deployable external comparator；
3. 从 Near 和 Contact metric journal 各选 5 条；
4. 仅对 10 个 target 重跑 OC-RAP 和 comparator，并保存 full trace；
5. 输出 H.264、1280×640、yuv420p MP4。

旧 journal 中的 decisions 在筛选时流式丢弃，避免为了选 10 个场景再次把全部重对象加载进内存。

选择分两层：

- `strict_material_improvement`：达到显式 TTC、clearance、exposure、escape、re-contact 或 stable-stop 阈值；
- `best_available_nonregressive`：严格候选不足 5 条时，只从 score>0、发生 intervention、无安全回归且无关键指标实质回归的场景补齐。

视频标题和 `VIDEO_INDEX.json` 会明确记录 selection tier 和 material improvements，不把 fallback 片段冒充强改善样例。

## 5. 是否重新训练

不需要重新训练。

上传日志显示：

- Safe Wayformer：已有 checkpoint 并复用；
- Safe GameFormer：已有 checkpoint 并复用；
- Safe BeTop：本轮已完成 30 epochs；
- Near GameFormer：本轮已完成 30 epochs，并生成/复用 checkpoint。

恢复命令设置：

```text
DO_TRAIN_SAFE=false
DO_TRAIN_NEAR=false
FORCE_RETRAIN_ALL=false
DO_OFFLINE=false
```

若服务器上的某个 checkpoint 文件已被删除或 contract 校验失败，脚本会明确退出，而不会静默重训。届时只需重训那个缺失方法，不需要重训全部 baseline。

## 6. 验证

- Python compileall：通过；
- 全部 shell 脚本 `bash -n`：通过；
- focused regression tests：26 passed；
- synthetic journal finalize / stale-index recovery：通过；
- exact-5 fallback selection：通过；
- MP4 smoke test：H.264、1280×640、yuv420p，通过 ffprobe。
