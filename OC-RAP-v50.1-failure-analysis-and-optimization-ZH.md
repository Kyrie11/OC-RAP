# OC-RAP v50.1：完整外部 Baseline / 三 Regime 闭环失败分析与修复

## 1. 结论

上传的 `all_regime_external_baselines_v49_full.zip` 并不是“训练或闭环跑到一半后失败”，而是在 **Safe regime 的闭环数据预检阶段立即退出**。压缩包中只有：

- `EXTERNAL_BASELINE_FIDELITY.json/.md`
- `safe.launcher.log`
- `safe/closed_loop_dataset_support.json`

没有任何训练日志、checkpoint、Near/Contact launcher 日志或闭环结果。因此：

1. 本次失败没有产生需要废弃的模型；
2. 不需要因为这次失败重新训练所有外部 baseline；
3. 直接触发失败的是无效 WOMD 路径 `/absolute/path/validation_tfexample.tfrecord@150`；
4. v49 代码还错误地把 `@150` 解释成“最多扫描 150 个 scenario”，而不是 150 个 TFRecord shard；
5. Safe 预检返回 3 后，父脚本受 `set -e` 影响退出，所以底部的 `EXTERNAL_BASELINE_RUN_INDEX.json` 写入逻辑根本没有执行。

OC-RAP 三 regime 的运行结果包未随本轮上传，因此无法从日志恢复其最后一个 Python traceback；但旧脚本存在确定性的同源错误：它给子脚本传递不带 `@150` 的 shard prefix，同时传入 `WOMD_LIMIT=0`，从而禁止子脚本自动补 `@150`。如果磁盘上只有 `prefix-00000-of-00150` 到 `prefix-00149-of-00150`，预检必然无法解析裸 prefix。

## 2. 上传结果中的直接证据

Safe launcher 输出的关键字段为：

```text
womd_pattern=/absolute/path/validation_tfexample.tfrecord@150
raw_scenario_scan_limit=150
resolved_womd_files=[]
num_resolved_womd_files=0
schema_supports_closed_loop=false
```

其中存在两个独立问题：

- `/absolute/path/...` 是 placeholder，而不是服务器上的真实文件前缀；
- `raw_scenario_scan_limit=150` 说明旧预检器把 `@150` 当成了 scenario 数量限制。

旧 `tools/check_closed_loop_dataset_support.py` 的 `_split_limit()` 会把 `@150` 剥离，然后只检查裸文件 `validation_tfexample.tfrecord` 是否存在。真实 WOMD shard 名称是 `validation_tfexample.tfrecord-00000-of-00150` 这类文件，因此会得到空列表。

## 3. 为什么没有生成 EXTERNAL_BASELINE_RUN_INDEX.json

旧总控脚本具有如下执行顺序：

1. `set -euo pipefail`；
2. 运行 Safe launcher；
3. Safe 预检返回非零状态 3；
4. 父脚本立即退出；
5. 位于脚本最末尾的 index Python block 没有执行。

v50.1 改为：

- 每个 regime 在开始、完成、失败时写入独立 `safe.phase.json`、`near.phase.json`、`contact.phase.json`；
- 默认一个 regime 失败后继续运行其余 regime；
- 用 `EXIT` trap 保证无论在哪一步失败，都生成 `EXTERNAL_BASELINE_RUN_INDEX.json`；
- index 逐方法检查 result、progress、scene journal、target 数量和 run fingerprint，而不只是枚举已有 JSON。

## 4. WOMD `@150` 的正确语义

`prefix@150` 是 TensorFlow/Waymax 的 shard-set 规格，表示完整集合：

```text
prefix-00000-of-00150
...
prefix-00149-of-00150
```

它不是 scenario 截断，也不能与 `MAX_SCENARIOS=150` 等同。v50.1 将两个概念完全分离：

- `@150`：文件分片集合声明；
- `MAX_SCENARIOS` / `closed_loop.max_scenarios`：要完成的闭环 target 数；
- runner 内部根据离线数据中的 `source_scenario_index` 推导的 scan bound：仅用于避免解码最后一个目标之后的无关 raw records，绝不改变 shard 集合。

新预检器要求 150 个 shard 全部可解析，并会明确列出缺失 shard、placeholder、重复文件和裸 prefix 错误。

## 5. OC-RAP 三 Regime 旧脚本的确定性问题

### 5.1 丢失 `@150`

旧 `run_ocrap_three_regime_closed_loop.sh` 默认值是不带 `@150` 的 prefix，并向子脚本传递 `WOMD_LIMIT=0`。旧子脚本只有在 `WOMD_LIMIT>0` 时才补后缀，因此最终仍是裸 prefix。

### 5.2 Safe 失败会阻断 Contact

旧脚本只有在 Safe 成功后才启动 Contact。Safe 的路径、显存或单个场景错误都会导致 Contact 完全没有结果。v50.1 三个 regime 都有独立状态并尽力完成。

### 5.3 单 GPU 时并发争抢

旧脚本在只有一个可见 GPU 时仍会并行启动 Safe 和 Near，两套 JAX/PyTorch runtime 可能同时编译和申请显存。v50.1 检测到单 GPU 后改为串行；两 GPU 时 Safe/Near 并行，然后运行 Contact。

### 5.4 不安全续跑

旧脚本默认 `RESUME_FORCE=true`，会忽略配置 fingerprint 差异继续使用历史 partial/journal。v50.1 默认 `false`，配置、checkpoint、目标集合发生变化时拒绝错误续跑。

### 5.5 模型目录兼容

已有 RC=20 产物可能使用：

```text
candidates/<variant>/...
```

或：

```text
dedicated_candidates/<variant>/...
```

v50.1 自动检查两种布局；也允许显式设置 `CHECKPOINT` 和 `GAMMA_REC_JSON`。

## 6. 是否需要重新训练外部 Baseline

### 6.1 本次失败不要求重训

失败发生在 Safe 预检，训练尚未开始。只要服务器上已有 v49 训练所得且通过 checkpoint contract 校验的模型，就可以直接复用。

### 6.2 真正有神经网络 checkpoint 的方法

需要训练权重的只有：

- Safe / Wayformer BC
- Safe / GameFormer-lite
- Safe / BeTopNet-lite
- Near-contact / GameFormer-lite

MARC-lite、RACP-lite、expected/CVaR/DRO-CVaR filter、predictive safety filter、全部 Contact baseline 和 nominal replay 都没有独立神经网络训练阶段。

### 6.3 v50.1 的训练策略

`DO_TRAIN_SAFE=true` 和 `DO_TRAIN_NEAR=true` 的含义变为“允许训练缺失或无效 checkpoint”，不是强制重训：

- checkpoint 存在并通过 `validate_external_checkpoint.py`：直接复用；
- checkpoint 缺失或 contract 不合法：仅训练对应模型；
- `FORCE_RETRAIN_ALL=true`：才强制重新训练上述四个可训练模型。

本次推荐使用 `FORCE_RETRAIN_ALL=false`。

## 7. 完整闭环的目标数量

根据上传的数据报告，在 `MAX_TARGETS_PER_SCENE=1` 时，“全部独立 scene”对应：

| Regime | 测试样本 | scene-time groups | 独立 scene / 默认闭环 targets |
|---|---:|---:|---:|
| Safe | 3216 | 402 | 175 |
| Near-contact | 4723 | 595 | 250 |
| Contact | 6687 | 747 | 209 |

`MAX_SCENARIOS=0` 表示运行 loader 最终加载的全部 targets。若想对每个 scene 的多个时间点都做闭环，需要显式提高 `MAX_TARGETS_PER_SCENE`，这不再属于“独立 scene 一条”的主协议。

## 8. 训练和闭环速度优化

### 8.1 不再为全量实验保存所有 render trace

旧流程对 Near/Contact 的所有方法和所有 scene 保存每一步 agent box，之后才挑 10 条视频。这会显著增加：

- JSON 序列化和磁盘写入；
- Python 对象内存；
- scene journal 文件体积；
- 多方法闭环总时间。

v50.1 的流程为：

1. 全量方法运行 metric-only 闭环，`RENDER_TRACES=false`；
2. 用完整、配对的逐 scene 指标选出 Near 5 条、Contact 5 条 target key；
3. 仅对这 10 个 target，用 OC-RAP 和选中的最佳外部 baseline 重跑 trace；
4. 渲染 10 个 MP4。

### 8.2 关闭主表不需要的教师审计

全量 external 和 OC-RAP 闭环默认：

```text
label_mode=fast
audit_every_n_steps=0
```

Near 的 oracle 只作为 teacher-only 诊断，默认不参与全量闭环；需要时限制为小规模场景。

### 8.3 Raw WOMD 提前停止

旧数据的 scene id 含 `__wxNNNNN`，v50.1 可恢复 legacy source index。如果所有目标都有 index，runner 在读到最大 target index 后停止，不再扫描后续无关 WOMD records。该优化与 `@150` shard 声明无关。

### 8.4 资源控制

- 每 GPU 使用独立 JAX persistent compilation cache，避免并行进程写同一 cache；
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`；
- 限制 OMP/MKL/OpenBLAS/TF intra-op 线程，避免多进程 CPU oversubscription；
- `MALLOC_ARENA_MAX=4`；
- 两 GPU Near baseline 使用动态回填：先完成的方法释放 GPU 后立即启动下一个；
- 单 GPU 视频 trace 串行执行，避免 OC-RAP 与 comparator 同时 OOM；
- 训练关闭 tqdm，并保持 DDP validation 不补齐重复样本；
- full run 保留 partial/journal，严格 fingerprint 下支持 scene 级断点续跑。

### 8.5 可选离线评测

最终三张主表只消费闭环 JSON/journal。若当前目标是尽快得到主表，外部总控可设置：

```text
DO_OFFLINE=false
```

需要论文附录的离线 Brier/ECE/selected-risk/severity 等指标时再设为 `true`。

## 9. MP4 逻辑复核和修复

### 9.1 原逻辑的问题

- 在全量实验中保存 trace，代价过高；
- Contact 中把“初始/因果碰撞”当成新的安全回归，会错误排除真正恢复好的片段；
- 只画车辆框，没有道路几何，不利于解释逃逸空间和越界；
- 选择与重渲染没有严格 target-key 预检；
- 单 GPU 时两个 trace 任务并发可能 OOM；
- 选中场景不足 5+5 时可能直到渲染后才暴露问题。

### 9.2 v50.1 的选择协议

Near 候选至少满足一项实质改善，例如：

- TTC p05 提升；
- clearance p05 或 terminal clearance 提升；
- critical-TTC exposure 减少；

同时不得新增碰撞/越界，不允许 TTC、clearance 或 exposure 出现实质回归。

Contact 将初始碰撞视为 causal anchor，不以 `overlap_any` 自动否决；重点比较：

- post-contact terminal clearance；
- free-space AUC；
- clearance gain；
- overlap duration；
- escape event；
- re-contact；
- new stable stop quality；
- off-road。

新增 re-contact/off-road 或明显延长碰撞后重叠时间的片段会被拒绝。

### 9.3 新渲染内容

选择性 trace 会为每个 scene 只保存一次附近 WOMD roadgraph polyline，并在视频中显示：

- 相同的 ego-centric 视野；
- 道路几何；
- OC-RAP 与 comparator 双栏车辆框；
- SDC 历史轨迹；
- 当前最小 clearance 圆；
- observed overlap 或 causal contact anchor；
- TTC、clearance、overlap、off-road；
- Near/Contact 关键指标差值；
- 两侧 rollout 长度不同时明确标注 held final state。

输出使用 H.264/yuv420p、`faststart`，便于论文网页和常用播放器播放。

选择阈值已暴露为环境变量。降低阈值不会取消“不得出现 unsafe regression”的硬约束。选不出严格的 5+5 时，视频阶段会明确失败，而不会用质量差或未配对场景凑数。

## 10. 新增/修改的关键文件

### 新增

- `src/ocrap/data/womd/sharded_path.py`
- `tools/validate_womd_spec.py`
- `tools/build_external_baseline_run_index.py`
- `tools/build_ocrap_three_regime_index.py`
- `scripts/lib/v50_runtime.sh`
- `scripts/run_selected_recovery_video_traces.sh`
- `tests/test_v50_full_regime_runtime.py`

### 修改

- `tools/check_closed_loop_dataset_support.py`
- `src/ocrap/simulation/closed_loop_runner.py`
- 三个 regime external launcher
- external 总控 launcher
- OC-RAP 单 regime / 三 regime launcher
- comparison table / critical-scene selector / MP4 renderer
- comparison artifact / top-10 video scripts

## 11. 验证结果

- 修改脚本 `bash -n`：通过；
- Python `compileall`：通过；
- 针对性测试：27 passed，1 个既有 Transformer warning；
- `@N` shard 解析、选定 target 预检、失败时 index、Contact 选择和 roadgraph context 均有回归测试；
- 外部总控提前失败 smoke test：退出码为 1，但仍正确生成不完整的 `EXTERNAL_BASELINE_RUN_INDEX.json`；
- MP4 smoke test：成功生成并由 `ffprobe` 读取的 H.264 MP4。

当前执行环境没有服务器上的 150 个真实 WOMD shard 和大 checkpoint，因此没有伪造最终闭环数值或 10 个真实视频。完整结果应在原实验服务器按附带指令运行。
