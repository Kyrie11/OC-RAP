# OC-RAP v48.45.4 失败归因与 v48.45.5 工程修复

## 结论

本轮上传结果不是 SOWR 算法失败。v48.45.4 的 shared source 已经成功构建并封存，但 A/B/C/D 四个 SOWR arm 在 adaptation 之前全部因为同一个 shared dataset protocol 缺失而以 RC=30 结束。因此这一轮不能用于 B-A、C-A 或 D-B-C+A 的算法归因。

直接根因是上一版 operator command 漏掉了 `partition_dedicated_calibration_v48_14.py`。SOWR/OCAF controller 默认要求 `$OCRAP_ROOT/calibration_v48_14_prism_4814` 下存在六个 scene-disjoint role root，但真实机器只有 `calibration_near_contact`、`calibration_contact`、`calibration_safe`，所以四个 arm 都必然在 `dataset_root_contract` fail-close。

v48.45.5 只修工程和 provenance，不修改 SOWR 算法、ROCT、top-k、shared continuous rule、harm budget、certificate 或 Natural gate。

## 1. 上传的 shared source 是有效的

`ocrap_v48_45_source_rebuild_s7(1)` 中：

- `S1_SOURCE_POLICY_STATUS.json`: balanced exit code = 0，precision exit code = 0，`both_succeeded=true`；
- Balanced checkpoint SHA256 = `070218c66e506d66f25a12bf53b4127581992d75481bbb635d7ea658f4cfd352`；
- Precision checkpoint SHA256 = `8f7528b76ce4b2424c5f153fe3109844de27a08a976a067b1292baee393f768d`；
- source checkpoint contract `valid=true`；
- source quality contract `valid=true`；
- source rebuild 未读取 test roots。

因此上一轮 v48.45.4 对 S1 nounset bug 的修复是有效的，这一轮不需要重训 source，只需保留并复用这份 sealed source。

## 2. 四个 arm 的第一真实失败点完全一致

Main、A、B、C 的 `PIPELINE_FAILED.json` 均满足：

- `stage = dataset_root_contract`
- `raw_exit_code = 4`
- `normalized_exit_code = 30`
- `certificate_executed = false`
- `gate_evaluated = false`
- adaptation exit code 仍为 null
- `test_roots_read = false`

各 arm 的 `DATASET_ROOT_CONTRACT.json` 也完全一致：

- 六个 canonical path 判断全部为 true；
- 六个 `*_exists` 判断全部为 false；
- `safe_root_exists = true`；
- Near/Contact path distinct = true；
- 没有 legacy alias。

缺失的目录为：

```text
calibration_v48_14_prism_4814/
  evidence_adapt_train_near_contact
  evidence_adapt_train_contact
  evidence_adapt_dev_near_contact
  evidence_adapt_dev_contact
  certificate_pool_near_contact
  certificate_pool_contact
```

这说明 controller 自身没有选错 leaf；缺失的是顶层实验准备步骤。

## 3. 为什么不能直接把 train/val/calibration 目录硬塞给 controller

当前 v48.36+ certificate protocol 有明确的数据角色语义：

- `evidence_adapt_train`：只用于 target evidence/SOWR/OCAF adaptation；
- `evidence_adapt_dev`：只用于 early stop / shared rule fitting；
- `certificate_pool`：独立 certificate verify；
- `calibration_safe`：Safe calibration；
- `test_*`：本轮不读取。

如果简单把 `train_near_contact` 当 evidence-adapt train、`val_near_contact` 当 evidence-adapt dev、`calibration_near_contact` 当 certificate，虽然工程上可能能跑，但会改变 v48.44/v48.45 已预注册的 calibration protocol，从而把“修工程”与“改实验设计”混在一起，破坏这次 SOWR 2x2 的可比性。

因此 v48.45.5 保留已有 v48.14 scene-disjoint protocol：只从 `calibration_near_contact/contact` 中按 scene 确定性分为 45% adaptation train、15% adaptation dev、40% certificate，seed 固定 4814。Source 仍然只用 train/val。

## 4. v48.45.5 新增的工程保证

### 4.1 Shared protocol bootstrap

新增：

```text
scripts/prepare_v48_45_protocol.sh
```

行为：

1. 只读取：
   - `calibration_near_contact`
   - `calibration_contact`
   - `calibration_safe`
2. 不接受 `test_*` 作为输入；
3. 调用现有 `partition_dedicated_calibration_v48_14.py`；
4. seed=4814；train/dev/cert fractions = 0.45/0.15/0.40；
5. 默认 hardlink，hardlink 不可用时由原 partition tool 回退 symlink/copy；
6. 有效旧 protocol 会验证后复用，不重新切分；
7. partial/invalid protocol 会先备份，再构建；任何失败会 rollback。

### 4.2 Independent protocol seal

新增：

```text
tools/check_v48_45_protocol_seal.py
```

它独立验证：

- 三个原始输入 manifest 的 `split_id` 必须全部是 `calibration`；
- 六个 role manifest 必须非空；
- `split_id` / `calibration_protocol_role` 必须严格匹配角色；
- manifest 引用的 sample 必须存在；
- sample path 不允许重复；
- adaptation-train/dev/certificate 的 scene 两两不相交；
- 三个 role scene union 必须精确等于原 calibration scene 集；
- 每一个 scene 的 role assignment 必须精确复现 `v48.14|4814|scene` 的 hash 分配；
- `split_provenance.json` 的 source/role/seed/fractions 必须一致；
- 记录源和派生 role manifest SHA256；
- `test_roots_read=false`。

因此四个 arm 不再仅依赖“目录存在”，而是共享一份经过确定性 provenance seal 的 calibration protocol。

### 4.3 2x2 launcher shared preflight

`run_v48_45_sowr_2x2_parallel.sh` 在启动任何 arm 前构建/验证一次 shared protocol。Direct single-arm invocation 也会自动 prepare/verify。这样 shared input 配置错误不会再产生四个重复 RC=30 后才被发现。

### 4.4 Source 保持冻结

本轮上传 source 已经有效，因此新 operator script 不删除、不重训 sealed source。只有 `SOURCE_REBUILD_COMPLETE.json` 不存在时才进入 resumable source rebuild。

## 5. 算法归因边界保持不变

四个 arm 仍是：

- A：v48.44-D reference，无 SOWR；
- B：A + root probability/recovery-margin witness recalibration；
- C：A + observation-kernel recalibration；
- D/Main：B + C。

固定项仍包括：

- 同一个 source checkpoint pair；
- 同一个 scene-disjoint calibration protocol；
- `PROPOSAL_TOP_K=5`；
- dual ROCT；
- 同一 ROCT scale；
- 同一个 shared continuous rule；
- 同一 Near/Contact harm budget；
- 同一 certificate 和 Natural gate。

没有增加 Safe/Near/Contact identifier、router、专用 threshold 或专用 policy。

只有当四个 arm 都获得 authoritative RC=0/20 后，才能分析：

```text
margin/root witness effect = B - A
observation-kernel effect   = C - A
interaction effect          = D - B - C + A
```

RC=20 是 pipeline-valid 的算法负结果；RC=30/其他仍然是工程/protocol failure，不得做算法归因。

## 6. 验证结果

完成验证：

- v48.45 focused: 29/29 passed；
- v48.42-v48.45: 56/56 passed；
- v48.36 controller/terminal suites: 32 passed / 1 skipped；
- v48.37-v48.41: 29/29 passed；
- 合计完成的 v48.36-v48.45 regression: 117 passed / 1 skipped；
- `compileall`: PASS；
- 100/100 shell scripts `bash -n`: PASS；
- v48.45.5 operator command `bash -n`: PASS；
- repository-wide nounset same-local self-dependency: 0 findings。

另外做了一个 controller-path integration smoke：用 synthetic calibration protocol + fake source checkpoint 启动 v48.45 A arm。它成功通过：

1. source checkpoint contract；
2. dataset root contract；
3. dedicated protocol audit；
4. multigroup loss contract；
5. OCAF bridge contract；

随后只在 `ocaf_cuda_group_broadcast_preflight_gpu0` 停止，因为当前验证容器没有训练 GPU。也就是说，这次上传中实际发生的 dataset/protocol RC=30 已经从真实 controller path 中消除。

## 7. 下一步

直接运行：

```bash
bash OC-RAP-v48.45.5-protocol-bootstrap-and-SOWR-run-commands-ZH.txt
```

不要删除：

```text
runs/ocrap_v48_45_source_rebuild_s7
```

该 source 本轮已经有效。

新脚本会先准备和 seal calibration protocol，然后清理仅 A/B/C/D 的旧 RC=30 输出，再运行严格 SOWR 2x2。若任何 arm 再出现工程失败，operator script 会直接打印对应 `PIPELINE_FAILED.json`，不继续把它误当算法结果。

- 为防止“能跑完但共享输入漂移”污染消融归因，每个 arm 记录 protocol seal SHA256；最终 2×2 comparator 对 protocol/source/gate/dataset manifest identity 做 fail-closed attribution contract。
