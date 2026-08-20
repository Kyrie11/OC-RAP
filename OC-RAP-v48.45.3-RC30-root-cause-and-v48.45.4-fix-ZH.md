# OC-RAP v48.45.3 RC=30 根因审计与 v48.45.4 工程修复

## 1. 结论

这次上传的 `ocrap_v48_45_source_rebuild_s7` **不能用于 SOWR 算法归因**。它不是 Natural gate 的算法失败，而是在 shared-source rebuild 的 `S1_source_policy_heads` 阶段发生工程失败。S0 shared recovery backbone 已完成；S1 两个 source heads 都在训练函数真正启动之前退出，因此后续 A/B/C/D 没有共同、完整、hash-sealed 的 source checkpoint，继续跑消融没有统计/因果意义。

v48.45.4 只修工程，不改变 SOWR、ROCT、top-k、shared rule、harm budget、数据划分或 certificate/gate。当前最合理的下一步是恢复 source 并完成原先预注册的 2x2，而不是基于 RC=30 再添加算法模块。

## 2. 论文与当前算法语义

论文 OC-RAP 的核心不是“在三个 regime 上各自做策略”，而是把 **deployable recoverability** 定义成候选 prefix 的统一性质：

1. 预测 recovery-sufficient latent roots；
2. 预测 post-prefix observation embedding 并构造 observation-equivalence/compatibility；
3. 对 observation-indistinguishable roots 强制共享同一个 compatible recovery option；
4. 用 OC-MERO 先在同一 shared option 上做 lower-tail recovery，再对 roots 做 lower-tail 聚合；
5. CRISP 把 calibrated deployable recoverability 作为 admission property，在 nominal 已可恢复时保持 nominal utility。

这也是 v48.45 SOWR 的正确上游定位：它不增加一个新的 downstream residual/head，而是重新校准 paper-semantic witness：`root_logit_head`、`margin_head` 和/或 `obs_embed_head`，随后再次冻结 witness，再运行相同的 downstream OCAF/ROCT/certificate。

严格 2x2 的算法因果解释保持：

- **A**：v48.44-D dual-ROCT reference，不做 SOWR；
- **B**：A + root probability / recovery-margin witness recalibration；
- **C**：A + observation-kernel recalibration；
- **D/Main**：A + B+C。

三个 regime 只作为训练/评估 strata；SOWR 本身不读取 Safe/Near/Contact identifier，也不增加 regime-specific threshold/policy/router。

## 3. 上传结果的证据链

### 3.1 S0 实际成功

`SOURCE_QUALITY_CONTRACT_RECHECK.json` 记录的 S0：

- train samples: **50,114**
- validation samples: **12,250**
- epochs completed: **13**
- best epoch: **7**
- best validation loss: **5.2515695061**
- `init_checkpoint=""`（scratch S0）
- 原运行机器上的 `shared_recovery_backbone/.../best.pt` 存在

因此当前失败与 v48.45.3 之前修过的 empty-override/`None` 问题不同；那个问题已经被越过。

### 3.2 S1 在训练开始前失败

上传包保留：

```text
[source rebuild] S0 shared recovery backbone on GPU 0
source policy status: balanced=1 precision=1
```

同时：

- `SOURCE_REBUILD_FAILED.json.stage = S1_source_policy_heads`
- `raw_exit_code = 30`
- `source_rebuild_complete = false`
- balanced `train_summary.json` 不存在
- precision `train_summary.json` 不存在
- 上传包中甚至没有 `candidates/balanced` / `candidates/precision` 目录

这说明两个后台 S1 函数都在 `mkdir -p "$run/logs"` 之前就退出，而不是 loss NaN、OOM、checkpoint save 或 quality gate 失败。

## 4. RC=30 的直接根因

原代码：

```bash
train_source_variant() {
  local variant="$1" gpu="$2" run="$SOURCE_OUT/candidates/$variant"
  ...
}
```

文件顶部启用了：

```bash
set -Eeuo pipefail
```

Bash 对同一个 `local` builtin 的所有 RHS 先进行参数展开，之后才完成这些局部变量赋值。因此执行第三个 RHS 时，当前 local `variant` 尚未建立；在 `set -u` 下 `$variant` 是 unbound variable，函数直接终止。

最小复现：

```bash
bash -uc 'f(){ local variant="$1" gpu="$2" run="/tmp/$variant"; }; f balanced 0'
# variant: unbound variable
```

两个 S1 都在后台调用这个函数，因此观察到 `balanced=1 precision=1`。随后 source rebuild 外层用 `[[ s0==0 && s1==0 ]] || exit 30` 把它规范化成 RC=30。

这条证据链完整解释了结果，不需要假设 GPU、数据、模型或算法原因。

## 5. v48.45.4 已修复的工程问题

### 5.1 S1 nounset 修复

改为：

```bash
local variant gpu run
variant="$1"
gpu="$2"
run="$SOURCE_OUT/candidates/$variant"
```

并新增 `S1_SOURCE_POLICY_STATUS.json`，保留 balanced/precision 两个原始退出码，避免下一次只看到外层 RC=30。

### 5.2 保留已完成 S0

v48.45.3 的 operator commands 在 source 未 seal 时会 `rm -rf "$SOURCE_RUN"`。对当前结果来说这会把已经花成本训练完成的 S0 一并删掉。

v48.45.4 不再删除整个 source。若：

```text
shared_recovery_backbone/model_v48_trac_sr/best.pt
shared_recovery_backbone/TRAINING_COMPLETE.json
```

二者都存在，则直接复用 S0，只重训 S1。若只有一个存在，rebuild 脚本才把不完整 S0 清理后重训。

### 5.3 双 GPU 并发语义修复

原 2x2 launcher 默认 `MAX_PARALLEL_ARMS=4`。但每个 arm 内部已经同时运行：

- Balanced -> GPU0
- Precision -> GPU1

所以 4 arms 并发意味着最多 **8 个训练进程 / 2 GPUs**，可能引入 OOM、data-loader/IO contention、非确定性抖动，反而污染 ablation 归因。

v48.45.4 默认 `MAX_PARALLEL_ARMS=1`：**每个 arm 仍同时使用两张 GPU**，但 arm 之间顺序执行。若确认显存和 IO 足够，可手动设为 2 做吞吐优化；首次可归因 round 推荐 1。

### 5.4 同类 latent shell bug

仓库扫描还发现同样的 `local` 同命令自依赖风险存在于：

- `scripts/run_v48_36_ocaf_ablations.sh`
- `scripts/run_v48_34_exploratory_closed_loop_baselines_and_videos.sh`
- `scripts/run_v48_34_1_exploratory_closed_loop_baselines_and_videos.sh`

均已改成先声明、后赋值。新增 repository-wide regression 自动扫描此错误类。

## 6. 为什么当前不能继续改算法

上一轮 v48.44 的结果是有效的算法失败：RC=20、pipeline valid、certificate/gate 都执行，因此它支持“frozen witness calibration/identifiability 可能是 bottleneck”的 SOWR 假设。

但本轮 v48.45.3 在 S1 source 构建处停止，没有完成 source，更没有开始可比较的 SOWR A/B/C/D。因此本轮没有新的 B-A / C-A / D interaction 数据。此时再改 loss、width、threshold、head、top-k 或 risk budget，会把“修工程”与“改算法”混在同一轮，失去因果解释。

所以 v48.45.4 的正确选择是 **算法冻结，工程修复**。

## 7. 数据报告对下一轮解释的约束

数据本身支持做统一 recoverability，而不适合做 regime router。关键统计如下：

| split | regime | samples | groups | scenes | artifact frac | neg. deployable | oracle recoverable | R_dep mean | R_dep p05 | incompatible alias frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | safe | 20,000 | 2,500 | 1,171 | 0.0000 | 0.0988 | 0.9012 | 0.9369 | -1.0397 | 0.0000 |
| train | near | 13,324 | 1,800 | 600 | 0.1894 | 0.5531 | 0.6363 | -1.7941 | -9.1663 | 0.1615 |
| train | contact | 16,790 | 2,000 | 500 | 0.1662 | 0.5435 | 0.6227 | -1.7923 | -9.3737 | 0.0948 |
| val | safe | 2,328 | 291 | 132 | 0.0000 | 0.0735 | 0.9265 | 0.9737 | -0.6310 | 0.0000 |
| val | near | 3,445 | 433 | 176 | 0.2459 | 0.5042 | 0.7417 | -0.8008 | -5.1330 | 0.2039 |
| val | contact | 6,477 | 723 | 211 | 0.2186 | 0.4615 | 0.7571 | -0.5607 | -4.3168 | 0.1354 |
| cal | safe | 2,544 | 318 | 135 | 0.0000 | 0.0535 | 0.9465 | 1.1097 | -0.2453 | 0.0000 |
| cal | near | 6,039 | 765 | 316 | 0.2399 | 0.4483 | 0.7917 | -0.5089 | -3.3925 | 0.1957 |
| cal | contact | 16,843 | 1,896 | 543 | 0.2121 | 0.4169 | 0.7952 | -0.3514 | -3.0816 | 0.1423 |
| test | safe | 3,216 | 402 | 175 | 0.0000 | 0.0690 | 0.9310 | 0.9883 | -0.8338 | 0.0000 |
| test | near | 4,723 | 595 | 250 | 0.2441 | 0.4880 | 0.7561 | -0.6895 | -4.6120 | 0.2092 |
| test | contact | 6,687 | 747 | 209 | 0.2180 | 0.4440 | 0.7740 | -0.5722 | -4.7033 | 0.1405 |

值得关注的是 Near 和 Contact 的 alias incompatibility 从 train 到 val/test 上升：

- Near: `0.1615 -> 0.2039 / 0.2092`
- Contact: `0.0948 -> 0.1354 / 0.1405`

这和上一轮发现的 development -> certificate witness 泛化断层方向一致，因此 **SOWR 假设是合理的**；但这只是数据/历史结果支持的假设，不是当前 RC=30 的算法归因。

Safe 的 alias incompatibility 为 0，而且报告中大量 warning 来自 safe 数据没有 targeted future；报告的 `failures=[]`，因此这些 warning 不能被误判为当前 RC=30 的 pipeline 原因。

## 8. CCF-A readiness：内部目标，不是官方门槛

当前仓库的预注册内部 target 可以继续使用，但应在论文中表述为 project readiness criterion，而不是“CCF-A 官方阈值”。

### Near-contact

建议至少满足：

- verify recall: **0.25–0.33**
- harmful-selected UCB: **<= 0.25**（fit <= 0.22）
- verify precision LCB: **>= 0.40**（fit >= 0.50）
- min-TTC p05: **+0.2 s** 和/或 clearance **+0.1 m**，带 CI
- collision / hard-brake 降低
- nominal utility loss <= **2–3%**

### Contact

建议至少满足：

- verify recall: **0.20–0.30**
- harmful-selected UCB: **<= 0.25**（fit <= 0.22）
- verify precision LCB: **>= 0.40**（fit >= 0.50）
- secondary collision absolute reduction **>= 2 pp**
- post-contact TTC **+0.2 s**
- overlap / delta-v 下降
- stable-stop / route-rejoin 提升，yaw non-inferiority

### Safe

Safe 不是要追求更激进的 recovery capture，而是证明统一 recoverability primitive 不破坏正常驾驶：NUP paired lower CI > -1%（pre-register 最多 -2%）、progress delta > -0.5%、无 safety/comfort degradation、intervention <1–2%、FRA <=1%。

## 9. 下一轮实验设计与判定顺序

### Stage 0 — source engineering gate

必须全部通过后才进入算法实验：

1. S0 `best.pt + TRAINING_COMPLETE.json`；
2. S1 balanced/precision raw exit 都为 0；
3. `SOURCE_REBUILD_COMPLETE.json` 存在；
4. two checkpoint hash 与 manifest 一致；
5. source quality contract valid；
6. test roots 未读取。

任何一项失败都归类为 engineering failure，不允许解释算法。

### Stage 1 — v48.45 strict 2x2

四个 arm 使用同一个 sealed source、相同 seed protocol、相同 top-k=5、shared continuous rule、harm budgets、calibration/certificate pipeline。

因果量：

- **Margin/root main effect** = B - A
- **Observation-kernel main effect** = C - A
- **Interaction** = D - B - C + A

首先看 development 和 certificate 是否方向一致，而不是只看某个 aggregate recall。

### Stage 2 — 预注册诊断读数

Near：

- deployability AUC / false-veto frontier
- positive_selected / verify recall
- harmful false-safe / harmful-selected UCB
- development -> certificate gap

Contact：

- development safe-positive 的 `pred_adv>=0` 是否从历史 **0/37** 变为稳定非零
- proposal/candidate SP AUC
- positive_selected / recall
- harmful UCB
- secondary-collision/post-contact metrics（只有 algorithm arm 先通过 certificate 后再花 test budget）

判断：

- 若 B 明显改善 Near frontier 且 certificate 可复现：root/margin witness 是瓶颈之一；
- 若 C 首次修复 Contact sign geometry 且不增加 harm：observation kernel 是瓶颈之一；
- 若 D > B、C 且 interaction 为正：两个 witness 存在互补；
- 若 development 改善但 certificate 不改善：仍是 witness generalization/calibration 问题，不应继续放大 downstream scale；
- 若 recall 上升但 harmful UCB 越预算：不接受，不能用 risk trade 换 paper claim。

### Stage 3 — Test 使用原则

不要拿 test 反复调 threshold。先在 train/val/calibration/certificate 完成因素选择和预注册 gate；仅对固定后的 main algorithm + 必要 baseline/ablation 做最终 held-out test，并报告 scenario bootstrap CI。

## 10. 仍需后续处理但本轮不改的 novelty 风险

当前论文 Appendix 仍写有 `Regime-conditioned recovery admission` / “used only in low-headroom contact regimes”。代码历史上也保留了 `bucket_id`、`DELTA_REGIME_EXPERTS=true` 等 legacy source geometry。虽然 v48.45 的 SOWR/ROCT shared rule 本身没有新增 regime routing，这些遗留语义仍可能让审稿人质疑“统一 recoverability primitive”是否被 regime heuristic 支撑。

**本轮不能删除它们**：当前 source/下游 checkpoint geometry 依赖这些结构，直接移除会同时改变 source architecture 与 SOWR factor，破坏本次工程恢复后的 2x2 归因。

若 v48.45 完成后获得有效算法结果，下一阶段应把这些 legacy bucket-conditioned 机制替换为连续、可观测的统一变量（例如 deployable headroom、shared-option compatibility/conflict pressure、harm envelope），并做一个独立的 “remove regime/bucket conditioning” 结构消融。那才是面向 CCF-A novelty 的干净改动。

## 11. 本地验证

- v48.45 focused: **25 passed**
- v48.42–v48.45: **52 passed**
- `compileall`: PASS
- 99/99 `scripts/*.sh`: `bash -n` PASS
- 新 operator command: `bash -n` PASS
- repository-wide same-local nounset dependency scan: **0**
- 原错误 minimal reproduction: unbound `variant`
- 修复 pattern minimal reproduction: successful `/tmp/balanced`

