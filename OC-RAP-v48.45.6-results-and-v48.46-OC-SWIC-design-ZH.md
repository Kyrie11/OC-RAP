# OC-RAP v48.45.6 有效归因审计与 v48.46 OC-SWIC 设计/工程/性能报告

## 1. 结论

本次上传的 v48.45.6 A/B/C/D(Main) 已经全部是可用于算法归因的有效结果，而不是 pipeline fail：四个 arm 均 authoritative RC=20、pipeline_valid=true、certificate 已执行、Natural gate 已评估，且 test_roots_read=false。四臂 dedicated-calibration protocol seal、Balanced/Precision source checkpoint SHA256、gate protocol 与 development/certificate manifest 完全一致。

因此 v48.45 SOWR 的结论是一个有效的**算法负结果**：joint SOWR 没有通过 gate，但 B/C 提供了不同方向的局部正信号。

## 2. v48.45.6 SOWR 2x2 归因

Precision A 基线：Near certificate recall=0.1111、harmful-selected UCB90=0.2361、deployability harmful-vs-safe-positive AUC=0.4510、safe-positive false-veto=14/16；Contact certificate recall=0、UCB90=0.2921、deployability AUC=0.5252、false-veto=30/31，development safe-positive pred_adv>=0 仅 1/37。

B(root+margin) 的可保留信号是 discriminability/capture：Near/Contact candidate safe-positive AUC 分别约 +0.026/+0.027，Contact certificate 从 0 positive 变成 1/20（recall=0.05），false-veto 30/31→29/31。代价是 Near/Contact certificate harmful UCB90 恶化到 0.2980/0.3151。因此 B 不能整体吸收。

C(obs-only) 的可保留信号是 risk suppression：Near certificate harmful UCB90 0.2361→0.1532，Contact 0.2921→0.2368；Contact development safe-positive pred_adv>=0 从 1/37→2/37。但 Near certificate positive capture 从 1 降为 0。因此 C 也不能整体成为主算法。

D/Main 没有把两侧优点叠加：Near/Contact certificate UCB90=0.2980/0.3189，Contact development sign 回到 1/37，整体行为接近 B。其 interaction 对风险是负面的，所以 simultaneous root+margin+obs joint SOWR 被拒绝。

Balanced selector方向一致：B/C/D 将 Contact development recall 0.1176→0.1765，但 certificate 最高仍仅 0.05；Near 仍很弱。

## 3. witness loss 归因

B/D 中 margin、deployability、oracle、option-q/admission/best-option validation loss 均下降，但 root loss 略恶化；C 的 observation loss 显著下降并伴随 certificate risk UCB 下降。因此下一版：

- 不再更新 root_logit_head；
- observation witness 保留为风险侧候选；
- margin witness 保留为 capture 侧候选；
- 不再 simultaneous joint train，而是 obs→freeze→margin→freeze。

## 4. 更根本的论文—实现语义错配

核心 `src/ocrap/algorithms/ocmero.py` 与论文公式一直是 row-wise：对每个 post-prefix observation-conditioned q[i,l] 先 max_l，再在 root probability 上做 outer LCVaR。compatibility kernel 已在 q[i,l] 内把不可区分 roots 绑定到同一个 compatible recovery option；不同、可区分 observation classes 不需要全局使用同一 recovery option。

历史 v48.5 起的 DRS/best-option 辅助路径曾把它解释成“一个候选所有 roots 全局只能选一个 recovery option”，这是比论文 observation-consistency 更严格的约束。数据报告进一步说明为什么它会主要伤害 Near/Contact：Safe best-option diversity mean≈1，而 Near/Contact across train→test 大约 1.4→1.6；Near test incompatible-alias fraction≈0.209，Contact≈0.141。当前 Near 14/16、Contact 29–30/31 safe-positive deployability false-veto 与这一语义错误高度一致。

v48.46 新增 observation-class option selection/success 与对应可微 loss，但保留 historical global 路径作为对照。

## 5. v48.46 OC-SWIC 2x2

为了确保 B-A/C-A 可解释，**四个 arm 最终 calibration/certificate/closed-loop 的 evaluation semantics 全部固定为 observation_class**。训练监督是可变因素：

- A: legacy global option-witness/teacher training supervision；无 staged witness；
- B: observation-class-aligned training supervision；无 staged witness；
- C: legacy global training supervision + sequential obs→margin witness；
- D/Main: observation-class-aligned training supervision + sequential obs→margin witness。

因此 B-A=训练语义对齐效应；C-A=顺序 witness 效应；D-B-C+A=interaction。gate protocol、dataset labels、risk budget、source checkpoint、dual-ROCT、top-k=5、shared rule 全部相同。comparator 对任何 protocol/source/manifest mismatch 直接 fail-closed。

这比让 B/D 同时更换 certificate label definition 更严格，因为后者会造成 change-of-metric confounding。

## 6. CCF-A 内部 readiness 与当前缺口

以下只是项目内部 readiness bar，不是任何 CCF-A venue 官方阈值。

Near-contact：建议 verify recall 0.25–0.33、harmful-selected UCB<=0.25、precision LCB>=0.40，并在 closed-loop min-TTC p05 +0.2s 和/或 clearance +0.1m（带 CI）。当前 Precision 最好 recall 仍只有 0.111，且 AUC/false-veto 很差，因此核心缺口是 recoverable positive 被系统性 veto，而不是 proposal top-k 不存在。

Contact：建议 recall 0.20–0.30、UCB<=0.25，并争取 secondary-collision absolute reduction>=2pp、post-contact TTC +0.2s，同时改善 stable-stop/rejoin/overlap/Δv 且 yaw non-inferior。当前 certificate 最好 recall=0.05，远未达到；development safe-positive pred_adv>=0 仅 1–2/37，说明 recovery benefit 的物理 sign 仍不稳定。

Safe：当前数据结构明显更简单（artifact≈0、incompatible alias≈0、best-option diversity≈1），但 v48.45.6 全部 RC20，没有进入 paper-quality Safe closed-loop claim。最终目标仍应是统一算法的 paired non-inferiority，而不是 Safe-specific policy：NUP paired lower CI > -1%（最多预注册 -2%）、progress > -0.5%、intervention <1–2%、FRA<=1%，安全/舒适无显著退化。

## 7. 哪些技术组成应保留

与论文主线高度一致、值得保留：recovery-sufficient latent roots；post-prefix observation kernel；signed recovery margin；OC-MERO 的 observation-conditioned existential lower-tail aggregation；oracle-vs-deployable gap；CRISP/calibration 的 false-admission 控制；scene-disjoint train/dev/calibration/certificate/test protocol；regime-wise evaluation 但共享同一算法。

OCAF/dual-ROCT 目前仍可作为 downstream physical calibration/selector implementation，因为它们至少保留 candidate-relative physical evidence 和 bounded risk contract；但其复杂度已经明显超过论文主体，应避免继续堆 residual/head，否则 CCF-A 审稿时容易让真正 novelty 被工程 heuristic 淹没。

## 8. 需要修改/清理的设计

1. 历史 global-one-option DRS 语义需要被 observation-class semantics 替代；v48.46 用 2x2 验证，不直接删除 historical path。
2. v48.45 simultaneous SOWR 和 root-logit update 没有支持，停止。
3. shared source 仍保留 legacy DELTA_REGIME_EXPERTS/bucket internal geometry。为保持本轮因果归因，v48.46 不同时重建 source；若 v48.46 有明确正信号，最终投稿前应单独做 controlled source rebuild 移除 legacy bucket-conditioned policy internals。
4. 论文 Appendix 的 “Regime-conditioned recovery admission” 应在最终实验稳定后改写为 continuous deployable headroom + observation compatibility + harm envelope，而不是 contact label。

## 9. 不再重复的失败方向

遵守 ALGORITHM_CHANGELOG 的 stop rule：不增加 ROCT width/scale，不放松 harmful budgets，不 densify threshold grid，不扩大 top-k，不 positive oversampling，不继续 generic pairwise/listwise stacking，不恢复 generic harm residual/unbounded factor，不重复 v48.42 partial-pooling/rank-skip，不恢复 v48.43 POET free alias transport，不 broad fine-tune encoder，不引入 regime routing。

若 v48.46 仍失败，下一步应该检查 recovery-option taxonomy/continuous parameter coverage、margin teacher calibration、observation-class identifiability，而不是再堆 downstream capacity。

## 10. 性能瓶颈与 v48.46 加速

v48.45.6 上传结果实际串行 wall time：A≈59.2min、B≈97.3min、C≈71.6min、D≈67.6min，总约296min。v48.45.6 已把旧 A≈2.13h 降到≈59min，说明 selected-NPZ + in-process tensor cache 有效。剩余主要时间是 factor/witness training，同时 train/val NPZ 在每个 variant/stage 仍重复 materialize；日志中一次 materialization 约90–250s。

v48.46 新增 persistent decoded-tensor cache：跨进程/跨 witness/factor stage 共享最终 tensor，cache key 只绑定 manifest/path、feature geometry 和真正改变 tensor 的设置，model head/optimizer/ROCT/option semantics 不再造成无意义 cache miss。fcntl file lock + atomic replace 支持 A/B 与 Balanced/Precision 并发 warm cache。

新的 2GPU launcher 先 A+B 并发（GPU0/GPU1），再 C+D 并发。每个 ablation 内 Balanced/Precision 默认可在该 arm 的同一 GPU 并发；若观察到 OOM/吞吐下降，设置 V4846_VARIANT_MODE=serial，此时仍保持两个 ablation 分别在两张 GPU 同步运行。

仅从 v48.45.6 的 wall time估算，不考虑新增 persistent cache，pairing 理论 wall 上界约 max(A,B)+max(C,D)=97.3+71.6≈168.9min，相比296min约减少43%。真实 A30 结果应以新 telemetry 为准。

launcher 每30s记录 GPU util/memory/power 与 host load/MemAvailable，便于下一轮继续判断是否 GPU under-utilization、CPU解码或共享存储抖动。`tools/summarize_v48_46_runtime_telemetry.py` 会在运行后给出每张卡的平均/中位/P10/P90 utilization、低利用率比例、峰值显存占比和最低 host MemAvailable。persistent cache 最终采用 schema-v3 的 `weights_only + mmap` 优先读取，使同一节点上的多个 trainer 共享 OS page cache；缓存损坏则在 flock 内自动重建。

## 11. 工程验证

最终 v48.46 本地验证：v48.42–v48.46 + v48.45 engineering/protocol 组合测试 70/70；v48.37–41 29/29；v48.36 OCAF 14 passed/1 skipped；v48.36 transfer/terminal/idempotence 18/18；去重后合计131 passed/1 skipped。compileall PASS，103/103 shell scripts bash -n PASS，v48.45 nounset-local regression PASS。新增 persistent-cache mmap/corrupt-rebuild 与 explicit evaluation-semantics 回归均通过。

本环境没有用户的 A30/WOMD 数据，因此不声称已经得到真实 v48.46 GPU/certificate 结果。
