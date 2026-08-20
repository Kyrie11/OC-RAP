# OC-RAP v48.50 authoritative 2×2 归因与 v48.51 DCP-DRFC-BCDE 设计

日期：2026-08-18

## 0. 结论先行

本轮修复后重新得到的 v48.50 A/B/C/D 四臂可以做严格算法归因：四臂 authoritative exit code 都是 RC=20，`pipeline_valid=true`，certificate 与 Natural gate 均真实执行，attribution contract valid，source/calibration/protocol identity 一致，`test_roots_read=false`。因此 RC20 是**算法 Natural-gate failure**，不是上一轮的工程 RC30。

最重要的机制结论不是“exact coordinate 无效”，而是：

> **Exactness 必须负责 material decision sign；smooth boundary geometry 必须负责 hard-equivalence class 内的 local ordering。把 hard/exact coordinate 整体替换 smooth coordinate，会损失 Near 的有效排序；把 hard coordinate 同时拿来做连续 magnitude 回归，又会造成 safe-positive veto 与 ranking collapse。**

因此下一版不继续增加 capacity/router/threshold，而把论文主线从一般的 Decision-Equivalent Certificate Transport 推进为 **Boundary-Complete Decision Equivalence (BC-DE)**：同一个 observation-consistent recovery primitive 同时满足 material-sign preservation 与 boundary-local order preservation。

v48.51 已在代码中落地为严格 2×2：

- A：v48.50-A reference = old DRFC + smooth NAP；
- B：A + BC-FC（Boundary-Complete Frontier Calibration）；
- C：A + BC-NAP（Boundary-Complete Native Advantage Preservation）；
- D/Main：BC-FC + BC-NAP。

所有 arm 都继续共享同一个 Safe/Near/Contact policy primitive；没有 regime ID/router、regime-specific threshold/loss/budget、额外 proposal top-k、learned admission residual 或新 head。

---

## 1. v48.50 结果是否可靠、能否归因

本轮四臂都是：

- authoritative RC=20；
- `pipeline_valid=true`；
- certificate executed；
- Natural gate evaluated；
- attribution identity / factor contract valid；
- source checkpoint、protocol seal、calibration manifests 对齐；
- `strategy_regime_conditioning=false`；
- `test_roots_read=false`。

因此本轮可以解释 B−A、C−A 与 D−B−C+A。RC20 只表示当前算法没有通过 preregistered Natural gate，不表示工程失败。

v48.50 因子定义：

| Arm | Upstream DEFC | Downstream Exact NAP | 解释 |
|---|---:|---:|---|
| A | 0 | 0 | old DRFC + smooth NAP reference |
| B | 1 | 0 | 只测试 forward-exact/backward-smooth frontier calibration |
| C | 0 | 1 | 只测试 exact hard-DRS native advantage transport |
| D | 1 | 1 | 两者组合 |

---

## 2. 核心结果

### 2.1 Precision certificate（主因果读数）

| Arm | Near recall | Near precision | Near harmful UCB90 | Contact recall | Contact precision | Contact harmful UCB90 |
|---|---:|---:|---:|---:|---:|---:|
| A | **0.222** | **0.0488** | 0.1119 | 0 | 0 | 0.5675 |
| B / +DEFC | **0.222** | 0.0256 | **0.0419** | **0.050** | 0.0208 | 0.3815 |
| C / +Exact NAP | 0 | 0 | 0.0640 | 0 | 0 | 0.3616 |
| D / both | 0.111 | 0.0222 | **0.0352** | **0.050** | 0.0192 | **0.2924** |

Balanced certificate 也呈现同一总体趋势：

| Arm | Near recall | Near harmful UCB90 | Contact recall | Contact harmful UCB90 |
|---|---:|---:|---:|---:|
| A | 0.111 | 0.1085 | 0 | 0.6583 |
| B | 0.111 | **0.0258** | **0.050** | 0.3520 |
| C | 0 | 0.0536 | 0 | 0.2673 |
| D | **0.222** | 0.0891 | 0 | **0.2466** |

Balanced/Precision 的绝对数不同，但因果结论一致：DEFC 能降低一部分 harmful admission；Exact-NAP-only 会损失 Near positive recall；组合并没有形成 clean synergy。

### 2.2 Development 是当前真正的 gate bottleneck

Precision development：

- A Near：recall 0，positive selected 0；
- B Near：recall **0.375**、3 positives / 34 selected、raw precision **0.0882**、LCB90 **0.0434**；
- C Near：recall 0；
- D Near：recall 0.125、1 positive / 21 selected、precision 0.0476、LCB90 0.0144；
- Contact：A/B/C/D development precision 都是 **0**。

协议要求 Near fit `min_precision_lcb=0.50`、verify `0.40`；Contact 同样是 fit 0.50 / verify 0.40。于是当前直接失败层不是“只差一点 recall”，而是 **benefit precision / centering 与 shared admission rule 仍相距一个数量级**。

---

## 3. 严格 2×2 算法归因

## 3.1 B−A：DEFC 的 principle 有效，但 v48.50 实现不是 clean win

DEFC 的正向证据：

1. Near development 从完全没有正例变为 recall 0.375、joint semantic eligible 4/19；
2. Near certificate recall 保持 0.222 的同时 harmful UCB90 从 0.112 降到 0.042；
3. Contact certificate 首次出现 recall 0.05；
4. Contact harmful UCB90 从 0.567 降到 0.382。

这说明 **部署时真正消费的 exact boundary 必须进入 upstream calibration**。旧 smooth-only frontier calibration 确实存在 train/inference decision-coordinate gap。

但负向证据同样明确：

- Near candidate safe-positive AUC 从 0.527 降到 0.431；
- Near precision 从 0.0488 降到 0.0256；
- Contact candidate safe-positive AUC 0.681 → 0.617；
- Contact proposal safe-positive AUC 0.627 → 0.595；
- Near certificate DRS safe-positive false-veto 变为 7/16；
- Contact development DRS safe-positive false-veto 变为 11/37。

DEFC 把 harmful DRS false-safe 压下来的同时，开始把真正 safe-positive 也量化到错误一侧。因此其问题不是 exact sign supervision 本身，而是 **hard DRS 是量化坐标，却被要求同时拟合 continuous magnitude/order**。

归因结论：

- **保留：exact deployed sign supervision；**
- **拒绝：hard-coordinate magnitude regression。**

## 3.2 C−A：Exact NAP full overwrite 明确 reject

Exact NAP 的主要结果：

- Near certificate recall 0.222 → **0**；
- Contact candidate safe-positive AUC 0.681 → **0.579**；
- Contact proposal safe-positive AUC 0.627 → **0.464**；
- Contact certificate recall 仍为 0。

这证明 v48.49-C / v48.50-A 的 smooth NAP 并不是一个应该被“更精确 hard value”简单替换的错误 proxy。Smooth DRS 记录了 q 距离 hard threshold 的 boundary depth，给同一个 hard equivalence class 内的候选提供了真实有用的 local ordering/tie resolution。

另一方面，Exact NAP 也暴露一个值得吸收的机制：

- C Contact development final joint sign 从 A 的 0/37 提升到 3/37；
- D 到 4/37；
- B 的 Contact development exact native nonnegative fraction 已达到 7/37，但 smooth final sign 仍是 0。

因此 hard/exact coordinate 含有 **material sign information**，只是它的 ranking resolution 不足。

归因结论：

- **拒绝 Exact NAP 作为 full downstream replacement；**
- **保留 hard exact coordinate 作为 material-sign anchor；**
- **继续保留 smooth NAP 作为 within-boundary ordering。**

## 3.3 D−B−C+A：没有 clean positive interaction

Near development joint sign：A=0/19、B=4/19、C=0/19、D=1/19。也就是说 B 的收益大部分在 D 中被 Exact NAP 抵消，interaction 为负。

Contact D 虽达到 4/37 development joint sign，并把 certificate UCB90 降到 0.292，但仍：

- recall 只有 0.05；
- UCB90 仍高于 0.25；
- development positive recall 仍为 0；
- candidate/proposal safe-positive ranking 均不如 A。

所以不能把 D 当下一版 reference。v48.51 应回到 A 作为共同 reference，只吸收 B/C 中被证实有价值的**局部原则**。

---

## 4. 当前主要瓶颈是什么

### 4.1 首要机制瓶颈：boundary quantization 与 local ordering 冲突

Hard DRS 对物理部署 sign 是正确 coordinate，但会把同一 hard state 内的 q-depth 压平。Smooth DRS 能分辨 q-depth，但如果让它独自决定最终 admission sign，又会出现 Contact exact-positive 被 smooth transport erase。

当前 v48.50 没有显式分工，于是：

- smooth-only：排序好，但 centering/sign 不够；
- exact-only：sign 有信息，但排序/recall 塌陷；
- exact magnitude calibration：减少 harmful false-safe，却产生 safe-positive veto。

这是本轮最有论文价值的机制发现。

### 4.2 直接 Natural-gate 瓶颈：benefit precision / centering

Near 最好的 B development raw precision 只有 0.088、LCB90 0.043；协议门槛是 0.50/0.40。Contact development 四臂 precision 都为 0。

因此下一步不能只追 recall，也不能通过放宽 threshold 换 recall；首先要让“跨过 benefit materiality boundary”的候选真正集中到 safe-positive 上。

### 4.3 Contact 是最大的 regime-level empirical gap

Contact 当前：

- certificate best recall 0.05；
- Main Precision harmful UCB90 0.292；
- development final joint sign best 4/37；
- final development positive recall 仍 0。

A 的 candidate/proposal AUC 又已经证明存在一定 ranking signal，所以当前更像 **sign/centering + certificate decomposition mismatch**，而不是完全没有学习信号。

### 4.4 DEP/GAP 仍有 sensitivity/specificity trade-off

Contact 的 deployability 与 gap-quality 对 safe-positive 的 false-veto 依然高；不同 arm 只能在 harmful false-safe 与 safe-positive false-veto 之间交换，没有形成统一 Pareto improvement。

这意味着如果 v48.51 仍失败，下一层应该审 teacher PCD correctness、DEP/GAP normalization 与 root probability calibration，而不是继续叠 admission module。

### 4.5 Proposal coverage 不是当前主因

所有 arm development 都 `proposal_oracle_feasible=true`，Near 有 9 个、Contact 有 20 个 safe-positive proposal groups。继续增大 top-k/candidate family/macro width 不针对当前 failure layer，也重复历史 stop signal。

---

## 5. 哪些设计应加入主算法，哪些应停止

应加入/保留：

1. v48.49-C / v48.50-A 的 **smooth NAP ordering**；这是 Near recall 和 Contact ranking 的重要来源。
2. v48.50-B 的 **exact deployed boundary supervision principle**；它能改善 Near sign 与 harmful UCB。
3. v48.50-C/D 暴露的 **hard exact material sign**；但只作为 sign anchor，而不是 full overwrite。
4. shared rule、observation-class execution、test-root seal、fail-closed Natural gate 等 protocol machinery。

应停止：

1. Exact-NAP full overwrite；
2. 旧 DEFC 的 hard-coordinate magnitude regression；
3. MC-NCP 及其 tolerance 微调；
4. threshold-grid densification；
5. proposal top-k/candidate/macro width 扩张；
6. aggressive positive oversampling / hardest-negative distribution distortion；
7. generic pairwise/listwise stacking；
8. learned admission residual / extra policy head；
9. one-sided safe-positive component penalty；
10. frontier-tanh；
11. POET/SOWR/DWOK/broad encoder fine-tuning 等 changelog 已有 stop signal 的路线；
12. 任何 Safe/Near/Contact-conditioned policy/router/threshold/budget。

---

## 6. 下一版：v48.51 DCP-DRFC-BCDE

### 6.1 CCF-A 主线

建议论文方法线收敛成：

> **Observation-Consistent, Boundary-Complete Decision-Sufficient Recoverability**

链条：

`recovery-sufficient roots -> observation-consistent legal recovery -> OC-MERO certificate -> boundary-complete decision-equivalent transport -> non-compensatory calibrated admission`

核心不是“又加两个 module”，而是对 decision equivalence 给出更严格定义：

- **Material sign preservation**：当原生 exact certificate 已明确落在 material boundary 的一侧，learned/smooth transport 不得翻转其最终 decision sign；
- **Equivalence-class order preservation**：当 exact/hard certificate 因量化而无法区分候选时，transport 必须保留与同一 zero crossing 一致的 boundary-local ordering。

这两个条件共同构成 **Boundary-Complete Decision Equivalence**。

### 6.2 BC-FC：Boundary-Complete Frontier Calibration

BC-FC 不增加 head，只更新现有 `margin_head`。

它把 v48.50 DEFC 拆为两条职责通道：

**Sign channel**

- model-predicted root weights；
- hard `q_best >= gamma` DRS，forward exact / backward STE；
- exact `sigmoid(R_dep)`；
- exact `exp(-relu(gap))`；
- exact PCD；
- 只优化 balanced sign BCE。

**Order channel**

- smooth boundary DRS；
- 相同 continuous DEP/GAP；
- smooth PCD；
- symmetric SmoothL1 magnitude/order regression。

关键区别：不再要求 discontinuous hard DRS 的“幅值”拟合 teacher，只要求其 decision side 正确。这样保留 B 的 exact-boundary 收益，同时避免把 hard quantization 当连续 regression target。

### 6.3 BC-NAP：Boundary-Complete Native Advantage Preservation

BC-NAP parameter-free。保持现有 smooth NAP，并让 hard exact value 只负责 material sign。

定义 candidate-relative：

`d_exact = V_exact(candidate) - V_exact(nominal)`

`d_smooth = V_smooth(candidate) - V_smooth(nominal)`

继续复用现有 `positive_gain g = 0.015`，不新增 threshold：

- `d_exact >= g`：exact 已 materially positive，使用 `max(d_exact, d_smooth)`，smooth 不能把正号抹掉；
- `d_exact <= -g`：exact 已 materially negative，使用 `min(d_exact, d_smooth)`，smooth 不能把负号翻正；
- `|d_exact| < g`：hard certificate 处于 material-equivalence band，使用 `d_smooth` 做 local ordering；
- 最后 `benefit_margin = d_BC - g`。

这正对 v48.50 的两个观测：Near 需要 smooth ranking；Contact 有 latent exact-positive sign 不能再被 smooth erase。

### 6.4 严格 2×2

| Arm | BC-FC | BC-NAP | 意义 |
|---|---:|---:|---|
| A | 0 | 0 | v48.50-A reference |
| B | 1 | 0 | upstream sign/order decomposition |
| C | 0 | 1 | downstream material-sign / deadband-order transport |
| D/Main | 1 | 1 | interaction |

旧 v48.50 DEFC 与 Exact-only NAP 在四臂都关闭，避免把已知失败机制混入新主效应。

---

## 7. 预注册判断标准

### Near

- D Precision certificate recall >= 0.25；
- harmful-selected UCB90 <= 0.25；
- development joint sign 至少不低于 v48.50-B 的 4/19；
- candidate safe-positive ranking 不能重现 v48.50-B 的明显 collapse。

### Contact

- development joint sign >= 6/37；
- Precision certificate recall >= 0.10；
- harmful-selected UCB90 <= 0.25。

这只是进入下一证据层的 mechanism screen，不是最终投稿性能目标。

### Safe

- standard calibration valid 是必要条件；
- 只有 D/Main authoritative RC=0，才允许执行 scene-disjoint paired Safe non-inferiority + stress/closed-loop；
- RC20 必须阻断 test/closed-loop，避免用最终 test 反向调算法。

### Stop rules

- 若 B/BC-FC 重复 DRS safe-positive veto/ranking collapse：停止调 loss temperature/weight，转入 predicted-root probability calibration 和 teacher/native component correctness audit；
- 若 C/BC-NAP 不能同时保留 Near ranking 与提高 Contact sign：停止 value/admission transport 变体，转入 teacher PCD decomposition、DEP/GAP normalization、root/recovery-witness calibration；
- 若 B/C 都正向但 D 强负交互：检查共享 margin-head gradient conflict，不加 router/regime conditioning。

---

## 8. 距离 CCF-A 的差距

### 8.1 方法/故事性

方法故事已经比 v48.48–v48.50 更聚焦：Observation Consistency + Boundary-Complete Decision Equivalence 是一个统一原则，可以解释为什么 safe/near/contact 都使用同一个 recovery primitive，而不是三套 regime-conditioned policy。

要达到 strong CCF-A 叙事，论文应进一步形式化：

1. 定义 material decision equivalence；
2. 定义 hard equivalence class；
3. 给出 BC transport 的 sign-preservation proposition；
4. 给出 equivalence band 内 smooth-order preservation 条件；
5. 把 BC-FC/BC-NAP 写成该原则的 training/deployment realization，而不是独立 module 名词堆叠。

### 8.2 Empirical gap 仍然明显

当前四臂全部 RC20，说明经验闭环远未完成：

- Near best recall 只有 0.222；development precision LCB 远低于 gate；
- Contact best recall 0.05，development positive recall 仍为 0；
- Safe 只有 standard calibration evidence，authoritative paired non-inferiority/closed-loop 尚未解锁；
- test roots 按正确 protocol 尚未读取。

另外，从本轮上传材料本身还不能证明最终论文已经具备 multi-seed confidence、完整 SOTA/baseline comparison 和三 regime 的 closed-loop non-inferiority；这些必须在算法通过 Natural gate 后补齐，而不是现在提前读 test。

因此“距 CCF-A 的差距”当前主要在 **empirical closure**，其次才是 method formalization；不是再堆更多 architecture novelty。

---

## 9. 代码落地与工程审计

已落地：

- `boundary_complete_frontier_calibration_loss`；
- BC-NAP model flag 与 candidate-relative transport；
- checkpoint metadata / inference reconstruction；
- model contract；
- v48.47 nested witness stage isolation；
- factor-cache identity；
- v48.51 A/B/C/D runner；
- two-GPU launcher；
- v48.51 strict 2×2 comparator；
- D/Main RC0-only post-gate wrapper；
- v48.51 design contract；
- v48.51 regression tests；
- 更新后的 `ALGORITHM_CHANGELOG.md` / `ALGORITHM_CHANGELOG_V48.md`。

工程验证：

- v48.51 新机制定向测试：4 passed；
- v48.47–v48.51 algorithm-focused：40 passed；
- model/training/engineering/stage-transfer：32 passed；
- terminal-state/idempotent/OCAF：26 passed、1 skipped；
- v48.45 engineering/stage-isolation：25 passed；
- 总 targeted regression：**123 passed、1 skipped**；
- `python -m compileall -q src tools tests` PASS；
- **118 个 shell scripts** 全部 `bash -n` PASS；
- 根执行指令 `bash -n` PASS；
- synthetic post-gate：RC20 正确阻断；RC0 + 合法 D factor 正确授权；
- synthetic v48.51 comparator：strict 2×2 attribution contract valid。

当前环境没有 `/data0/...` 数据和 A30，因此不能在这里声称真实 v48.51 end-to-end 训练 / Natural gate 已通过；需要在你的机器上执行下一轮。

---

## 10. 下一轮执行

把代码放到：

`/home/senzeyu2/code/OC-RAP-v48.51-DCP-DRFC-BCDE`

然后：

```bash
cd /home/senzeyu2/code/OC-RAP-v48.51-DCP-DRFC-BCDE
bash OC-RAP-v48.51-DCP-DRFC-BCDE-two-GPU-run-commands-ZH.txt
```

输出目录：

- `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_51_dcp_drfc_bcde_ablation_A`
- `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_51_dcp_drfc_bcde_ablation_B`
- `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_51_dcp_drfc_bcde_ablation_C`
- `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_51_dcp_drfc_bcde_main`
- `/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.51-DCP-DRFC-BCDE-2x2-audit.json`

正确读取顺序仍是 **B−A → C−A → D−B−C+A**；先判断 BC-FC 是否解决 exact sign 与 ranking 的冲突，再判断 BC-NAP 是否能保留 Near smooth ranking 并释放 Contact exact material sign，最后才解释 D/Main。
