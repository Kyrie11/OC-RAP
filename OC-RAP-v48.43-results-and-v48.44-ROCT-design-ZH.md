# OC-RAP v48.43 POET 结果归因与 v48.44 ROCT 设计

日期：2026-08-10

## 0. 结论先行

本轮四个 v48.43 A/B/C/D 结果均为 `authoritative_exit_code=20`、`pipeline_valid=true`，因此可以做算法归因。

- **B 不吸收**：没有真正旋转 Near deployability frontier。相对 A，Near deployability safe-positive false-veto 仍为 **16/18**，conditional harmful-vs-safe-positive AUC 反而 **0.416 -> 0.409**，certificate harmful-selected UCB90 **0.546 -> 0.640**。
- **C 不吸收**：没有真正提升 Contact physical benefit capture。Contact candidate safe-positive AUC 仅 **0.544 -> 0.556**，proposal safe-positive AUC **0.555 -> 0.526**，certificate recall 仍 **0.10**，development safe-positive 中 `pred_adv>=0` 仍为 **0/37**。
- **D 不吸收为主算法**：它确实暴露了有用结构信号，但 shared rule 完全崩塌。Contact candidate safe-positive AUC 到 **0.578**；deployability false-veto **26/31 -> 17/31**；DRS harmful-vs-safe-positive AUC **0.554 -> 0.706**。然而 Contact deployability harmful false-safe **0.148 -> 0.255**，Near deployability false-veto恶化为 **17/18**，certificate Contact recall 变 **0**，development Near/Contact 都 `selected=0, positive_selected=0`。

因此 v48.43 **没有一个 POET 参数模块值得原样吸收**。值得吸收的只是一个更高层次的实验事实：**candidate-specific post-prefix structural evidence 确实含有信息，但“只描述 observation alias + 自由注入整个 task context”不是正确的参数化方式。**

下一版应从论文原始定义继续往下走：不是问“roots 是否 alias”，而是问 **observation-equivalent roots 是否具有 shared compatible recovery option**。这就是 v48.44 ROCT。

---

## 1. 论文与当前代码的结构对齐

论文正文把 OC-RAP 的核心定义得非常清楚（`post-collision.tex`）：

- 约第 162 行：candidate prefix 之后先形成 post-prefix observation equivalence classes，再判断每个 class 是否存在 shared recovery policy；
- 约第 223--239 行：若两个 latent roots 对 deployed system 不可区分，则 recovery decision 必须是 post-prefix observation 的函数，**同一 observation class 中所有 roots 必须共享 compatible recovery option**；
- 约第 290--328 行：oracle recoverability 先对每个 hidden root 单独 `max option`；deployable recoverability 必须先按 observation consistency 聚合再选 shared option；当不可区分 roots 需要 incompatible recovery options 时，oracle/deployable gap 严格增大，这正是 false recoverability admission 的来源。

v48.43 POET 只把 root entropy / alias mass / peak alias / max-root-probability 注入 evidence adapter。它回答的是“候选改变了多少观察歧义”，但没有直接回答论文真正的判别命题：**这些被 alias 的 roots 到底能不能共享 recovery option**。

这解释了为什么 POET D 在 Contact 出现局部改善却又产生跨坐标污染：alias 本身既可能是 benign ambiguity，也可能是 incompatible-recovery ambiguity。

---

## 2. v48.43 A/B/C/D 量化归因

### 2.1 Near-contact

| Arm | candidate safe+ AUC | proposal safe+ AUC | recall | harmful UCB90 | deployability AUC | deployability false-veto | deployability harmful false-safe | dev positive selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.652 | 0.622 | 0.111 | 0.546 | 0.416 | 16/18 | 0.162 | 0 |
| B | 0.640 | 0.586 | 0.111 | 0.640 | 0.409 | 16/18 | 0.131 | 0 |
| C | 0.637 | 0.586 | 0.111 | 0.556 | 0.407 | 16/18 | 0.131 | 0 |
| D | 0.677 | 0.629 | 0.000 | 1.000* | 0.393 | 17/18 | 0.316 | 0 |

`*` D 的 selected=0，UCB=1 是空选择的保守约定，不能解读为真实污染率 100%。

**B 的答案：没有真正旋转 frontier。** false-veto 一点没动、AUC 下降；harmful false-safe 虽略降，但最后 harmful-selected UCB 明显更差。这是 score/trade-off shift，不是 safe-positive 与 harmful 的条件可分性提升。

Near 当前不是缺 generic harm capacity；最主要错误仍是 deployability 对 safe-positive 的系统性过度 veto。

### 2.2 Contact

| Arm | candidate safe+ AUC | proposal safe+ AUC | recall | harmful UCB90 | deployability AUC | deployability false-veto | deployability harmful false-safe | DRS AUC | dev positive selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.544 | 0.555 | 0.100 | 0.522 | 0.595 | 26/31 | 0.148 | 0.554 | 0 |
| B | 0.558 | 0.527 | 0.100 | 0.528 | 0.630 | 29/31 | 0.079 | 0.589 | 0 |
| C | 0.556 | 0.526 | 0.100 | 0.510 | 0.625 | 29/31 | 0.074 | 0.584 | 0 |
| D | 0.578 | 0.564 | 0.000 | 0.732 | 0.635 | 17/31 | 0.255 | 0.706 | 0 |

**C 的答案：没有真正提升 Contact physical benefit capture。** candidate AUC 小涨约 1.2pp，但 proposal AUC下降、recall 不变、development 正例仍为 0。

**D 的答案：没有第一次让 shared rule 两边同时抓正例。** 恰恰相反，D 的 nearest shared rule 在 Near/Contact 都 `selected=0`。但 D 的 Contact deployability false-veto 和 DRS AUC 是本轮唯一值得继续追的局部信号：说明 post-prefix candidate structure 与真实 frontier 有关系，只是当前注入方式破坏了全局单调/尺度一致性。

---

## 3. 最关键的新诊断：Contact physical-advantage sign collapse

我额外对 `candidates/precision/calibration/dev_diagnostic_*_v48.proposal_rows.jsonl` 做了 development proposal-row 诊断，safe-positive 定义仍用本项目的 `teacher_adv > 0.015 && !teacher_harmful`。

### Contact safe-positive (`n=37`)

| Arm | median pred_adv | pred_adv >= 0 | joint semantic eligible (`opp>=.5 & harm<=.5 & pred_adv>=0`) |
|---|---:|---:|---:|
| A | -0.122 | 0/37 | 0/37 |
| B | -0.127 | 0/37 | 0/37 |
| C | -0.122 | 0/37 | 0/37 |
| D | -0.180 | 0/37 | 0/37 |

这比“RC=20”更重要。shared rule 的 score semantic domain 本来就从 `score_threshold >= 0` 开始，因此当前 Contact safe-positive 在算法分数几何上**数学上不可选**。继续 densify threshold grid 没有意义；允许负 score threshold 又会破坏目前 physical-advantage 语义和论文叙事。

所以 Contact 首要瓶颈不是校准器搜索，而是 **learned physical advantage / joint reserve 的方向性错误**。

Near 的 safe-positive 还有少量 `pred_adv>=0`（A: 4/17），但仍被 deployability harm veto 大量挡掉。因此两者实际上可以由同一个统一缺陷解释：**模型没有把 observation-consistent recovery-option compatibility 变成正确方向的 candidate-relative physical evidence。**

---

## 4. CCF-A 投稿内部目标与当前差距

以下数值来自代码包根目录 `OC-RAP-CCF-A-targets.csv`，它们是项目内部 submission-readiness bar，**不是任何 CCF-A venue 的官方硬阈值**。

### Near-contact

内部目标：

- verify precision LCB >= **0.40**（fit >=0.50）；
- recall >= **0.25--0.33**；
- harmful-selected UCB <= **0.25**（fit <=0.22）；
- closed-loop min-TTC p05/LCB >=约 **+0.2 s** 和/或 clearance >= **+0.1 m**；
- collision/hard-brake 降低，NUP loss <=2--3%。

当前最好 recall 仍只有 **0.111**，A 的 harmful-selected UCB90 **0.546**。投稿级目标意味着 roughly：**safe-positive capture 至少提高 2--3 倍，同时 harmful selection 风险约减半**。这显然不是调一个 threshold 就能解决。

### Contact

内部目标：

- recall >= **0.20--0.30**；
- harmful-selected UCB <= **0.25**（fit <=0.22）；
- secondary collision/re-contact 绝对下降 >=约 **2 pp**；
- post-contact TTC >=约 **+0.2 s**；
- overlap/delta-v 降低；stable-stop/route-rejoin 改善且 yaw non-inferior。

当前 A/C recall **0.10**，D 更降为 0；harmful UCB90 大约 **0.51--0.73**。需要至少把正例 capture 翻倍并将 harmful risk 大幅压低。

### 对 CCF-A 论文的非数值要求

在达到上述内部 bar 后，仍建议至少保证：

1. **统一机制**：Safe/Near/Contact 只作为 evaluation slices，不出现三套 policy/state machine；
2. **可解释的新原理**：核心增益应能回到 observation-consistent recoverability / shared-option compatibility，而不是工程堆叠；
3. **强消融**：A/B/C/D 可以明确证明 structural statistic 与 semantic-local injection 各自作用；
4. **统计可信**：最终 test/closed-loop 应使用 controller 允许的 held-out protocol、多 seed/CI，并避免在 RC=20 阶段读取 test；
5. **强 baseline**：论文最终需要与合理 learned planner / safety shield / recovery-aware baselines 对齐比较，而不是只比较内部版本。

---

## 5. v48.44：ROCT — Recovery-Option Compatibility Transport

### 5.1 核心思想

POET 的错误不是“候选后观测结构无用”，而是它只建模 `alias`，没有建模 `alias × recovery-option compatibility`。

v48.44 对每个 candidate prefix `a`，复用已经训练好的 frozen：

- latent-root probabilities `p_k`；
- post-prefix observation embeddings / kernel `C_ij(a)`；
- root-option signed recovery margins `m_{k,l}(a)`；
- 同一份 OC-MERO operator。

形成四维 bounded structural signature：

1. `dep_unit = 0.5*(tanh(R_dep)+1)`；
2. `gap_unit = tanh(relu(R_orc-R_dep))`；
3. `conflict_pressure`：对 observation-aliased root pair `(i,j)`，计算最优 option 的 `min(success_i_l, success_j_l)`；若没有共同高支持 option，则 conflict 高，再乘 alias/root probability mass；
4. `shared_feasible_mass`：按 root probability 聚合的 observation-consistent feasible recovery mass。

随后只使用 candidate 相对 nominal 的变化：

`Delta psi_ROCT(a) = psi_ROCT(a) - psi_ROCT(a0)`。

### 5.2 为什么这比 POET 更符合论文

它把论文中“两个 roots 看起来一样”与“两个 roots 必须共享同一个 recovery option”合并成同一个可学习 evidence。换句话说，ROCT 能区分：

- **benign alias**：两个 roots 不可区分，但存在同一 brake/yield/stabilize option 都可行；
- **harmful alias**：两个 roots 不可区分，但最优 recovery option 相互冲突。

只有后者才应该强烈降低 deployability / joint reserve。

### 5.3 Semantically-local injection

为了避免 v48.43 D 的 cross-coordinate negative transfer，v48.44 不再把 structural vector 注入整个 benefit/harm OCAF context。

- **Benefit-side ROCT**：只对 unified benefit logit 添加 bounded correction；
- **Safety-side ROCT**：只对 component index 1（deployability）添加 bounded correction；
- DRS、gap、hard-rule、harm-proxy、shared OCAF bridge 不被 ROCT safety correction 旋转。

adapter 为 4->1、bias-free、zero-init，teacher 全 detach；correction 用 `tanh` bounded。v48.44 实验共享 `ROCT_SCALE=3.0`，对应 `tau_b=0.05` 时最大 physical benefit-margin correction 0.15，覆盖当前 Contact safe-positive median sign deficit 的量级，同时不走 v48.39 的 unbounded-residual 路径。

### 5.4 为什么仍然是三 regime 统一算法

ROCT 的输入只有 candidate-relative physics / latent roots / observation kernel / recovery-option margins。没有 regime id，没有 `if contact`，没有 Near/Contact 专属 head，没有 regime threshold。Safe/Near/Contact 仍只是 dataset/report slice。

因此它提升的是论文的统一性质，而不是把问题重新拆成三个状态机。

---

## 6. 下一轮 2x2 设计

| Arm | Deployability ROCT | Benefit ROCT | 因果问题 |
|---|---:|---:|---|
| A | × | × | retained v48.43-A reference |
| B | ✓ | × | shared-option compatibility 是否真正修 Near deployability frontier |
| C | × | ✓ | 是否修 Contact physical-benefit sign / capture |
| D/Main | ✓ | ✓ | 两者是否互补，并让一个 shared rule 同时在 Near/Contact 选到 safe-positive |

所有 arm 强制关闭：v48.42 partial-pooling、rank skip；v48.43 POET；unbounded factors；full factorization；regime routing。保留 dual OCAF、bounded HAF/component veto、support reliability、joint reserve、top-k=5 和同一个 shared rule。

### 预注册判定

**B**：Near false-veto 需实质下降（strong go <=12/18），deployability AUC 同时上升（strong go >=0.56），且 harmful false-safe/UCB 不能明显变坏。只降 false-veto、不升 AUC = calibration shift，失败。

**C**：最先看 Contact development safe-positive `pred_adv>=0` 是否从 0/37 变成非零（strong go >=25%）；再要求 candidate/proposal benefit AUC 大约 >=+0.05、certificate recall 向 >=0.20 移动，harmful UCB 不恶化。

**D**：Near 和 Contact development **都必须 `positive_selected>0`**，且使用同一 rule；必须同时保留 B 和 C 的结构性收益，不能靠放宽 harmful budget。constraint deficit 数字下降但 positive=0 仍算失败。

如果 v48.44 仍失败，下一步**不要**再加 ROCT width/scale，不再调 dense grid，也不放松 harm budget；应检查 frozen recovery-option set / margin teacher / observation kernel 的 calibration 与 option coverage。

---

## 7. 历史无效修改：本轮明确避免

根据 `ALGORITHM_CHANGELOG.md`，本轮没有重复以下已失败路径：

- threshold-grid densification；
- proposal top-k expansion；
- aggressive positive oversampling / hardest-negative population distortion；
- generic pairwise/listwise ranking stacking；
- learned admission residual；
- v48.38 one-sided tail loss；
- v48.39 unbounded benefit/harm；
- v48.40 `frontier_tanh`；
- v48.41 full component factorization；
- v48.42 partial-pooling harm residual；
- v48.42 bounded rank-benefit skip；
- v48.43 alias-only free dual-context POET amplification；
- 任何 regime-conditioned routing/threshold/policy。

---

## 8. 工程落地与复核

已修改/新增：

- `src/ocrap/models/ocrap.py`：ROCT signature、candidate-relative transform、bounded benefit/deployability-local correction；
- `src/ocrap/cli/train.py`：root/option validity masks + ROCT checkpoint/train-summary fields；
- `src/ocrap/models/inference.py`：checkpoint/config materialization + validity masks；
- training/adaptation shell：ROCT config、factor-cache identity、trainable prefix、stage architecture；
- model/training/stage-transfer contracts：ROCT fail-closed binding；
- `tests/test_v48_44_roct.py`：零初始化 identity、nominal-zero、bounded signature、component locality、gradient detach、script/contract tests；
- v48.44 A/B/C/D launchers + comparator/frontier diagnostics。

当前复核：

- `python -m compileall -q src tools tests`：PASS；
- focused v48.36/37/38/39/40/41/42/43/44 tests：**76 passed, 1 skipped**；
- v48.44 dedicated tests：**7 passed**；
- **94/94 shell scripts** `bash -n` PASS；
- CPU full-forward + backward ROCT smoke：finite，benefit/deployability adapter 均有非零 gradient；
- structural teacher (`root_logit_head`, `obs_embed_head`, `margin_head`) 在 direct evidence adaptation 中 detach，不被 ROCT sparse loss 旋转。

没有把“完整历史 pytest 全通过”写成结论；仓库历史 suite 本来就有旧版本资源缺失问题，本轮只对当前相关 contract/matrix 做 fail-closed 检查。

---

## 9. 论文侧需要同步修正的一个重要点

`post-collision.tex` 约第 684 行仍有 `Regime-conditioned recovery admission`，写明第二条 protective certificate “used only in low-headroom contact regimes”。这与当前最有价值的论文主线冲突：**同一个 observation-consistent recoverability principle 应跨 Safe/Near/Contact，而不是 Contact 特判。**

建议在算法达到 submission-ready 后重写该段，使 protective semantics 由通用的 headroom / option-compatibility 连续性质触发，而不是由 regime label 触发。上一轮分析也已指出这一点。

另外，本轮实际挂载的上传文件列表里没有单独的 `reports.zip`；代码包根目录仍包含 `OC-RAP-dataset-report-summary.csv`。其中 `supports_womd_primary_claim=false` 仍是最终投稿时需要清理的 provenance/claim 风险。但按本轮要求，这不作为算法优化方向。

---

## 10. 最终判断

当前最该做的不是“再让模型更大/更多 loss/更密阈值”，而是**把论文定义中的 observation-consistent shared recovery option 直接变成 candidate-level evidence，并保持 semantic-local injection**。

v48.44 ROCT 是目前最值得做的一步，因为它同时满足：

1. 针对 v48.43 真正观测到的失败（Near deployability false-veto + Contact pred_adv sign collapse）；
2. 不重复 changelog 已证伪的容量/损失/阈值路线；
3. 不使用 regime-specific 策略；
4. 与论文最核心的 oracle-vs-deployable recoverability 定义直接对齐；
5. 2x2 实验能给出清晰 falsifiable causal answer。
