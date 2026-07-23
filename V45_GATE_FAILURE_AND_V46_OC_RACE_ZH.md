# OC-RAP v45 Natural Gate 失败诊断与 v46 OC-RACE 优化报告

日期：2026-07-23

## 1. 结论先行

本轮 `v45 RAVE` 没有通过 natural gate，**首要原因不是 gate 太严格，而是 direct-value 学习目标与实现同时出现了系统性偏差**：Near-contact 的教师优势分布有大量精确并列（median 与 q75 均为 0），但 v45 将所有 `teacher_delta <= 0` 都当成硬负例，持续把候选相对 nominal 的预测优势压到负区间；与此同时，名义上的两个 Near/Contact expert 没有任务专门化监督，只通过一个均衡正则进行软混合，两个 expert 在统计上不可辨识，容易退化为近似相同的头。因此两个 v45 checkpoint 在 Near-contact 上均找不到有限的 opportunity+score rule，流水线按预设 falsification rule 正确停止，没有进入 offline/Waymax。

另外发现一个会影响全部后续结论的 **P0 工程错误**：Stage-0 传入 `FREEZE_PREFIXES=""` 试图全量解冻 clean base，但 `scripts/train_ocrap_v39_ocrac.sh` 使用 `${FREEZE_PREFIXES:-默认列表}`，空字符串会被重新替换成默认冻结列表。也就是说，用户命令中的 `RETRAIN_CLEAN_BASE=1` 并不能证明 base 被完整解冻重训。旧 clean-base 的 `train_summary.json` 若仍含非空 `freeze_param_prefixes`，必须重新训练。

v45 的 Contact “通过”也不能视为安全证据：balanced/precision 在验证折分别选中 7/9 个动作，其中 2/3 个是有害选择，即条件有害率为 28.6%/33.3%。旧 gate 用“有害选择数 / 全部 group 数”稀释风险，因此仍标记 valid。v46 改为同时约束：

1. 全 group 的 harmful exposure；
2. **真正会执行的 selected actions 中的 harmful rate**；
3. 预测优势与教师优势的相关性；
4. scene-disjoint fit/verify；
5. development 与 publication contract 分离。

本报告给出的 v46 修改具有明确的理论方向和可证伪实验路径，但**不能在未运行新实验、官方/匹配 baseline、多随机种子闭环评估之前宣称 SOTA**。

---

## 2. 对论文的整体理解

### 2.1 Motivation

论文的核心问题不是“是否能避免当前碰撞”，而是：**当前动作是否保留了部署时真正可选择、可执行的后续恢复余量**。传统 branch-wise/oracle recoverability 允许对每个隐藏未来选择不同恢复动作，但部署车辆在执行 prefix 后只能看到 post-prefix observation；若多个隐藏未来在该观察下不可区分，却要求互相冲突的恢复动作，那么 branch-wise 可恢复性是 oracle artifact。

论文提出的核心概念是 **oracle-to-deployable recoverability gap**：

- local safety：当前短时风险是否低；
- oracle recoverability：已知隐藏未来身份时，每个分支是否各自存在恢复动作；
- deployable recoverability：只根据执行后的可观测信息，是否仍存在对 observation-equivalent roots 共同有效的恢复动作。

### 2.2 核心算法链

论文中的 OC-RAP 主体可以概括为：

1. 为每个候选 executable prefix 构造 counterfactual futures；
2. 将完整未来压缩为 **recovery-sufficient latent roots**，而不是直接按轨迹几何聚类；
3. 学习 post-prefix observation compatibility/equivalence kernel；
4. 对每个 root 与 recovery option 预测 signed recovery margins；
5. 用 OC-MERO 在 observation-equivalent roots 上先求 shared recovery option，再做 lower-tail aggregation；
6. 用 anti-oracle loss 抑制“oracle 高、deployable 低”的假恢复；
7. 用 CRISP/校准 admission rule 决定候选是否可进入动作集合；
8. nominal 可部署恢复性足够时保持 nominal，否则仅选择通过恢复与动作性约束的候选。

这条逻辑是论文最有潜力的主创新。v45/v46 的 direct-value expert 应被定位为 **用于扩展稀疏 intervention coverage 的选择性 residual admission 模块**，不能替代 OC-MERO，也不应让论文主线变成普通 MoE 排序器。

### 2.3 要计算和论证的指标

论文主指标包括：

- **FRA**：False Recoverability Admission Rate；
- **ODG**：Oracle–Deployability Gap；
- **DRS**：Deployable Recovery Success；
- **bounded NUP**：Nominal Utility Preservation；
- collision/intervention 及 calibration reliability；
- Contact/Post-contact：secondary collision、stable stop、maximum yaw-rate violation、route rejoin、harm proxy/impact severity。

论文实验部分设计了五个 regime，而当前代码/报告主要完整覆盖 Safe、Near-contact、Contact 三个数据根。因此最终投稿还需把论文中的五-regime声明、数据表和实验表统一，不能保留超出当前证据范围的占位结论。

---

## 3. v45 结果的直接证据

### 3.1 训练表现

| variant | best epoch | best validation direct loss | epoch 2 | epoch 3 | 训练/验证样本 |
|---|---:|---:|---:|---:|---:|
| balanced | 1 | 2.5666 | 3.1281 | 3.0407 | 30,114 / 4,373 |
| precision | 1 | 4.2477 | 5.3300 | 5.1676 | 30,114 / 4,373 |

两个 variant 都在第 1 epoch 最好，随后明显恶化。v45 使用 head-only 高学习率（balanced `1.5e-4`、precision `1e-4`）、direct loss 权重 10、且没有显式 gradient clipping；这不是 natural gate 失败的唯一原因，但说明优化过程不稳定，继续堆 epoch 或放宽 gate 没有依据。

### 3.2 Calibration 关键统计

| variant/regime | corr | pair MAE | teacher adv median/q75/q95 | predicted adv median/q75/q95 | verify selected | positive | harmful | precision | old valid |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| balanced Near | 0.1570 | 0.2863 | 0 / 0 / 0.0458 | -0.2495 / -0.1663 / -0.0267 | 0 | 0 | 0 | N/A | false |
| balanced Contact | 0.2726 | 0.2761 | 0 / 0 / 0.0810 | -0.2473 / -0.1623 / -0.0434 | 7 | 5 | 2 | 0.7143 | true |
| precision Near | 0.1533 | 0.3052 | 0 / 0 / 0.0458 | -0.3832 / -0.2474 / -0.0298 | 0 | 0 | 0 | N/A | false |
| precision Contact | 0.2939 | 0.2794 | 0 / 0 / 0.0810 | -0.3810 / -0.2547 / -0.0408 | 9 | 6 | 3 | 0.6667 | true |

Natural gate 要求同一 checkpoint 的 Near 与 Contact 都有效。两个 checkpoint 的 Near 均无有限规则、verify selection 为 0，因此 `VALID_RUNS` 为空，流程退出。这个停止是正确行为。

### 3.3 为什么预测优势整体偏负

v45 数据中 teacher advantage 的 median 和 q75 都是 0，意味着至少一半到四分之三的候选对是 nominal tie。原 loss 中：

```python
neg_mask = t_delta <= 0
```

这会把 tie 当作“候选应显著差于 nominal”，并施加负 margin。大量 tie 的梯度远多于稀疏正机会，导致 balanced/precision 的 predicted advantage median 分别约为 -0.25/-0.38。模型不是只“保守”，而是被错误监督成系统性负偏。

### 3.4 Contact 的旧 valid 是分母错误造成的假安全感

旧 calibration 同时报告：

- balanced：verify 2 harmful / 7 selected = 28.6%；
- precision：verify 3 harmful / 9 selected = 33.3%。

但 gate 使用：

- balanced：2 harmful / 122 groups，group-exposure UCB90 = 0.0483；
- precision：3 harmful / 122 groups，group-exposure UCB90 = 0.0599。

当 selector 只执行少量动作时，用全部 group 作分母会把错误执行风险稀释掉。对部署最直接的问题应是：**在系统决定干预的条件下，有害干预概率多大**。因此 v46 新增 `harmful_selected_ucb90` 并把它作为硬门。

### 3.5 stated `FINAL_RUN=1` 与归档 JSON 不一致

上传的 v45 JSON 中约束仍是 development contract：`required_min_scenes=20`、`max_verify_harmful_group_ucb=0.12`。而用户给出的命令声明 `FINAL_RUN=1`。因此归档结果只能证明“开发约束下 Contact 曾被标记 valid”，不能证明执行过 publication contract。原因可能是旧环境变量覆盖、复用旧 calibration artifact、或该压缩包不是最终命令新生成的同一批产物；仅凭现有文件无法唯一判定。

v46 的每个 calibration JSON 显式写入 `contract_mode`、`valid_for_development`、`valid_for_deployment` 和 `valid_for_active_contract`，并在 final mode 强制独立 calibration roots，避免再次出现这种歧义。

---

## 4. 算法层面的缺陷与优先级

### P0：两个 expert 不可辨识

`ALGORITHM_CHANGELOG.md` 对 v45 的设想是 task-specific Near/Contact experts，但实际训练脚本使用 `soft_observation` mixture，只有均衡正则，没有 expert-specific target。若两个头结构相同、输入相同、损失只作用于加权和，则交换/复制两个头不会改变主损失，容易发生 expert collapse。v46 对 expert 0/1 分别施加 Near/Contact bucket loss，再用共享观察 router 进行部署时插值。

### P0：tie 被错误当作 hard negative

已由分位数与预测分布直接验证。v46 使用三段式 target：

- `delta >= positive_gain`：正机会；
- `delta <= -negative_gain`：真正负例；
- 中间 dead zone：温和回归，不施加负 margin。

### P0：风险统计与部署事件不一致

旧 gate 控制全 group exposure，却没有控制 selected-action conditional risk。v46 同时报告并约束两种 Wilson upper bound。最终论文应明确 claim 是“在 scene-independent calibration assumptions 下的有限样本二项上界”，不能将其写成无条件 conformal 保证。

### P1：训练排序集合与可部署集合不一致

v45 listwise loss 对 group 内所有候选排序，包括部署 direct head 不允许 admission 的 candidate family。模型优化的最优排序可能在部署时不可执行。v46 只在 `nominal + allowed recovery macros` 上计算 listwise target。

### P1：软 router 可能从候选本身猜 task

候选 macro、prefix parameters、planned states/control 都不应决定“当前 scene 属于哪个恢复 expert”。否则同一 scene-time 的不同候选可能路由到不同 expert，破坏相对 nominal 比较。v46 的 `shared_raw` router 只读取候选无关的 agent history、BEV、route、map、dynamic-map 块；测试确认同 scene observation 下更换候选块不会改变 router logits。

### P1：全局 checkpoint 指标掩盖 worst regime

v45 用混合数据的 `loss_direct_recovery_value` 选 checkpoint。某一 regime 改善可掩盖另一 regime 退化。v46 记录 Near、Contact 与 worst-regime loss，并以 `max(loss_near, loss_contact)` early stopping。

### P1：候选生成上限

当前训练中有价值的 teacher-positive opportunity 很稀疏，且 broad brake/yield/pull-over/stabilize 家族整体可能贡献大量负样本。值函数只能排序已有候选，不能创造不存在的恢复轨迹。v46 在 fit fold 上选择有正支持的 macro family，再固定到 verify fold；但若最终 positive recall 仍低，下一步应优先改 candidate lattice/continuous parameter proposal，而不是继续调阈值。

### P2：不确定性 logvar 分支价值很低

v45 `direct_value_point_weight=0`，使 logvar 基本未受监督，而部署规则也不使用该 self-reported variance。v46 为兼容旧 checkpoint 保留接口并给极小 point anchor，但最终 ablation 若证实无贡献，应删除 logvar head，避免论文中出现“看似 probabilistic、实际未使用”的模块。

### P2：opportunity head 与 score head 可能冗余

双门机制有合理解释：先判断“是否存在改善机会”，再在机会集合中判断“改善幅度是否足够”。但必须做 single-score 与 two-stage ablation；若 opportunity head 不能显著提升 coverage-risk Pareto frontier，应简化。

---

## 5. 工程层面的错误与结果风险

### 5.1 Clean-base 没有真正全量解冻

原脚本：

```bash
export FREEZE_PREFIXES=${FREEZE_PREFIXES:-encoder,...}
```

`:-` 会把“已定义但为空”也替换成默认值。Stage-0 的 `FREEZE_PREFIXES=""` 因此失效。v46 改为 `${FREEZE_PREFIXES-default}`，并在训练后读取 summary，要求 `freeze_param_prefixes=[]` 才写 clean marker。

### 5.2 冻结 encoder 仍处于 train mode

PyTorch 的 `model.train()` 会重新开启 frozen encoder 内的 dropout。参数虽不更新，但 direct head 训练时看到随机 feature，calibration/inference 时却看到确定 feature，形成 train–calibration feature shift。v46 将“所有参数都冻结”的模块子树保持为 eval mode，而 direct heads/router 继续 train mode。

### 5.3 Validation 被同时用于 early stopping 与 calibration

当前 val 用于 checkpoint selection，又被切成 fit/verify 来选 admission threshold。即使 fit/verify 内部 scene-disjoint，也不独立于 checkpoint selection。development 可这样快速筛选，但论文 final 必须使用独立 calibration roots，test 只能一次性评估。

### 5.4 当前数据尚不支持 final paper contract

报告中：

| split | samples | scenes | scene-time groups | 关键状态 |
|---|---:|---:|---:|---|
| train Safe | 20,000 | 1,171 | 2,500 | calibration empty；future-source warnings 很多 |
| val Safe | 2,328 | 132 | 291 | calibration empty |
| test Safe | 3,216 | 175 | 402 | held-out，但 `supports_womd_primary_claim=false` |
| train Near | 13,324 | 600 | 1,800 | 1.88% post-contact |
| val Near | 2,219 | 113 | 279 | pure Near；calibration empty |
| test Near | 2,237 | 115 | 282 | pure Near；calibration empty |
| train Contact | 16,790 | 500 | 2,000 | 100% post-contact label |
| val Contact | 2,039 | 70 | 228 | 100% `post_contact_counterfactual` |
| test Contact | 2,367 | 74 | 264 | 100% `post_contact_counterfactual` |

所有报告均显示 calibration split empty，且 `supports_womd_primary_claim=false`。Contact val/test scene 数低于 final calibration 最低 100 scene，而且其 Contact 是 counterfactual construction，不等同于真实 observed post-impact states。论文中“真实碰撞后控制”措辞必须受限，或补建真实/仿真接触状态数据。

### 5.5 Safe future-source 完整性问题

Safe 报告包含大量“缺少 replay/reactive/targeted 中至少一种 future”的 warning。Near/Contact 有 targeted futures，Safe 主要只有 replay/reactive。若 Safe 与压力 regime 使用不同 future family，OC-MERO 的 root richness 和 calibration difficulty 不一致，可能造成 nominal lock 看起来很好但不是同等强度比较。最终应统一最低 future-source contract，或明确 Safe 只用于 nominal preservation 而非 recovery stress claim。

---

## 6. 与 external baselines 的关系

本轮实际上传的只有 5 个文件，没有三个 `external_baselines.zip` 原始结果包。因此以下数值只能沿用代码包 `ALGORITHM_CHANGELOG.md` 的既有 snapshot，并结合 baseline 脚本检查方法逻辑，不能重新核验每个原始 JSON：

- Safe：nominal/log replay/Wayformer-BC-lite 与 OC-RAP 均保持 NUP=1、FRA=0、PCD≈0.6163；GameFormer-lite PCD≈0.6189，但 NUP≈0.9468、intervention≈39.2%；BeTopNet-lite intervention≈17.6% 且无 PCD 增益。
- Near：现有 OC-RAP nominal/old certificate PCD≈0.5735、FRA≈0.0761、NUP=1；lite baseline 中 predictive safety filter PCD≈0.5466，intervention≈44.6%。这只证明旧 OC-RAP nominal-preserving baseline 强，不证明 v45 direct head 有效，因为 direct path 未执行。
- Contact：现有 OC-RAP PCD≈0.5723、FRA≈0.0780、NUP=1；severity minimization FRA≈0.0557，但 NUP≈0.7212、PCD≈0.4159、intervention≈80%。

脚本实现表明：

- Safe：Wayformer-BC、GameFormer-lite、BeTopNet-lite；
- Near：MARC-lite、RACP-lite、expected/CVaR/DRO-CVaR filters、predictive safety filter、oracle recovery upper bound、GameFormer-lite；
- Contact：post-impact MPC-lite、post-crash braking、post-collision restoration、severity minimization。

这些是 `*_lite` 或工程复现，不是官方 SOTA checkpoint。CCF-A 投稿需要：统一 observation/candidate/action budget、同一 scene split、同一 Waymax closed-loop protocol、完成全部 scene、多 seed，并把 oracle method 明确标为 upper bound 而非可部署 baseline。

---

## 7. v46 OC-RACE 的修改

名称：**Observation-Consistent Regime-Adaptive Calibrated Experts (OC-RACE)**。

它不是替换论文主算法，而是对 OC-RAP 的 selective residual admission 做结构化修复。

| 模块 | v46 修改 | 预期作用 |
|---|---|---|
| target | dead-zone-aware positive/tie/negative loss | 消除 tie 被压成负优势的系统偏差 |
| experts | Near/Contact expert-specific supervision | 使 expert 可辨识、避免 collapse |
| router | candidate-invariant shared-observation router | 防止从 candidate macro/prefix 猜 task；保持同组相对比较一致 |
| mixture | expert loss + 小权重 deployable mixture loss | 兼顾专门化与最终混合输出 |
| listwise | 仅 nominal + deployable recovery macros | 对齐训练排序与部署动作集合 |
| macro set | fit-fold support selection，verify 前冻结 | 丢弃无正支持 macro，避免硬编码 merge 或 broad noisy set |
| calibration | scene-disjoint fit/verify | 降低同 scene 泄漏 |
| risk | group exposure UCB + selected conditional UCB | 防止低 coverage 稀释有害执行风险 |
| gate | correlation + finite rule + precision + 双 UCB | 只有可学、可用且风险受控才进入 Waymax |
| checkpoint | worst-regime validation metric | 防止 Near/Contact 一好一坏 |
| optimization | LR 降到 5e-5/3e-5，loss weight 1，clip 1.0 | 降低 epoch 1 后发散 |
| frozen model | frozen subtrees 保持 eval | 消除 dropout feature shift |
| clean base | 空 freeze 正确解析 + summary 后置验证 | 确保 clean refresh 真正全量训练 |
| contracts | development/final 明确分离 | 防止开发结果误标 publication evidence |

### 修改文件

- `src/ocrap/models/losses.py`
- `src/ocrap/models/ocrap.py`
- `src/ocrap/models/inference.py`
- `src/ocrap/cli/train.py`
- `src/ocrap/config/defaults.py`
- `scripts/train_ocrap_v39_ocrac.sh`
- `scripts/train_ocrap_v46_race.sh`
- `scripts/calibrate_ocrap_v46_race.sh`
- `scripts/run_ocrap_v46_race.sh`
- `run_v46_two_gpu_fast_commands.txt`
- `tools/calibrate_direct_value_risk_v46.py`
- `tools/select_v46_candidate.py`
- `tools/check_v46_quick_gate.py`
- `tests/test_v46_race.py`
- `ALGORITHM_CHANGELOG.md`

---

## 8. 哪些保留、哪些修改、哪些应停止

### 继续保留

- oracle-to-deployable gap 与 observation consistency；
- recovery-sufficient roots；
- OC-MERO shared-option aggregation；
- anti-oracle loss；
- Safe nominal lock；
- physical actionability；
- calibration 失败即停止昂贵闭环；
- 2→4→8 的 staged Waymax cost ladder。

### 修改后保留

- direct-value residual：必须 tie-aware、expert-identifiable、deployment-aligned；
- soft routing：只允许共享观察输入，且需与 hard router/uniform router 做 ablation；
- broad macro candidates：训练可保留，部署只允许 fit-supported family；
- risk calibration：双分母、独立 scene、独立 final calibration set。

### 不应继续重复

- 继续调 additive q 试图解决零 coverage；
- 仅放宽 threshold/rescue/bypass 让 gate “通过”；
- raw flattened action adapter；
- bucket-agnostic scene-time grouping；
- 将 tie 全当 hard negative；
- 将 validation calibration 标成 final；
- 用 harmful/all-groups 替代 harmful/selected-actions；
- 在 candidate frontier 缺少正机会时继续堆 value-head complexity。

---

## 9. Novelty 与 CCF-A 投稿定位

### 9.1 真正有竞争力的 novelty

最强主张仍应是：**recoverability certificate 必须受 post-action observation equivalence 约束；branch-wise 存在恢复动作并不推出部署可恢复**。这比“又一个风险代价”或“又一个 emergency controller”更清晰。

OC-RACE 可以作为第二层贡献：在保持 OC-MERO certificate 不变的前提下，提出 **observation-routed, regime-specialized, selectively calibrated residual admission**，用于处理 Near/Contact 的不同价值尺度和极端类别不平衡。

### 9.2 与近期工作的边界

近期 partial-observability/contingency work 已出现“不同环境假设共享一致轨迹段”的思路；occlusion-aware contingency planners 也同时优化 exploration/fallback；2025 年风险厌恶 calibration 工作给出了 prediction set 到 max-min action 的决策论联系；2025–2026 robust conformal planning 还指出交互式 policy update 会破坏 exchangeability。论文不能泛称“首次在部分可观测环境做共享决策”或“无分布假设安全保证”。应把差异精确写为：

- shared object 是 **post-prefix recovery option/affordance**，不是只共享当前 trajectory prefix；
- equivalence 由执行动作后的 observation compatibility 定义；
- 显式测量 oracle-to-deployable gap/FRA；
- calibration 针对 action admission，并区分 group exposure 与 selected-action conditional harm；
- 对 policy-induced distribution shift 只作 limitation，除非引入 robust/online recalibration。

### 9.3 理论部分建议

至少补充以下命题/定理：

1. **Oracle upper bound**：`R_dep <= R_orc`，并给出 gap 为 0 的充分条件；
2. **Observation refinement monotonicity**：更细的合法 observation partition 不降低 deployable recoverability；
3. **Shared-option feasibility**：OC-MERO 对 equivalence class 的 max-min/shared-option 解释；
4. **Calibration guarantee**：在 scene-i.i.d. calibration 与固定 selector/threshold 下，Wilson/binomial upper bound 控制 selected harmful probability；
5. 明确 adaptive policy/interaction shift 时该保证不自动成立。

---

## 10. 下一轮实验顺序

### Phase A：先验证修复是否命中根因

按以下 ablation 逐步运行，不能只跑 full model：

1. v45 原始；
2. `+ tie dead zone`；
3. `+ expert-specific supervision`；
4. `+ shared-observation router`；
5. `+ deployable listwise set`；
6. `+ fit-supported macros`；
7. `+ dual-risk gate`；
8. full v46。

每项至少报告：pair MAE、Spearman/Pearson、positive pair recall、top-1 opportunity capture、selected precision、selected harmful UCB、coverage、macro support、Near/Contact worst-regime loss。

### Phase B：判断瓶颈是 scorer 还是 candidate frontier

计算 oracle best candidate 的 PCD gain 分布与 upper-bound intervention curve：

- 若 oracle positive opportunity rate 很低：改 candidate generation；
- 若 oracle positive 多但模型捕获差：改 representation/loss；
- 若离线捕获好、闭环差：检查 replanning mismatch、teacher label mismatch、observation shift；
- 若 Contact 明显好于 Near：Near 的 weak-positive/tie imbalance 仍未解决。

### Phase C：论文级闭环

- 独立 train/val/cal/test；
- 每 regime 至少数百 calibration/test scenes；
- 3–5 seeds；
- paired bootstrap scene-level CI；
- 所有 baseline 同预算；
- 报告平均值不能替代 worst seed/low-tail；
- 一次性 held-out test，不能反馈调参。

---

## 11. 对结果提升的合理预期

v46 最可能改善的是：

- Near 的预测优势从整体负偏回到围绕 0、正机会右移；
- 两 expert 的 bucket loss 与 router accuracy 出现可解释分化；
- Contact 不再以 28–33% selected harmful rate 假通过；
- checkpoint 不再牺牲某一 regime；
- clean base 与 frozen feature 行为可复现；
- 无效 macro 被 fit-only 证据移出 deployable set。

但以下情况仍可能使 v46 不通过：

- candidate set 中没有足够正 PCD gain；
- Near teacher target 本身噪声大、绝大多数是 tie；
- `post_contact_counterfactual` 与 Waymax 真闭环接触状态存在 domain gap；
- 当前 calibration scene 数不足以得到严格 conditional-risk UCB；
- shared observation 无法可靠区分 Near 与 Contact，需要连续 risk state 而非二 expert。

如果 v46 仍失败，应优先分析 oracle frontier 与 label reliability，而不是继续叠加 v47 阈值补丁。

---

## 12. 验证状态

本地完成：

- 全量单元测试：`100 passed`；
- Python compileall：通过；
- shell syntax：通过；
- 新增回归测试覆盖：tie dead zone、真正负例、frozen dropout、conditional UCB、shared-router candidate invariance。

新代码尚未在用户 GPU/WOMD/Waymax 环境跑出 v46 数值，因此这里只能声明“修复已实现且静态/单元测试通过”，不能声明性能已达到 SOTA。
