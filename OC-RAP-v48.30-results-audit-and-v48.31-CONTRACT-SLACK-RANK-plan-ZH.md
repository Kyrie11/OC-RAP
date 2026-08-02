# OC-RAP v48.30 结果、工程合同与算法根因审计
## 以及 v48.31 CONTRACT-SLACK-RANK 优化方案

日期：2026-08-02  
审计对象：`post-collision.tex`、`OC-RAP.zip`、`reports.zip`、v48.30 主实验与消融结果、`大模型建议.md`

---

## 0. 结论先行

### 0.1 `RC=20` 的直接含义仍然成立

v48.30 四个主分支均满足：

- pipeline 正常完成；
- gate 已执行；
- 未读取 test/stress roots；
- rejection 类型为 `development_rule_fit_rejection`；
- top-3 proposal-constrained oracle 在 Near 与 Contact 都存在安全正机会。

所以，当前部署 selector 确实没有通过注册的 Natural gate，不能把结果包装成“仅仅是 gate 太严格”。

但本轮进一步审计发现：**v48.30 对“为什么失败”的算法归因被多处工程合同偏差混淆**。因此可以确认“当前 selector 不合格”，却不能据此断言“连续 safety slack 思路本身无效”。

### 0.2 根因不是 proposal 不可行，而是“动作身份排序 + 工程合同错位”

Near 的 proposal-evidence safe-positive AUC 约为 0.735–0.784，说明模型能一定程度识别“哪些 scene-time group 有恢复机会”；但 proposal top-1 teacher correlation 约为 +0.021 / -0.023，certificate 又完全没有选中 safe-positive。Contact 的相同相关性约为 -0.118 / -0.116，safe-positive AUC 约为 0.532 / 0.457。

这意味着主要问题是：

> 模型能粗略识别机会场景，却不能稳定识别同一组候选中“哪一个动作”才是安全恢复动作，更不能把它稳定排在 nominal、高收益但有害动作、以及宏类型捷径之前。

### 0.3 Near-contact 有正信号，但尚不具备 CCF-A 主结果证据

Precision Near 的 adaptation-dev near-miss rule：7 次选择、3 个安全正样本、1 个 harmful、safe recall 0.375、平均 teacher advantage +0.273。这是目前最值得保留的局部信号。

但 certificate 上：6 次选择、0 个安全正样本、3 个 harmful、平均 advantage -0.294，且选择全部集中于一个 macro。该结果只能支撑“有可优化信号”，不能支撑可发表的安全准入结论。

### 0.4 Contact 尚未证明 safe admission

Balanced/Precision Contact certificate 分别选择 17/28 次，安全正样本均为 0，harmful 为 6/16，平均 teacher advantage 为 -0.215/-0.275。proposal top-1 correlation 为明显负值。

因此 Contact 目前不是“略低于投稿门槛”，而是**核心动作排序尚未建立**。其 recoverability 表征与 top-3 proposal 可保留，但 admission/ranking 不能作为论文主结果。

### 0.5 Safe regime 当前只完成阈值校准，不等于策略已通过独立证书

v48.30 的 Safe calibration 有效，`gamma_rec=1.2729634`；但 `SAFE_REGIME_STATUS.json` 明确说明没有注册 scene-disjoint Safe policy certificate，Safe 只通过 calibrated recovery threshold 与 paired non-inferiority closed loop 检验。当前 v48.30 shadow 又因缺失脚本在仿真前退出，所以“三个 regime 都已证明”仍不成立。

---

## 1. 论文主线与代码优化必须保持一致

论文最有价值的主线不是三种场景分类，而是：

1. recovery-sufficient latent roots；
2. post-prefix observation equivalence；
3. shared recovery policy，而不是 hidden-root-conditioned oracle action；
4. OC-MERO 对 observation-consistent deployable recoverability 的聚合；
5. CRISP 把 recoverability 当成 candidate admission property，并优先保持 nominal；
6. calibration 控制 false recoverability admission。

因此后续算法应该使用同一个统一决策语义：

\[
\Delta \mathbf m(a)=
\bigl[
\Delta m_{\rm DRS},
\Delta m_{\rm dep},
\Delta m_{\rm gap},
\Delta m_{\rm hard},
\Delta m_{\rm proxy}
\bigr],
\]

其中每个量都相对 nominal、连续、可解释，并在全部数据上使用同一个函数、同一套参数和同一个 admission contract。Near/Contact 只允许作为报告分层与合同核查分层，不能作为模型输入或策略路由条件。

### 1.1 论文中需要删除的内部矛盾

`post-collision.tex` 的主要 Method 叙事是统一的，但附录存在一段“Regime-conditioned recovery admission”，明确提出只在 low-headroom contact regime 使用第二条 protective certificate。这与论文前文、用户要求以及当前希望强化的 novelty 不一致。

投稿前应把该段改为：

- support-conditioned，而非 regime-conditioned；
- 所有候选共享相同 nominal-relative margin；
- 允许 macro semantic eligibility，但不允许 contact ID 改变阈值或策略；
- hard rule 仍是独立不可补偿 veto；
- 只有数据支持度决定某个 learned margin 的置信权重，而不是场景类别决定策略。

---

## 2. v48.30 主实验的真实结果

| 分支 | Dev near-miss：选中/正/害 | Dev recall | Dev advantage | Certificate：选中/正/害 | Cert recall | Cert advantage | Proposal safe AUC | Proposal top-1 corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced Near | 6 / 1 / 0 | 0.125 | +0.104 | 5 / 0 / 3 | 0 | -0.355 | 0.735 | +0.021 |
| Precision Near | 7 / 3 / 1 | 0.375 | +0.273 | 6 / 0 / 3 | 0 | -0.294 | 0.784 | -0.023 |
| Balanced Contact | 11 / 2 / 4 | 0.118 | -0.098 | 17 / 0 / 6 | 0 | -0.215 | 0.532 | -0.118 |
| Precision Contact | 13 / 0 / 2 | 0 | -0.042 | 28 / 0 / 16 | 0 | -0.275 | 0.457 | -0.116 |

Oracle support 并不缺失：certificate top-3 proposal 内含 9 个 Near safe-positive group 和 20 个 Contact safe-positive group。Oracle precision LCB 分别约为 0.846 和 0.924。

这排除了两个错误方向：

- 不是必须把 proposal 从 top-3 扩到 top-8；
- 不是 certificate 数学上完全不可满足。

真正失败的是 learned evidence reranker/admission 没有把现成的安全候选选出来。

---

## 3. 确认的工程错误与其对算法分析的影响

### E1. 训练期“部署指标”与 calibration 的可执行样本总体不同

v48.30 的训练验证 `_direct_policy_batch_stats` 没有应用 certificate 中的三项过滤：

- `feasible == true`；
- `hard_violation <= max_hard`；
- `nominal_deviation >= 0.002`。

结果是训练验证报告 Near/Contact group 约 119/290，而 adaptation-dev calibration 实际使用 110/279。机会数也不完全一致。

**影响：** checkpoint 在一个总体上选取，规则却在另一个总体上拟合。任何 epoch 优劣、stage 改动优劣、甚至“模型开始看到正样本”的结论都会被混入 population shift。

### E2. all-abstain 屏障定义错误

v48.30 的屏障依据 raw admission rate 判断是否 abstain。只要模型选了 harmful 或无收益动作，就会被视为“不是 all-abstain”。

**影响：** 一个完全没有有效安全准入、但会错误干预的 checkpoint，可能比真正谨慎的 checkpoint 获得更低惩罚。

正确语义应是：当 exact safe contract 可用时，`valid_safe_admission_count == 0` 才是 all-abstain；invalid/harmful switch 不能解除屏障。

### E3. “自然总体训练修复”只作用于 stage 2

v48.30 stage 1 仍然采用分层、有放回采样；stage 2 才改为自然、无放回，但 stage 2 又冻结 benefit/harm factor heads，只训练很小的 admission residual。

**影响：** 决定候选动作身份与组内顺序的主体仍然由改变先验的 stage 1 学得。v48.30 的核心修复没有覆盖主要参数，因此 A/B/C 消融几乎不变并不奇怪。

### E4. checkpoint objective 没有把“两种困难分层都存在 safe top-1”设为词典序前置条件

当前 soft population metric 可以因 candidate AUC、soft mass 或平均 regret 改善而选择某个 epoch，即便该 epoch 在某个 reporting stratum 完全没有 safe top-1 hit。

**影响：** 优化方向偏向“是否有机会”分类，而不是“安全动作是哪一个”的组内识别。

### E5. 五个 learned safety factor 中两个没有可识别支持

基于训练 index 的全局支持审计：

| 因子 | eligible count | unique | std | positive count | 建议 learned reliability |
|---|---:|---:|---:|---:|---:|
| DRS margin | 5440 | 24 | 0.437 | 1386 | 1 |
| deployability margin | 5440 | 2158 | 0.212 | 2543 | 1 |
| gap margin | 5440 | 928 | 0.309 | 1138 | 1 |
| hard-rule learned margin | 5440 | 2 | 0.072 | 0 | 0 |
| harm proxy | 5440 | 1 | ~0 | 0 | 0 |

`harm_proxy` 在 reports 中也是全零。hard-rule 在 exact eligible population 内没有正例，因为大量明显硬违规样本已经被 executable filter 排除。

**影响：** 对五个 learned logits 直接取 max 会让无数据支持的随机 logit 成为全局 veto，或者让优化器浪费容量拟合不可识别目标。

修复不是拆分 regime，而是引入**全局、regime-agnostic support reliability**：无支持坐标向语义安全先验收缩；独立 measured hard veto 继续原样保留。

### E6. 候选级 AUC 被错误当作组内动作识别证据

Near 的 AUC 较高，但 top-1 correlation 接近 0；Contact top-1 correlation 为负。AUC 能被 scene context、危险程度、macro frequency 等 nuisance 特征提升，却不保证候选内正确排序。

**影响：** “分类头有效”不能推出“规划器有效”。主要 checkpoint 和消融分析必须以 proposal-contained safe top-1、safe admission precision/recall、harmful mass、regret 为主。

### E7. v48.30 development-shadow 物理链路实际没有运行

上传结果中的 `dev_shadow_controller.log` 显示：

```text
python: can't open file '.../tools/audit_v48_30_shadow_provenance.py': [Errno 2] No such file or directory
```

代码包中的 shadow 脚本还引用了另外两个不存在的工具：

- `check_v48_30_shadow_runtime_contract.py`；
- `check_v48_30_regime_targets.py`。

**影响：** 本轮包不支持任何 v48.30 closed-loop Near/Contact 改善或退化结论。物理结果只能在修复后重新运行 adaptation-dev shadow 得到。

v48.31 已补齐 v48.31 provenance/runtime/physical-support/regime-target 工具，并新增测试，确保脚本引用的工具真实存在。

---

## 4. RC=20 的根本原因分解

### 第一层：当前 rule fitting 确实不可行

四个分支都没有找到满足最小选择量、precision LCB、harmful UCB、safe recall、宏集中度和正收益的联合规则。不能靠降低 gate 或手工挑阈值解决。

### 第二层：组内 action identity 没学好

最直接证据是：

- proposal 内有 safe-positive；
- scene-level/candidate-level AUC 尚可；
- proposal evidence top-1 correlation 接近 0 或负；
- certificate 选中 safe-positive 为 0；
- Precision Near dev 有局部信号，但证书完全翻转。

模型的主要失败不是“不知道什么时候危险”，而是“不知道在同一个危险时刻应该选哪条恢复前缀”。

### 第三层：训练合同使动作身份错误难以被修正

stage 1 的重采样先验、stage 2 的冻结范围、错误的 all-abstain 屏障、与 calibration 不一致的 eligible population 共同使错误 checkpoint 有机会获胜。

### 第四层：统一 slack 中混入不可识别坐标

连续、统一、非退化的物理余量方向是正确的，但“连续”不等于“每个坐标都值得学”。当某个坐标在当前数据上恒定时，应做支持度收缩，而不是让随机 learned logit 参与 max-veto。

### 第五层：宏类型与 scene nuisance 捷径

Near certificate 的选中动作出现 0.8–1.0 的 macro concentration。说明模型可能将“某个 macro 常见于正样本”当成动作语义，而不是使用 candidate-vs-nominal 的物理变化来判断。

---

## 5. Near-contact 与 Contact 的投稿成熟度

### 5.1 Near-contact

**可保留的正信号：**

- Precision Near dev 的 3/7 safe-positive、recall 0.375、advantage +0.273；
- proposal safe-positive AUC 约 0.735–0.784；
- top-3 proposal oracle 支持充足；
- 部分消融也稳定保留 dev 正样本。

**主要缺陷：**

- certificate safe recall 为 0；
- 一半左右选中动作 harmful；
- certificate advantage 显著为负；
- top-1 correlation 接近 0；
- macro shortcut 明显；
- 本轮没有有效 closed-loop 物理证据。

**投稿判断：** Near 已达到“值得继续做、可形成有说服力消融故事”的研究原型阶段，但未达到 CCF-A 主结果程度。论文中不能称为 calibrated safe admission 已经成立。

### 5.2 Contact

**可保留的部分：**

- top-3 proposal 内有 20 个 safe-positive group；
- 统一 recoverability 表示、observation consistency、candidate generator 与 physical margin 设计仍有研究价值；
- harm AUC 有一定信号，说明危险候选不是完全不可识别。

**主要缺陷：**

- safe-positive AUC 约 0.457–0.532，接近随机或低于随机；
- top-1 correlation 约 -0.116；
- dev 和 certificate 都没有稳定 safe admission；
- certificate 选中动作的平均收益为负；
- harmful 比例高；
- 本轮 closed-loop 没有真正运行。

**投稿判断：** Contact 尚未达到可作为主要实验证据的阶段。当前只能作为“困难设置与失败分析”，不能支撑模型在 post-contact 下表现良好的核心 claim。

### 5.3 三种 regime 的总体 CCF-A 条件

在不重构数据集的约束下，至少需要：

1. exact train/dev/certificate contract 全部一致；
2. Near 与 Contact 的 proposal-contained safe top-1 在 development validation 上都非零且多 seed 稳定；
3. 两个 regime 都取得 `RC=0`，而不是只在 near-miss frontier 上改善；
4. certificate safe-positive selected 非零，harmful UCB 符合注册 gate；
5. Safe paired non-inferiority 有非空同源配对场景；
6. Near 的 clearance/TTC/exposure 与 Contact 的 free-space/clearance/re-contact 指标有方向一致的 paired 改善；
7. 至少 3 个随机种子，报告置信区间、failure rate 与 macro concentration；
8. 论文附录移除 regime-conditioned selector 叙事。

---

## 6. 哪些设计保留、修改或暂时移除

### 6.1 保留

- observation-consistent recoverability 主线；
- recovery-sufficient roots 与 post-prefix equivalence；
- OC-MERO/CRISP 作为论文核心；
- frozen top-3 proposal；
- nominal + top-k categorical one-action policy；
- raw benefit 与 safe admission 分离；
- candidate-vs-nominal 连续物理 margin；
- independent hard veto；
- bounded admission；
- natural certificate 与 dev-frozen rule；
- gate 的三态返回码 0/20/30；
- 不读取未经授权的 test/stress roots。

### 6.2 修改

- stage 1、2、3 全部改为 natural without replacement；
- checkpoint 指标改为 exact executable population；
- all-abstain 改为 valid-safe admission 语义；
- 将 safe top-1 支持设为 checkpoint 词典序主目标；
- 五 factor loss 与 runtime slack 使用全局 support reliability；
- stage 2 后增加低学习率 joint calibrator refinement；
- listwise/hard-negative 直接优化 proposal 内 safe action identity；
- 在 calibration 前强制比较 train validation group/opportunity counts；
- shadow 物理链路补齐 provenance 与 non-empty fail-closed。

### 6.3 暂时不作为主模型依赖

- top-8 proposal；
- unbounded admission；
- fixed oversampling prior；
- 仅凭 candidate AUC 选 checkpoint；
- 无支持的 harm_proxy learned veto；
- exact eligible population 内无正例的 learned hard-rule veto；
- regime ID、regime-specific threshold 或三套策略；
- 未经 repaired shadow 验证的 closed-loop claim。

---

## 7. v48.31 CONTRACT-SLACK-RANK 设计

### 7.1 统一、非 regime-routing 的安全语义

对所有候选使用同一个 nominal-relative component vector：

\[
\mathbf s(a)=
[s_{\rm DRS},s_{\rm dep},s_{\rm gap},s_{\rm hard},s_{\rm proxy}],
\]

其中正值表示越过非退化容差。learned effective margin 为：

\[
\tilde s_k=s_k^{\rm prior}+r_k(s_k-s_k^{\rm prior}),
\quad r_k\in[0,1],
\]

`r_k` 来自全训练总体的支持度，而不是 regime。当前审计建议 `r=[1,1,1,0,0]`。独立 measured hard veto 不受 `r_k` 影响。

统一 worst slack：

\[
S(a)=\max_k \tilde s_k(a).
\]

统一 safe utility：

\[
U_{\rm safe}(a)=B(a)-\lambda\,[S(a)]_+.
\]

同一个函数在 Safe、Near、Contact 中连续变化；不同 regime 只反映 margin 数值不同，不改变模型结构、阈值或策略。

### 7.2 三阶段优化

**Stage 1：Factor identity**

- natural, no replacement；
- 学 raw benefit ranking、支持度加权 component BCE 与 signed-margin regression；
- exact eligible validation；
- 不训练 admission。

**Stage 2：Admission identity**

- natural, no replacement；
- 冻结 factor heads；
- 训练 bounded admission residual；
- proposal top-3 内 listwise safe utility + hardest negative；
- checkpoint 使用 `direct_contract_safe_rank_risk`。

**Stage 3：Low-rate joint calibration refinement**

- natural, no replacement；
- 只解冻 benefit calibrator、harm calibrator、admission calibrator；
- 低学习率；
- 不解冻主 encoder/tournament，以避免破坏已有 proposal 排序；
- stage-transfer checker 确保其他参数没有静默变化。

### 7.3 新 checkpoint metric

`direct_contract_safe_rank_risk` 在原 population risk 基础上强惩罚：

- Near 或 Contact 任一报告分层 safe top-1 hit 为 0；
- safe top-1 recall shortfall；
- valid-safe all-abstain；
- invalid admission；
- safe top-1 regret。

这不是让模型识别 regime；它只是确保同一个模型在两个困难报告分层上都没有完全退化。

### 7.4 新 fail-closed 工程合同

- factor support contract；
- model/inference contract；
- all-stage sampling/training contract；
- stage transfer contract；
- train-validation vs adaptation-dev calibration count contract；
- shadow provenance/runtime/paired-scene contract；
- certificate gate 仍维持原注册阈值。

---

## 8. 下一轮消融及解释规则

四组消融每波两张 GPU，各一项 Balanced/Precision，最大并发 2：

1. `A_contract_natural_no_reliability_no_joint`
2. `B_add_support_reliability_no_joint`
3. `C_add_joint_refinement_no_reliability`
4. `D_full_contract_slack_rank`

解释：

- B>A：无支持 factor 的全局收缩有效；
- C>A：低学习率 joint calibrator refinement 能修复动作身份；
- D>max(B,C)：两者互补；
- A 仍失败且 top-1 无改善：问题不只是工程合同，需升级 candidate interaction/listwise teacher；
- dev 改善、certificate 仍翻转：优先查 nuisance/macro shortcut 与 scene-disjoint generalization；
- offline 改善、paired shadow 不改善：teacher 与闭环物理目标错位，应在后续注册 candidate-level temporal recovery supervision，而不是继续叠加 gate。

---

## 9. 容易再次出现的工程错误清单

在运行前后必须逐项确认：

- 训练、validation、adaptation-dev、certificate 使用同一 `positive_gain`、macro IDs、factor tolerances、max hard、min deviation；
- stage 1/2/3 都是无放回自然采样；
- stage 2 factor 参数完全冻结；
- stage 3 只改变三个 calibrator；
- checkpoint 与 inference 的 component count、scale、prior、reliability、slack temperature、bounded admission 完全一致；
- calibration rule 的 safe opportunity 是 proposal-contained safe-positive，不是 raw positive；
- all-abstain 不被 harmful switch 解除；
- shadow target scene 与 raw WOMD source split/shard 同源；
- paired control/method target key 非空且一致；
- Contact anchor 语义是 causal post-contact target；
- empty metrics 用 invalid/null 表示，不能用 0 冒充成功；
- `RC=20` 时不得读取 test/stress；
- `RC=30` 时不得给出算法结论；
- 不能在看过 certificate 后修改 gate 再称为 untouched final certificate。

---

## 10. 本地验证边界

v48.31 已在当前环境完成：

- 全量 pytest；
- `compileall`；
- 所有 shell `bash -n`；
- v48.31 脚本工具引用完整性检查；
- ZIP 完整性与 SHA256（见交付文件）。

当前环境没有真实 WOMD/Waymax 数据挂载与两张 A30，因此没有运行训练、certificate 或 closed-loop；不预先声称 v48.31 会得到 `RC=0`。

下一轮最关键的判据不是 AUC 是否再升一点，而是：

1. exact contract checker 通过；
2. Near 与 Contact 都出现稳定的 proposal safe top-1 hit；
3. certificate safe-positive selection 从 0 变为非零；
4. harmful selection 与 macro concentration 同时下降；
5. repaired paired shadow 给出连续物理量的方向一致改善。
