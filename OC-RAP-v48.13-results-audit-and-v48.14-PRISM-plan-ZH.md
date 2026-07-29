# v48.13 TERRA完整结果审计与v48.14 PRISM设计

## 1. 最终判断

v48.13没有通过Near+Contact Natural gate。主实验、4801/4802/4803三seed均没有任何variant同时满足两个regime的部署条件，因此没有合法的stress closed-loop结果。

Natural gate失败不是简单的阈值问题。v48.13已经把“高召回proposal”和“proposal内证据选择”分开，而实验明确表明：

1. top-k proposal已经学得很好；
2. proposal内harmful/dead/beneficial证据没有学好；
3. 尤其Contact的harm证据在不同scene之间不迁移；
4. 现有train与calibration/val/test存在真实的数据合同漂移；
5. 这轮上传的v48.13消融实际上无效，不能作为模块归因证据。

新的v48.14不再重训已经有效的proposal。它冻结proposal，仅使用scene-disjoint dedicated calibration的一部分，对轻量evidence adapter做目标域适配；剩余scene作为完全独立的certificate pool。

---

## 2. v48.13 Natural gate状态

### 2.1 主实验

| Variant | Regime | Candidate AUC | exact top-1 corr | Proposal oracle-best hit | Proposal any-positive hit | Proposal evidence benefit AUC | Proposal harm AUC | Verify选择 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7723 | -0.1108 | 0.9592 | 1.0000 | 0.7490 | 0.5189 | 0 |
| Balanced | Contact | 0.8251 | 0.0236 | 0.9697 | 0.9848 | 0.7335 | 0.3944 | 0 |
| Precision | Near | 0.6875 | 0.0627 | 高 | 高 | 可用但不足 | 不稳定 | 0 |
| Precision | Contact | 0.8225 | 0.0053 | 高 | 高 | 约0.78 | 约0.48 | 21但不合格 |

Precision Contact的21个verify动作不能部署：

- precision 0.381；
- precision LCB90 0.230；
- harmful rate 0.333；
- harmful UCB90 0.513；
- recall 0.242；
- 平均exact-teacher advantage -0.112；
- 选择集中于macro 5。

Natural gate拒绝是正确的。

### 2.2 三seed稳健性

三seed中没有任何Near/Contact组合通过。主要平均结果：

| Variant | Regime | Candidate AUC | top-1 corr | Harm AUC | Verify选择均值 |
|---|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7810 | -0.0917 | 0.5772 | 0 |
| Balanced | Contact | 0.8271 | 0.0447 | 0.4767 | 12 |
| Precision | Near | 0.7058 | -0.0102 | 0.5952 | 0 |
| Precision | Contact | 0.8358 | 0.0289 | 0.5360 | 17 |

Contact偶尔产生coverage，但precision置信下界低、conditional harmful UCB高，且平均teacher advantage接近零或为负。这不是单一seed偶然失败。

---

## 3. v48.13哪些设计有效

### 3.1 Set-valued top-k proposal明确有效

Balanced主实验：

- Near oracle-best proposal hit 0.959；
- Near any-positive hit 1.000；
- Contact oracle-best hit 0.970；
- Contact any-positive hit 0.985。

这说明TERRA已经解决了一个重要子问题：**真正有价值的恢复候选几乎总能进入top-3**。

Exact top-1仍弱，但在top-k证据重排框架中，proposal阶段不需要先精确解决唯一winner。该设计应完整保留，不应再回到单候选分类或all-pairs强制排序。

### 3.2 Proposal训练与Evidence训练分阶段冻结有效

Evidence训练没有反向破坏proposal，两个对象能够独立诊断。这一隔离方式既有算法价值，也提高论文消融的可解释性。

### 3.3 Evidence rerank统一执行合同方向正确

运行逻辑为：

```text
recovery candidates → frozen top-k proposal → evidence filtering/reranking → abstain
```

Stage E训练了proposal内成员，因此不同于历史上未训练runner-up的fallback。该合同应保留。

### 3.4 Opportunity-normalized macro certificate正确

Teacher正机会本身高度集中于macro 5。使用“模型额外集中度”而非绝对占比，避免把数据支持误判为模型捷径。当前Natural gate主要先败在precision和harm，因此macro并非第一瓶颈。

---

## 4. v48.13哪些设计没有起效

### 4.1 Proposal-distribution evidence没有学会harm

最强Contact candidate benefit AUC超过0.82，但proposal harm AUC约0.39–0.53，evidence-teacher correlation接近0或为负。

因此模型能识别“可能有恢复收益”，却不能可靠识别“这个看似有收益的候选实际上有害”。Natural gate的核心正是拒绝false-safe harmful recovery。

### 4.2 Same-group counterfactual pair仍不足

该目标理论上能消除scene严重程度，但实际top-k proposal中不一定同时出现足够的harmful和safe候选，pair监督稀疏。它不能替代大量、目标分布一致的harm证据样本。

v48.14不会删除该项，而把它放到D消融中验证；主增益来源改为target-domain evidence adaptation与hard-harm mining。

### 4.3 Exact top-1仍不稳定，但已不是当前第一优先级

Near exact top-1仍为负，Contact仅略正。然而top-k hit已接近饱和。继续投入大量损失优化唯一winner，可能重复历史上无效的pairwise/listwise尝试。现阶段更合理的是在高召回proposal中学习安全证据。

---

## 5. Dedicated calibration性质

三套dedicated calibration均已完整构建：

| Regime | Scenes | Scene-time groups | Samples |
|---|---:|---:|---:|
| Safe | 135 | 318 | 2,544 |
| Near | 316 | 765 | 6,039 |
| Contact | 543 | 1,896 | 16,843 |

### 5.1 它们与val/test比train更一致

Near：

- train `r_dep_star`均值约-1.794；
- calibration约-0.509；
- val/test约-0.801/-0.690；
- train hard violation均值0.089；calibration 0.035；val/test 0.009/0.016；
- train harm_proxy非零，calibration/val/test全部为0。

Contact：

- train `r_dep_star`均值约-1.792；
- calibration约-0.351；
- val/test约-0.561/-0.572；
- train hard violation均值0.094；calibration 0.022；val/test 0.015/0.021；
- train harm_proxy非零，calibration/val/test为0。

Dedicated calibration略比val/test容易，但候选数量、hard violation、recoverability、artifact比例和val/test明显更接近，而与train存在大漂移。

### 5.2 不能把同一calibration scene同时用于适配和证书

v48.14将Near/Contact dedicated calibration按scene拆为：

- 45% evidence adaptation train；
- 15% evidence adaptation dev；
- 40% certificate pool。

Encoder和proposal完全冻结，只微调轻量evidence adapter。Certificate pool随后仍由校准工具按scene拆fit/verify。因此：

- 不读取test；
- adaptation、early stopping、threshold fit、held-out verify四种角色互不重叠；
- 论文可以把它表述为scene-disjoint calibration-stage policy evidence adaptation。

---

## 6. 工程层面发现的问题

### 6.1 `gamma_rec_by_bucket_v48.json`缺失原因

v48.13的Stage P和Stage E均设置：

```bash
SKIP_POST_TRAIN_CALIBRATION=1
```

因此标准OC-MERO calibration和gamma从未生成。这是流程缺失，不是Natural gate直接造成的。

v48.14的certificate finalizer会原子生成：

- `calibration_mix/safe/near/contact_v48.json`；
- `gamma_rec_by_bucket_v48.json`；
- `direct_value_risk_near/contact_v48.json`。

### 6.2 dedicated direct JSON缺失原因

当前上传目录没有完成`recalibrate_v48_13_on_dedicated_set.sh`后的`dedicated_candidates`产物。新流程不再依赖隐式目录状态，使用临时目录完整生成后原子替换，并写入`CERTIFICATE_CALIBRATION_COMPLETE.json`。

### 6.3 Safe nominal-only被错误要求提供gamma/calibration

旧runner在进入`SAFE_NOMINAL_ONLY`分支前就检查gamma和calibration。因此Safe虽然完全禁用恢复策略，仍被Near/Contact缺失证书阻塞。

v48.14已经修复：Safe nominal-only只需要checkpoint；stress运行仍严格要求gamma和有效Near/Contact证书。

### 6.4 v48.13消融实际上无效

旧脚本定义了：

```bash
GROUPS=(...)
```

`GROUPS`是Bash保留的用户组ID数组。实际只运行了名为`1012_balanced`和`1012_precision`的任务，A/B/C/D均缺失，`ABLATIONS_COMPLETE.json`不存在。

因此本轮不能根据上传的v48.13消融对proposal/evidence模块做因果归因。v48.14改用`ABLATION_SPECS`，并新增8任务完整性检查。

### 6.5 Ordered NLL参数未按预期传递

v48.13脚本计算了`ORDERED_TOP1/ORDERED_ALL`，但调用通用trainer时重新读取了另一组默认值，导致预期权重没有真正生效。v48.14已统一参数名和传递链。

---

## 7. 三个regime分别存在的问题

### 7.1 Safe

Safe算法目标仍应是严格nominal non-inferiority。当前没有新的大规模paired结果，不能声明：

- collision/offroad非增；
- route progression下降不超过0.5%；
- NUP增加不超过1%；
- jerk/yaw-rate p95增加不超过5%；
- intervention episode增加不超过2%–3%。

Safe无需等待Near/Contact gate。工程修复后可以立即在`calibration_safe`做开发阶段paired probe，方法冻结后才使用`test_safe`作最终一次评估。

### 7.2 Near-contact

优点：top-3 proposal几乎总包含正候选，benefit AUC约0.75。

缺点：

- exact top-1不稳；
- harm AUC约0.52；
- verify precision LCB、harm UCB、support不足；
- recall通常低于0.35；
- 仍没有Natural-gate授权闭环，clearance/TTC/DRS/PCD/FRA/ODG目标均未验证。

Near的主要目标是让适配后的evidence从proposal中安全挑选少量高置信动作，而不是先追求唯一winner完全正确。

### 7.3 Contact

优点：candidate benefit AUC约0.82–0.84，proposal hit接近0.97。

缺点：

- harm AUC接近随机甚至反向；
- fit到verify迁移失败；
- 非零coverage常具有负平均teacher advantage；
- 无法进入secondary overlap、recontact、stable stop、post-contact clearance等闭环验证。

Contact是v48.14最主要的优化对象：hard-harm target adaptation必须提高harm AUC并降低false-safe harmful selection。

---

## 8. v48.14 OC-TRAC-PRISM

**PRISM = Proposal-aligned Risk adaptation with Independent Scene-disjoint certification Model.**

### 8.1 保留高召回proposal

v48.13 top-k proposal完全冻结。这样不会用少量calibration数据破坏已经达到约0.96–1.00的proposal recall。

### 8.2 Dedicated evidence adaptation

只微调Near/Contact regime-specific `direct_delta_adapters`：

- 训练数据来自dedicated adaptation train；
- early stopping来自adaptation dev；
- encoder、tournament、其它head全部冻结；
- test完全封闭。

该步骤直接解决train与目标分布的evidence合同漂移，而不需要耗时重构完整train set。

### 8.3 Dynamic false-safe hard-harm mining

在有序三状态NLL上，对当前预测为安全但teacher为harmful的proposal动态增权。权重只用于采样难度，hardness被detach，模型不能通过调整权重规避损失。

同时保留较弱的missed-benefit hard mining，防止模型退化为全拒绝。

### 8.4 独立certificate pool

适配完成后，只在未参与训练/early stopping的certificate pool上：

1. 生成标准OC-MERO calibration和gamma；
2. fit policy rule；
3. scene-disjoint verify；
4. 执行Natural gate。

Natural gate阈值未被降低。

---

## 9. v48.14四组消融

| 组 | Dedicated target adaptation | Dynamic hard-harm | Same-group counterfactual |
|---|---:|---:|---:|
| A dedicated recalibration only | 否 | 否 | 原checkpoint |
| B target adaptation | 是 | 否 | 否 |
| C hard-harm adaptation | 是 | 是 | 否 |
| D full PRISM | 是 | 是 | 是 |

优先判断：

1. A→B：数据合同适配本身是否提高Contact harm AUC和fit→verify迁移；
2. B→C：false-safe harmful mining是否降低conditional harmful UCB；
3. C→D：same-group pair是否还有增益，若无则以后移除；
4. 只有D或C在独立certificate pool产生正平均teacher advantage的非零coverage，才进入stress closed-loop。

四组在每个variant wave内同时运行：GPU0跑A/C，GPU1跑B/D；每张卡两个任务。

---

## 10. 不能承诺“保证gate通过”

Natural gate是统计安全证书，任何未运行的算法都不能科学地保证通过。v48.14做的是修复已被结果证明的根因：

- 用错分布训练evidence；
- false-safe harmful样本权重不足；
- adaptation与certificate scene未隔离；
- Safe和gamma流程错误；
- ablation与NLL参数工程错误。

若v48.14仍失败，新的分层消融会明确判断失败属于：

- dedicated adaptation没有改善；
- hard-harm有效但support仍不足；
- same-group loss无效；
- 或proposal在dedicated分布上实际退化。

这比继续盲目增加loss或放宽Natural gate更能支持下一轮算法决策。
