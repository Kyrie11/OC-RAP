# OC-RAP v48.12结果审计与v48.13 TERRA设计

## 1. 最终结论

v48.12没有通过Near与Contact联合Natural gate，因此没有可归因于OC-RAP的stress closed-loop结果。失败不是单纯的阈值问题，而是两个不同层次的问题：

1. **Near：候选排序存在整体相关性，但无法稳定锁定精确top-1。** 三seed下Near candidate-rank correlation约0.13–0.15，但group top-1 correlation均值仍为负。
2. **Contact：收益信号较强，但伤害证据跨scene迁移失败。** Contact candidate-positive AUC约0.82，policy-top1 benefit AUC约0.79–0.82；然而harm AUC约0.51，fit规则到verify后出现高harm和负平均teacher advantage。

Natural gate拒绝当前策略是正确的保护行为。不能通过放宽precision、harm或support门槛制造表面coverage。

本轮还发现一个影响v48.12主实验归因的工程错误：主实验calibration没有继承staged training写入的policy-first/no-fallback合同，而三seed复校准和消融使用了正确合同。因此，**主实验4801的规则结果不能与三seed结果直接混用**；本报告的主要算法结论以正确合同下的三seed与完整消融为准。

---

## 2. v48.12正确合同下的三seed表现

### 2.1 排序和证据

| Variant | Regime | Candidate正收益AUC | Candidate排序相关性 | Group top-1相关性均值 | Policy-top1收益AUC | Policy-top1伤害AUC |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.672 | 0.145 | **-0.054** | 0.684 | 0.568 |
| Balanced | Contact | 0.822 | 0.214 | **0.077** | 0.792 | **0.512** |
| Precision | Near | 0.693 | 0.134 | **-0.035** | 0.678 | 0.559 |
| Precision | Contact | 0.814 | 0.195 | **0.101** | 0.825 | **0.513** |

结论：

- Candidate特征不是完全无效，尤其Contact的收益识别已经较强。
- Near的候选整体顺序与teacher有弱相关，但精确top-1不稳定，说明teacher winner存在弱可辨识性和近似并列。
- Contact top-1方向已经转正，但仍低于内部投稿准备门槛0.20。
- Contact最大瓶颈不是benefit，而是harmful-vs-dead证据几乎随机。

### 2.2 Natural gate

Near三个seed、两个variant均为0 verify selection。代表性Near near-miss在verify中可以达到：

- 9–11个选择；
- precision约0.36–0.55；
- 0个harmful selection；
- recall约0.16–0.24；
- 平均teacher advantage为正。

这说明Near存在“稀疏且可能安全”的策略雏形，但支持量、precision置信下界、recall和跨seed稳定性均不足。

Contact在seed 4801能产生非零选择，但fit到verify发生严重崩溃。例如Balanced：

| 指标 | Fit | Verify |
|---|---:|---:|
| 选择数 | 21 | 24 |
| Precision | 0.810 | 0.333 |
| Harmful rate | 0.048 | 0.500 |
| Positive recall | 0.515 | 0.242 |
| 平均teacher advantage | +0.177 | **-0.204** |

Precision分支表现相同：fit precision 0.80、harm 0.10、平均收益+0.167；verify变为precision 0.50、harm 0.438、平均收益−0.139。

因此Contact gate失败的根因是**证据排序不能跨独立scene迁移**。

---

## 3. v48.12消融归因

### 3.1 Recovery pairwise没有起效

| 消融 | Near top-1（Bal/Prec） | Contact top-1（Bal/Prec） |
|---|---:|---:|
| A Contract fix | 0.055 / 0.021 | 0.072 / 0.084 |
| B + Recovery pair | **-0.048 / -0.007** | 0.057 / 0.091 |

Recovery pair loss明显损害Near，对Contact只有微小且不稳定的变化。原因包括：

- 同一group内大量候选teacher差距很小，精确pair标签噪声高；
- O(K²)相关pair放大单个group的梯度；
- 直接优化所有清晰pair仍不等价于稳定召回一个可接受候选。

结论：v48.13主实验关闭v48.12 all-pairs排序，不再重复该修改。

### 3.2 Bipolar cross-group evidence只有局部收益

C消融在部分Near分支提升harm AUC，例如Precision Near从约0.563升至0.642，但Contact harm AUC最高仍约0.528，且verify选择平均teacher advantage依然为负。

根因是跨scene pairwise比较仍可利用scene严重度、采样构成和macro捷径；同时minibatch pair集合不稳定。

结论：保留有序三状态证据，但将主对比改为**同一scene-time proposal内部的counterfactual比较**。

### 3.3 Opportunity-normalized macro约束是正确工程改进

该约束避免把数据本身约88%的macro-5正机会集中误判为模型捷径。但当前规则在precision和harm迁移阶段已经失败，macro约束不是首要瓶颈。该设计继续保留。

---

## 4. 工程层面发现的问题与修复

### 4.1 主实验丢失staged policy合同

v48.12主结果JSON中：

```text
conditional_recovery_ranking = false
policy_first_no_fallback = false
```

而三seed和消融为true。原因是child staged script内部导出的环境变量在进程退出后不会返回parent controller。

v48.13修复：

- 每个variant写入`POLICY_CONTRACT.env`；
- controller在calibration前显式source；
- multi-seed与dedicated calibration同样读取该合同；
- calibration JSON保存proposal与selector overrides。

### 4.2 Stage-E checkpoint选择与拟部署策略不一致

v48.13最初设计需要在top-k proposal内用evidence rerank，但旧validation metric仍只评价rank top-1。这样best epoch不一定是最终策略最好的epoch。

已修复：validation/checkpoint现在按照相同合同：

```text
rank top-k proposal
→ evidence rerank
→ evidence margin与准入
→ certificate regret/harm/recall
```

### 4.3 Multi-seed与主实验gate参数不完全一致

旧multi-seed脚本没有显式传递conditional harmful-selection UCB参数。v48.13统一主实验、多seed和dedicated calibration的所有风险约束。

### 4.4 历史回归测试文件缺失

当前上传代码缺少`run_v47_two_gpu_fast_commands.txt`，导致一个历史回归测试无法执行。已从上一版完整代码恢复，最终全套157项测试通过。

---

## 5. 外部baseline对投稿目标的约束

### 5.1 Safe

Nominal/log replay在外部Safe离线结果中达到：

- DRS=1；FRA=0；ODG=0；bounded NUP=1；
- intervention=0；yaw violation=0。

因此Safe不是“争取更高恢复分数”，而是必须证明严格nominal非劣。当前OC-RAP没有新的gate-authorized paired Safe结果，不能声称达到Safe投稿目标。

### 5.2 Near-contact

`predictive_safety_filter`提供了最有竞争力的综合离线基线：

- DRS 0.973；FRA_exec 0.127；ODG 0.174；
- bounded NUP 0.988；intervention 0.446；
- route rejoin 0.996；stable stop 0.884。

OC-RAP当前没有合法closed-loop结果，尚不能声称优于该基线。后续论文优势应体现为：在更低干预或更低FRA/ODG下，提高clearance、TTC和恢复成功率。

### 5.3 Contact

50-scene closed-loop基线显示：

- `postimpact_mpc_lite` DRS约0.527，但intervention约0.976、NUP约0.454；
- `post_collision_restoration` DRS约0.494、intervention约0.833、NUP约0.807。

因此OC-RAP最有潜力的论文定位不是单纯追求最大恢复，而是：

> 接近恢复型MPC/修复策略的恢复能力，同时显著降低过度干预和NUP损失，并用可验证的选择性证书控制harm。

当前v48.12尚未产生支持这一主张的闭环证据。

---

## 6. 与三个regime投稿目标的差距

### Safe

尚未验证collision/offroad非增、paired 95% CI、route progression、jerk/yaw、intervention episode。需要至少约100个paired scenes并输出完整scene级指标。

### Near-contact

| 离线策略指标 | 当前 | 内部准备目标 |
|---|---:|---:|
| Group top-1 corr | -0.054 / -0.035 | ≥0.20 |
| Policy benefit AUC | 0.678–0.684 | ≥0.70 |
| Harm AUC | 0.559–0.568 | ≥0.60 |
| Verify recall | 0；near-miss 0.16–0.24 | ≥0.35 |
| Precision LCB90 | near-miss约0.18–0.32 | ≥0.60 |
| Harm UCB90 | near-miss约0.20–0.23 | 理想≤0.10 |

在Natural gate通过前，collision、clearance、TTC、exposure、DRS、PCD、FRA和ODG投稿目标均未获得合法验证。

### Contact

| 离线策略指标 | 当前 | 内部准备目标 |
|---|---:|---:|
| Group top-1 corr | 0.077 / 0.101 | ≥0.20 |
| Policy benefit AUC | 0.792 / 0.825 | ≥0.75，已达到 |
| Harm AUC | 约0.513 | ≥0.60 |
| Verify precision LCB90 | 0.20–0.31 | ≥0.60 |
| Verify harmful UCB90 | 0.64–0.66 | 理想≤0.10 |
| Verify平均teacher advantage | 负值 | 必须为正 |

最需解决的是harmful-vs-dead跨scene迁移，而不是继续加强benefit分类。

---

## 7. v48.13新算法：OC-TRAC-TERRA

**TERRA：Top-k Evidence-Reranked Recovery with Abstention。**

### 7.1 Set-valued top-k recovery proposal

不再要求模型在Near近似并列候选中精确命中唯一winner，而要求top-k proposal至少包含一个teacher可接受恢复候选。

新增诊断：

- positive-group oracle-best proposal hit；
- positive-group any-positive proposal hit；
- exact top-1 correlation仍报告，但不再是Stage-P唯一成功条件。

### 7.2 Proposal-distribution evidence

Stage E冻结tournament，并对top-k proposal中的每个候选训练有序三状态证据，采用rank-decay权重。这样所有运行时可能被执行的proposal成员都属于训练分布。

### 7.3 Same-group counterfactual evidence

在同一scene-time proposal中直接比较：

- beneficial vs non-beneficial；
- harmful vs non-harmful。

这会抵消共享scene严重度，减少Contact fit→verify反转。v48.12跨group bipolar pairwise在主实验中关闭。

### 7.4 Evidence rerank with abstention

统一合同：

```text
physical recovery set
→ frozen rank top-k proposal
→ opportunity/harm证据过滤
→ proposal内按evidence选择
→ 无通过候选则abstain
```

它不同于旧runner-up fallback：Stage E已经训练proposal内全部候选，因此不是分布外回退。

### 7.5 不能诚实保证Natural gate一定通过

TERRA修复了当前可证实的监督对象、训练分布和工程合同问题，但真实结果仍取决于数据可分性、独立scene支持量以及Contact标签漂移。Natural gate不能也不应被算法代码“保证通过”；只有满足held-out风险证据后才应通过。

---

## 8. 四组消融

| 组 | Proposal训练 | Proposal evidence | Evidence rerank | 目的 |
|---|---:|---:|---:|---|
| A_top1_contract | 关 | 关 | 关 | 正确工程合同下的top-1基线 |
| B_proposal_only | 开 | 关 | 关 | 验证top-k proposal是否提高候选覆盖 |
| C_evidence_rerank | 关 | 开 | 开 | 验证proposal分布证据与rerank |
| D_full_terra | 开 | 开 | 开 | 完整方法 |

两张A30上，四组在每个variant wave内同时运行：A/C共享GPU0，B/D共享GPU1，即每张卡两个约1GB任务。Balanced与Precision两个wave串行，以避免8任务同时争抢CPU和数据I/O。

---

## 9. 下一步判定顺序

1. Proposal gate：Near和Contact positive-group oracle-best hit≥0.75，any-positive hit≥0.90。
2. Proposal evidence：Near benefit AUC≥0.70、Contact≥0.75；harm AUC≥0.60；evidence-teacher correlation≥0.10。
3. Natural gate：非零verify coverage，平均teacher advantage为正，precision/harm置信界、recall/support和macro excess全部通过。
4. 通过1–2后固定checkpoint做4801/4802/4803。
5. 只有controller生成`NEXT_COMMANDS.txt`后才运行stress closed-loop。
6. 三套dedicated calibration全部完成后，对同一checkpoint重新校准，不重新训练。
