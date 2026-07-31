# OC-RAP v48.24 结果审计与 v48.25 INTEGRITY-BRIDGE 优化报告

## 0. 结论摘要

### RC=30 不代表 v48.24 算法已经被证明退化

本轮 Balanced、Precision 均完成训练并生成 checkpoint。RC=30 发生在 certificate 控制器阶段，不是训练崩溃。四个 Near/Contact certificate 文件均非空、scene-disjoint、完成了 gate 计算，但 v48.24 将“结构支持不足”这一合法的 Natural-gate 拒绝错误地编码成 worker `4`，再由 controller 映射为 `RC=30`。

正确语义应为：

- `RC=0`：pipeline 有效且 Natural gate 通过；
- `RC=20`：pipeline、数据与 certificate 工件有效，但结构支持或 learned selector 被 gate 拒绝；
- `RC=30`：空数据、损坏工件、协议/索引错误、训练/checkpoint 失败或异常。

因此 v48.24 的 RC=30 首先是工程返回码错误，而不是算法性能结论。

更关键的是，v48.24 不是对预期 SUPPORT-BRIDGE 的有效实验：脚本设置了 semantic harm prior 与 frontier 模式，但 `train.py` 构造 `OCRAPModel` 时漏传了这两个字段。实际模型仍运行旧逻辑，零风险 residual 被解释为约 50% harmful，admission 仍受到 `softplus(0)≈0.693` 的固定负偏置。该错误足以产生全 abstain。

### 不应删除 certificate

certificate 不是模型内部的“负担”，而是禁止未经统计授权的 selector 读取 held-out test/stress 的安全与科研可信度边界。删除 certificate 只会隐藏失败，不会解决 Near/Contact 的收益—风险排序问题。

需要修改的是 certificate 协议：旧协议把稀缺 safe-positive 样本再拆成 fit/verify 两半，造成 fit 半边结构不可行。v48.25 改为：

1. 只在 `evidence_adapt_dev` 拟合 opportunity/harm/score/rank 阈值；
2. 冻结 rule 和 SHA256；
3. 在完整、scene-disjoint 的 certificate population 上只做 verification；
4. certificate 标签不参与阈值选择；
5. 原 verify 数量、precision LCB、harm UCB 要求不下调。

这是一套新版本协议，不能事后重写 v48.24。当前 certificate 已经被多轮审阅，因此下一轮在其上的结果只能作为开发证据；最终 CCF-A 投稿结论仍应使用新封存/预注册的 certificate population。

---

## 1. RC=30 的直接原因

### 1.1 训练阶段正常结束

上传结果中：

- Balanced adaptation exit code = 0；
- Precision adaptation exit code = 0；
- 两个 `best.pt` 均存在；
- `PIPELINE_FAILED.json` 指向 `stage=certificate`；
- `test_roots_read=false`。

所以 RC=30 不是模型训练、CUDA 或 checkpoint 失败。

### 1.2 四个 certificate worker 都因结构支持不足返回 4

Balanced 与 Precision 的 Near/Contact 结果中：

- `num_groups > 0`；
- `num_scenes > 0`；
- `gate_evaluated=true`；
- `certificate_data_valid=true`；
- fit-side `proposal_constrained_oracle_gate.feasible=false`。

v48.24 的 `calibrate_policy_risk_v48.py` 将 `support_feasibility.overall=false` 返回为 4；certificate shell 又将任一 4 映射为 30。这把算法/数据合同拒绝错误包装成工程失败。

v48.25 已改为：

```text
结构支持不足或 learned gate 拒绝 -> worker 3 -> controller RC=20
空数据/损坏/协议异常             -> worker 4 -> controller RC=30
```

### 1.3 状态文件之间存在语义矛盾

v48.24 同时写出了：

- 根目录 `PIPELINE_FAILED.json`；
- variant certificate 中 `gate_evaluated=true`、`certificate_data_valid=true`；
- `learning_gates_v48_24.json` 中又存在与 pipeline failure 不一致的状态字段。

v48.25 统一状态语义：只要训练、数据、certificate 工件都有效，即使 gate 拒绝也写 `pipeline_valid=true`、`gate_evaluated=true` 并返回 20。

---

## 2. 会污染算法归因的工程错误

## 2.1 核心模型配置没有传入 `OCRAPModel`

v48.24 shell 设置：

```text
model.direct_recovery_evidence_frontier=true
model.direct_recovery_evidence_component_prior_logit=-2.0
```

但 v48.24 `src/ocrap/cli/train.py` 的 `OCRAPModel(...)` 构造器漏掉了：

```text
direct_recovery_evidence_frontier
direct_recovery_evidence_component_prior_logit
```

结果是模型使用默认 `frontier=false`。因此预期的：

```text
admission = benefit - [softplus(harm)-softplus(harm_prior)] + residual
```

没有实际执行，而是退回旧的非 centred 路径。脚本、日志中的配置值不能证明模型实例真正启用了对应功能。

这意味着：

> v48.24 的结果不能用于否定 semantic low-risk prior、identity-preserving frontier 或 direct safe-utility 设计，因为这些模块没有按计划接入实际模型。

v48.25 已补齐三个构造参数，并加入 AST 回归测试，防止再次出现“命令设置存在、模型运行时丢失”的 silent configuration drop。

## 2.2 validation 错误复用了训练 teacher index

v48.24 训练日志显示：

```text
train group index rows = 10015
validation group index rows = 10015
validation groups = 409
validation strata:
  harmful_only = 0
  beneficial = 0
  dead_or_mixed = 409
```

训练 index 的 path key 对应 `evidence_adapt_train_*`，无法匹配 `evidence_adapt_dev_*` 的 NPZ path。validation sampler 因而把全部 dev group 归为 dead/mixed，破坏了 checkpoint 选择的 Near/Contact 正机会分层。

v48.25：

- 独立构建 `evidence_adapt_dev_teacher_pcd_index.jsonl`；
- 通过 `training.validation_group_index_path` 传入；
- 若未提供 dev index，则关闭 validation 的 exact stratification，而不是静默使用错误 index。

## 2.3 checkpoint metric 允许“全 abstain”获胜

两个 v48.24 best checkpoint 在 adaptation-dev 都满足：

```text
direct_raw_admission_rate_near = 0
direct_raw_admission_rate_contact = 0
direct_positive_admission_recall_near = 0
direct_positive_admission_recall_contact = 0
```

但 `direct_frontier_selection_risk` 仍可因 soft safe mass/soft recall 改善而下降。也就是说，checkpoint metric 衡量了概率质量，却没有强制最终可执行 admission 非零。

v48.25 新增 `direct_integrity_selection_risk`：

```text
integrity risk
= frontier risk
+ 20 * Near/Contact hard positive-recall shortfall
+ 8  * all-abstain indicator
```

它不改变 held-out Natural gate，只负责避免训练/dev early stopping 选择一个永远不执行 recovery 的 checkpoint。

## 2.4 数据 profile 中的 `safe_positive_fraction=0` 是误导字段

该字段来自文件路径名启发式，不是 exact teacher composite label。真实 teacher index 中：

- safe-beneficial groups = 52；
- safe-beneficial scenes = 24；
- Near safe-beneficial groups = 11；
- Contact safe-beneficial groups = 41。

v48.25 将旧字段改名为 `legacy_safe_root_positive_fraction`，并令 `safe_positive_fraction=null`；exact 数值只由 teacher index 报告。

## 2.5 dev shadow 的 selector 来源需要与新 certificate 协议兼容

v48.25 在 adaptation-dev 上冻结 rule。若完整 certificate 拒绝部署，正式 deployment 仍禁止运行；但 `DEV_SHADOW_DIAGNOSTIC=1` 可以只在 adaptation-dev 上使用同一个 dev-frozen rule 做物理归因。该路径仍明确禁止 certificate/test/stress 输入。

---

## 3. “top-3 raw-benefit 高召回”假设的本轮验证

v48.24 的 top-k support curve 给出了决定性结果。

### Near fit

| proposal k | safe-positive groups |
|---:|---:|
| 1 | 3 |
| 3 | 3 |
| 5 | 3 |
| 8 | 3 |

### Contact fit

| proposal k | safe-positive groups |
|---:|---:|
| 1 | 7 |
| 3 | 10 |
| 5 | 10 |
| 8 | 10 |

结论：

- Near 中 top-1 已包含全部 fit safe-positive；扩大到 top-8 没有任何增量；
- Contact 从 k=1 到 k=3 有增量，但 k=3 后没有增量；
- v48.24 的 top-8 设计假设被否定；
- 当前瓶颈不是 proposal width，而是 safe label 支持、Evidence 排序、admission 边界与 certificate fit 协议。

因此 v48.25 默认恢复 top-3，避免引入更多 harmful/ambiguous action 和更大的排序难度。

---

## 4. certificate 是否导致 gate 一直无法通过

需要区分三层：

### 4.1 certificate 揭示了真实结构问题，但不是问题制造者

旧 Near certificate：

- fit safe-positive groups = 3；
- verify safe-positive groups = 6；
- Near fit 要选择至少 10，precision LCB ≥ 0.50；
- optimistic oracle 只能得到 3/10，LCB = 0.1538。

旧 Contact certificate：

- fit safe-positive groups = 10；
- 需选择至少 16，precision LCB ≥ 0.50；
- optimistic oracle 得到 10/16，LCB = 0.4652。

这证明旧 fit 半边本身不可行。certificate 没有“压坏模型”，而是暴露了预注册合同与样本支持不匹配。

### 4.2 旧内部 fit/verify 拆分浪费了稀缺正机会

完整 certificate 中，top-3 约有：

- Near safe-positive groups = 3 + 6 = 9；
- Contact safe-positive groups = 10 + 10 = 20。

若完整独立 population 只做 verify：

- Near 至少可由 oracle 选择 8 个全正机会，90% one-sided Wilson LCB 约 0.8297；
- Contact 可选择 10 个全正机会，LCB 约 0.8589。

这只是忽略 macro concentration 的 optimistic structural check，但说明完整 certificate population 并非天然不可行。问题在于旧协议把极少的 Near 正机会又拆成两套都要单独满足的门槛。

### 4.3 v48.25 的修改不是“降低 gate”

v48.25 只改变阈值来源和 certificate population 使用方式：

- 阈值在 adaptation-dev 拟合；
- 完整 certificate 只验证；
- verify 的最小选择数、precision LCB 和 harm UCB 不变；
- certificate label 不参与阈值搜索。

所以它不是为了让结果更容易过，而是让 threshold selection 与 independent verification 的统计角色更清楚，并避免把稀缺证书数据一半用于调参。

---

## 5. v48.24 是否解决了 Near-contact

### 5.1 Balanced Near

| 指标 | v48.23 | v48.24 | 判断 |
|---|---:|---:|---|
| candidate benefit AUC | 0.8514 | 0.8555 | 极小正向 |
| learned benefit AUC | 0.8261 | 0.8285 | 极小正向 |
| learned safe-benefit AUC | 0.8319 | 0.8203 | 下降 |
| learned harm AUC | 0.3642 | 0.3501 | 下降 |
| conditional harm AUC | 0.6692 | 0.4975 | 明显下降至随机附近 |
| learned correlation | -0.0798 | -0.1011 | 下降 |
| positive top-1 regret | 0.0054 | 0.1290 | 明显恶化 |
| fit/verify selected | 0/0 | 0/0 | 未形成 admission |

Balanced Near 只保留 raw-benefit 可分性，没有将其变成安全排序与准入。

### 5.2 Precision Near

| 指标 | v48.23 | v48.24 | 判断 |
|---|---:|---:|---|
| candidate benefit AUC | 0.7891 | 0.7374 | 下降 |
| learned benefit AUC | 0.7729 | 0.7265 | 下降 |
| learned safe-benefit AUC | 0.7915 | 0.7160 | 下降 |
| learned harm AUC | 0.5928 | 0.5582 | 下降 |
| conditional harm AUC | 0.5275 | 0.6106 | **局部正向** |
| nonpositive false switch | 0.2210 | 0.1530 | **局部正向** |
| harmful switch | 0.4923 | 0.6222 | 明显恶化 |
| positive top-1 regret | 0.0909 | 0.2024 | 明显恶化 |
| fit/verify selected | 0/0 | 0/0 | 未形成 admission |

Precision Near 的确存在部分有效信号：高机会条件下的风险区分和非正机会误切换有所改善。但真正的 action 选择更差——harmful switch 和 regret 上升，最终仍零 admission。因此不能说 Near 问题已解决。

### 5.3 与投稿目标的差距

上一轮目标要求：

- minimum clearance 至少提升约 0.10 m；
- minimum TTC 至少提升约 0.20 s；
- near-contact 与 critical-TTC exposure duration 降低；
- clearance/TTC deficit AUC 降低；
- PCD ≥ 0.54、FRA ≤ 0.12、DRS ≥ 0.88、bounded NUP ≥ 0.995；
- intervention rate ≤ 0.02，episode rate ≤ 0.012，run ≤ 1；
- route progression、jerk、yaw-rate 不劣化。

v48.24 没有成功运行可用的 dev-shadow closed loop，因此这些物理目标全部“未验证”，而不是“已达到”。离线层面又是零 admission，所以距离 CCF-A 主结果至少还缺：可执行准入、物理收益、低干预和置信区间四层证据。

---

## 6. v48.24 是否解决了 Contact

### 6.1 Balanced Contact

| 指标 | v48.23 | v48.24 | 判断 |
|---|---:|---:|---|
| candidate benefit AUC | 0.5507 | 0.5411 | 下降 |
| learned benefit AUC | 0.3840 | 0.3964 | 小幅上升但仍低于随机有效水平 |
| learned safe-benefit AUC | 0.5138 | 0.4921 | 下降 |
| learned harm AUC | 0.3973 | 0.3543 | 下降 |
| conditional harm AUC | 0.5218 | 0.4950 | 下降至随机附近 |
| learned correlation | -0.2407 | -0.2619 | 下降 |
| regret | 0.1824 | 0.2118 | 恶化 |

Balanced Contact 没有形成可用的收益或风险排序。

### 6.2 Precision Contact

| 指标 | v48.23 | v48.24 | 判断 |
|---|---:|---:|---|
| candidate benefit AUC | 0.5611 | 0.5483 | 下降 |
| learned benefit AUC | 0.4217 | 0.4140 | 下降 |
| learned safe-benefit AUC | 0.4959 | 0.4496 | 下降 |
| learned harm AUC | 0.6496 | 0.6166 | 下降 |
| conditional harm AUC | 0.5023 | 0.4431 | 下降 |
| nonpositive false switch | 0.2773 | 0.2446 | 小幅正向 |
| harmful switch | 0.4272 | 0.4579 | 恶化 |
| regret | 0.1060 | 0.2637 | 大幅恶化 |
| fit/verify selected | 0/0 | 0/0 | 未形成 admission |

Precision Contact 仅 false-switch 有轻微改善，核心 benefit、safe-benefit、harm、conditional harm、correlation 与 regret 均不支持有效改善。

### 6.3 Contact 的算法缺陷

当前 Contact teacher 的核心连续目标仍主要是一个 scalar PCD advantage。撞后恢复需要同时表达：

- secondary/re-contact；
- overlap duration 与 longest overlap run；
- post-contact clearance/free-space AUC；
- sustained escape 与 time-to-escape；
- stable stop；
- route、offroad、jerk、yaw-rate 约束。

一个 scalar PCD 可以提供方向，但无法充分区分“短期 PCD 上升却延长 overlap”与“真实创造逃逸空间”的 action。当前模型没有直接看到上述 Contact recovery decomposition 的辅助监督，所以 candidate benefit AUC 接近随机、learned correlation 为负、regret 较高是可解释的。

v48.25 主轮先修复工程完整性，避免同时改变 teacher 标签。若修复后 Contact 仍低于目标，下一版本应增加不暴露 regime ID 的 event-conditioned recovery auxiliaries，例如从通用候选特征预测：overlap reduction、free-space gain、escape probability、stable-stop probability，再由统一 safe utility 聚合；不应简单加入 Contact router。

### 6.4 与投稿目标的差距

上一轮目标要求：

- PCD ≥ 0.52、FRA ≤ 0.16、DRS ≥ 0.84、bounded NUP ≥ 0.985；
- intervention rate ≤ 0.04、episode rate ≤ 0.025、run ≤ 2；
- secondary/re-contact scene rate 至少降低约 0.02；
- overlap duration/run 降低；
- post-contact clearance/free-space AUC 提升；
- sustained escape 提升、time-to-escape 缩短；
- stable-stop rate 至少提升约 0.02；
- offroad、route progression、jerk、yaw-rate 受控。

当前没有可用的物理 paired 结果，且离线排序仍弱，因此尚未接近可投稿主张。

---

## 7. 模型、数据集和 macro action 的责任划分

### 7.1 当前首要原因：工程与模型 admission 设计

本轮最直接的问题是配置未接入、validation index 错误和 all-abstain checkpoint metric。这些错误会在任何数据上阻止 safe admission，必须先修复后再评价算法。

### 7.2 数据集：稀疏但不是完全不可学习

训练 teacher index：

| Regime | deployable candidates | safe-beneficial candidates | safe-beneficial groups | scenes | harm prevalence |
|---|---:|---:|---:|---:|---:|
| Near | 1425 | 25 | 11 | 7 | 0.540 |
| Contact | 4086 | 106 | 41 | 17 | 0.454 |

Near 尤其稀疏且 scene concentration 高，泛化方差必然较大。但仍有 11 个 group、25 个 candidate，不是零支持；先修复训练/验证完整性是合理顺序。

### 7.3 macro action：不是“缺动作”，而是语义与标签合同仍可能过严

k=3 到 k=8 不增加 safe-positive，说明问题不是简单扩大 macro 候选数量。当前 action set 中已经存在 proposal-contained safe actions。

但 component veto 将 DRS、deployability、gap、hard violation、harm proxy 任一分量超过 0.05 都视为 harmful。对 brake/stabilize 等恢复动作，轻微 soft-component 退化可能换来明显 overlap/escape 改善。将所有分量都作为不可补偿 hard veto，可能造成 safe-positive 极度稀疏。

本轮不直接改标签，以免在工程错误尚未排除时混合归因。后续可做“protective Pareto envelope”预注册消融：

- collision/offroad/hard-rule 继续不可补偿 veto；
- DRS/deployability/gap 作为有上下界的 Pareto 非劣约束；
- 只有真实 post-contact escape/stable-stop 获益时允许小幅 soft trade-off。

该标签修改必须新版本化，不能在看过 certificate 后直接调到能过 gate。

---

## 8. v48.25 INTEGRITY-BRIDGE

### 8.1 工程层

1. 修复 frontier/prior/admission config 传递；
2. adaptation-dev 独立 teacher index；
3. 正确 RC=20/30；
4. dev-frozen rule 与 full certificate verification；
5. shadow diagnostic 与正式 deployment 分离；
6. runtime 加载完整 selector contract；
7. 新增 Near/Contact target checker；
8. 新增回归测试。

### 8.2 算法层

1. 默认 top-k 恢复为 3；
2. semantic component prior `-2.0` 真正生效；
3. centred identity-preserving admission 真正生效；
4. admission residual 主模型取消 `tanh` 上界，但保持零初始化；
5. `direct_integrity_selection_risk` 阻止 all-abstain checkpoint；
6. safe-utility regression/listwise 保留；
7. nominal+top-k categorical objective 保留；
8. legacy Noisy-OR 继续关闭；
9. stronger benefit listwise/frontier contrast 只在 D 消融与主模型中统一验证。

### 8.3 四个消融的判读

| 组 | 变化 | 归因问题 |
|---|---|---|
| A_wiring_fix_bounded | 只修 config，保留 bounded admission/旧 metric | v48.24 是否主要败于 silent wiring bug |
| B_add_integrity_checkpoint | A + hard admission checkpoint barrier | all-abstain early stopping 是否主要瓶颈 |
| C_add_unbounded_admission | B + unbounded residual | bounded residual 是否无法跨过 nominal boundary |
| D_full_integrity_bridge | C + benefit listwise + stronger frontier | 连续收益与安全前沿能否在同一模型协同 |

关键判读：

- A 明显恢复 risk/admission：v48.24 的主要问题是工程配置；
- B>A：checkpoint all-abstain barrier 有效；
- C>B：bounded admission ceiling 是主要阻塞；
- D>C：连续 ranking/frontier 产生额外收益；
- 离线改善但 shadow 不改善：teacher/feature 与物理闭环错位；
- dev-shadow 改善但 full certificate RC=20：有限样本/域差异或 calibration generalization 问题；
- full certificate oracle 可行但 learned gate 失败：模型排序/admission 仍不足；
- full certificate structural oracle 失败：标签/宏动作/证书支持合同仍不兼容。

---

## 9. 下一轮结果的最低验收条件

### 工程有效性

必须同时满足：

```text
pipeline_valid=true
gate_evaluated=true
certificate_data_valid=true
test_roots_read=false
model construction audit: frontier/prior/bounded fields present
validation teacher index root = evidence_adapt_dev
```

### 离线 Near

至少要求：

- learned safe-benefit AUC 不低于 candidate safe-benefit AUC 太多；
- conditional harm AUC 明显 > 0.5；
- correlation 转正；
- positive top-1 regret 显著低于 v48.24；
- harmful switch 明显低于 0.62；
- adaptation-dev positive admission recall > 0；
- certificate verify 有非零 selections。

### 离线 Contact

至少要求：

- learned benefit/safe-benefit AUC > 0.5；
- correlation 转正；
- regret 显著低于 0.264；
- conditional harm AUC > 0.55；
- harmful switch 下降；
- certificate verify 非零 selections。

### 物理闭环

RC=20 时先运行 adaptation-dev shadow，按照 `check_v48_25_regime_targets.py` 输出定位物理差距。只有 RC=0 自动生成授权文件后才运行 held-out stress。

---

## 10. 验证状态

本地完成：

```text
python -m compileall -q src tools tests                     PASS
find scripts ... | xargs bash -n                            PASS
PYTHONPATH="$PWD/src" python -m pytest -q                    225 passed, 5 warnings
python tools/check_v48_25_regime_targets.py --help          PASS
```

当前环境没有真实 WOMD/Waymax 数据和两张 A30，因此未声称 v48.25 已通过 gate，也未虚构 Near/Contact 闭环收益。
