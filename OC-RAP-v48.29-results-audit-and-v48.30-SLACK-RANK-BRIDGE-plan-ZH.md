# OC-RAP v48.29 结果联合审计与 v48.30 SLACK-RANK-BRIDGE 优化方案

## 0. 审计范围与结论边界

本次联合审计覆盖：

- v48.29 `V48_29_COMPLETE.json`、`GATE_FAILED.json`、`GATE_FAILURE_DECOMPOSITION.json`；
- Balanced/Precision 的 factor-stage、admission-stage 训练轨迹、checkpoint 和模型合同；
- adaptation-dev 冻结 rule 与完整 certificate verification；
- 8 个 v48.29 消融任务；
- 4 组 adaptation-dev paired shadow closed-loop；
- v48.29 代码中的采样、损失、checkpoint、推理、certificate 与 closed-loop 参数传递。

没有读取 held-out test/stress。v48.29 的 `test_roots_read=false`，因此以下结论只用于开发和算法归因。

## 1. v48.29 的 RC=20 是否仍是 development-rule fitting failure

是。v48.29 的返回状态为：

- `pipeline_valid=true`；
- `gate_evaluated=true`；
- `gate_passed=false`；
- `raw_certificate_exit_code=20`；
- `test_roots_read=false`。

四个 Near/Contact 分支仍统一被标记为：

```text
development_rule_fit_rejection
```

但本轮失败形态相较 v48.28 已发生变化：hardest-negative 提高了 adaptation-dev 上的部分安全机会召回，却同时显著抬高了 recovery admission 先验，造成 development rule 无法同时控制 precision、harm 与 recall，并在 certificate 上进一步出现过度准入和 harmful 选择。

### 1.1 Proposal 与 certificate 不是数学不可行

完整 certificate 的 top-3 proposal-constrained oracle 仍然可行：

| Regime | Certificate groups | safe-positive groups | Oracle precision LCB |
|---|---:|---:|---:|
| Near | 290 | 9 | 0.8457 |
| Contact | 764 | 20 | 0.9241 |

因此目前不应：

- 删除 certificate；
- 降低 Natural-gate 要求；
- 把 top-3 扩展为 top-8；
- 把失败归因于候选集合完全缺少安全动作。

当前 gate 的作用仍然合理：它阻止了一个在 adaptation-dev 和 certificate 上仍会选择大量 harmful action 的 learned selector。

## 2. 主实验的实际失败位置

### 2.1 Balanced Near

adaptation-dev 最近 rule：

- 选择 20 个；
- safe-positive 5 个；
- precision 0.250，90% LCB 0.148；
- harmful 4 个，harmful UCB 0.335；
- safe-positive recall 0.625；
- selected teacher advantage 均值仅 `+0.0018`。

certificate：

- 选择 43 个；
- safe-positive 3 个；
- precision 0.0698，LCB 0.0342；
- harmful 22 个；
- recall 0.333；
- teacher advantage 均值 `-0.1485`。

这说明 v48.29 的 hardest-negative 确实增加了开发集安全机会命中，但并没有学出可迁移的安全排序；进入自然总体后，恢复动作被整体抬高，导致大量 false/harmful admission。

### 2.2 Precision Near

adaptation-dev：

- 选择 9 个；
- safe-positive 3 个；
- precision 0.333，LCB 0.172；
- harmful 2 个；
- recall 0.375；
- teacher advantage `+0.1485`。

这是四个分支里最明确的局部正向区域，说明 Precision Near 已部分学会识别有正收益的安全 action。

但 certificate：

- 选择 10 个；
- safe-positive 1 个；
- harmful 5 个；
- precision 0.10，LCB 0.0304；
- recall 0.111；
- teacher advantage `-0.1716`。

所以它仍未形成 safe admission。问题不是完全没有排序信号，而是该信号对场景分布和 hard-negative 重采样非常敏感。

### 2.3 Balanced Contact

adaptation-dev：

- 选择 18 个；
- safe-positive 4 个；
- precision 0.222，LCB 0.123；
- harmful 6 个；
- recall 0.235；
- teacher advantage `-0.0556`。

certificate：

- 选择 38 个；
- safe-positive 1 个；
- harmful 16 个；
- recall 0.05；
- teacher advantage `-0.2121`。

Contact 的 learned safe-positive AUC 只有约 0.402，correlation 为 -0.271，说明组内 action ranking 仍明显错误。

### 2.4 Precision Contact

adaptation-dev：

- 选择 11 个；
- safe-positive 1 个；
- harmful 1 个；
- precision 0.091；
- recall 0.0588；
- teacher advantage `+0.0102`。

相较 v48.28，harmful 数量从 2 降到 1，selected advantage 从负值变为略正，这是一个很小的正向信号。

certificate：

- 选择 24 个；
- safe-positive 0；
- harmful 10 个；
- recall 0；
- teacher advantage `-0.1684`。

因此 Contact 仍完全没有证明 safe admission。

## 3. v48.29 相较 v48.28 的真实增益与代价

### 3.1 有效迹象

v48.29 的 hardest-negative 对 Near 的 adaptation-dev recall 有明确影响：

- Balanced Near：0.25 → 0.625；
- Precision Near：0.25 → 0.375。

Precision Near 的 development precision LCB 从约 0.078 提高到 0.172，selected teacher advantage 从约 0.117 提高到 0.149。

Precision Contact 的 development harmful 数从 2 降至 1，selected advantage 从 -0.0356 变为 +0.0102。

这些结果支持继续保留：

- raw-benefit 与安全准入分离；
- 五个不可补偿风险因子；
- factor→admission 两阶段训练；
- best-safe-vs-hardest-negative 的组内监督；
- top-3 frozen proposal；
- categorical one-action policy；
- bounded admission；
- legacy Noisy-OR 关闭。

### 3.2 反向作用

v48.29 在 certificate 上普遍产生更多恢复动作：

- Balanced Near：15 → 43；
- Balanced Contact：18 → 38；
- Precision Near：7 → 10。

但新增选择主要不是安全机会，而是 harmful 或非正收益动作。Balanced Near harmful 从 11 增到 22；Balanced Contact harmful 从 4 增到 16。

所以 hardest-negative 本身不是错误，错误在于它与严重失真的 stage-2 训练分布组合后，把“局部排序约束”变成了“恢复动作高先验”。

## 4. RC=20 的更根本原因：stage-2 population prior shift

v48.29 训练 index 中：

- 总 scene-time groups：1167；
- safe-beneficial groups：52；
- 自然 prevalence：`52 / 1167 = 4.46%`；
- component-harmful groups：1098。

但 admission stage 的 sampler 强制：

```text
safe-positive = 50%
harmful = 30%
dead/mixed = 20%
replacement = true
```

safe-positive group 被放大约：

```text
0.50 / 0.0446 ≈ 11.2 倍
```

而且稀少的 52 个正组通过 replacement 被反复重复。训练没有 importance correction 将其还原为自然总体先验。

这造成三个直接后果：

1. categorical/setwise objective 在训练 batch 中经常看到“应选择 recovery”的场景；
2. hardest-negative 能提高正组内部排序，但同时不断推高 recovery-vs-nominal 分数；
3. 模型学到的是重采样分布下的高干预策略，而不是自然 population 中稀疏、选择性的 recovery policy。

训练轨迹也支持这一判断。Balanced stage-2 从 epoch 1 到 epoch 7：

- Near soft harmful mass：约 0.236 → 0.341；
- Contact soft harmful mass：约 0.245 → 0.354；
- Near soft false admission：约 0.447 → 0.778；
- Contact soft false admission：约 0.472 → 0.822。

Precision 也呈相同趋势。现有 checkpoint 选择 epoch 1，避免了更晚的进一步恶化，但 epoch 1 本身已经在错误的 50% 正组先验上训练。

因此，v48.29 的根本失败不是“certificate 太严格”，而是：

> admission stage 改变了任务先验，却要求 certificate 在自然总体上证明低干预、高 precision 和低 harmful；训练分布与部署/证书分布不一致。

## 5. 消融如何支持这一归因

### A：risk-centered reference

风险被同时作为软惩罚和硬 veto，覆盖有限，但仍有大量 harmful；说明单纯加强风险惩罚不能获得可靠 safe admission。

### B：veto-decoupled benefit-only

Balanced Near recall 提高到 0.444，但选择 69 个、harmful 38 个。它证明收益信号可被释放，也证明完全取消连续安全边界后会严重过度准入。

### C：加 hardest-negative

Balanced Near 的选择从 69 降至 19，harmful 从 38 降至 11，说明 hardest-negative 可以抑制一部分无差别恢复动作。

但 Precision 与 Contact 的 B/C 结果基本相同，说明 hardest-negative 不是普适的充分条件；其作用被重采样先验和安全边界表示限制。

### D：再加 frontier

C 与 D 基本完全相同，没有稳定增益。因此 frontier/listwise 不应继续作为主模型默认模块。

## 6. Near-contact 当前核心任务

Near 已经不再处于“完全看不到收益”的阶段。当前需要完成的是：

1. 在自然总体上保持 recovery 的稀疏先验；
2. 保证 best safe action 高于 nominal；
3. 同时保证高收益但任何安全因子越界的 action 低于 nominal；
4. 把 Precision Near 的正向区域迁移到 certificate；
5. 在 closed loop 中真正改善 clearance/TTC/exposure，而不是只增加干预。

v48.29 paired shadow 每分支只有 8 个场景：

- Balanced Near TTC minimum 约提升 0.015 s；
- Precision Near TTC minimum 约提升 0.0078 s；
- clearance 基本不变；
- exposure duration 基本不变；
- bounded NUP 分别下降约 0.0152 和 0.0071；
- intervention rate 增至约 4.17% 和 1.56%。

这不是投稿级 Near 改善。它说明当前动作偶尔影响 TTC，但没有形成持续扩大安全裕度的策略，且干预带来了 utility 代价。

## 7. Contact 当前核心任务

Contact 的静态 recoverability 表示仍有一定价值，但组内 action ranking 仍弱，尤其是：

- 哪个 action 更快结束 overlap；
- 哪个 action 避免 re-contact；
- 哪个 action 建立持续 free space；
- 哪个 action 实现 sustained escape；
- 哪个 action形成高质量 stable stop。

v48.29 shadow 已正确匹配并运行，但 8 个 Contact target 的事件支持不足：

- overlap duration 全为 0；
- secondary/re-contact 全为 0；
- escape event 全为 1，time-to-escape 为 0；
- stable-stop 全为 0。

这些事件指标处于 floor/ceiling saturation，不能用于证明算法已经解决二次碰撞、逃逸或稳定停车。

可计算的连续指标显示：

- Balanced Contact terminal clearance 下降约 0.0527 m；
- free-space AUC normalized 下降约 0.0169 m；
- bounded NUP 下降约 0.0181；
- Precision Contact clearance/free-space 只提高约 0.0007–0.0009，接近无变化。

因此 Contact 仍未达到投稿目标。下一轮应继续输出这些物理指标，但不能在当前饱和样本上把“事件没有变化”解释为成功。

## 8. 是否仍存在工程问题

v48.29 没有发现类似旧版本的 checkpoint/inference silent wiring、split provenance、gamma alias 或 JSON 序列化崩溃。模型合同和 shadow runtime contract 均通过。

但存在一个会系统性影响结果的训练合同错误：

- stage-2 sampler 改变了自然 prevalence；
- replacement 重复少数正组；
- checkpoint 在自然 validation 上只能选择“最不坏”的早期 epoch；
- 训练目标与 certificate population 的干预先验不一致。

它既属于算法设计，也属于工程实现合同：如果论文声称 unified selective policy，训练 sampler 必须显式说明是否改变部署先验，并提供 importance correction；v48.29 没有做到。

容易再次出现的工程风险包括：

1. stage architecture 没有记录 sampler 是否 stratified/replacement；
2. checkpoint metric 与实际 Natural-gate 目标不一致；
3. 新模型参数未完整写入 checkpoint/inference；
4. admission prior 与 hard veto 使用不同风险语义；
5. validation index 错用 train index；
6. regime 名称只用于评估却意外进入模型；
7. shadow target support 饱和却被当成算法成功；
8. ablation 环境变量污染主实验默认值。

v48.30 为这些环节增加 fail-closed contract 检查。

## 9. v48.30：统一三种 regime 的 SLACK-RANK-BRIDGE

### 9.1 统一物理语义，而非 regime 路由

v48.30 不向模型输入 Safe/Near/Contact ID，也不注册三套 planner。

对任意 candidate `a` 与 nominal `a0`，定义五个候选相对 nominal 的连续安全越界余量：

```text
m_DRS(a)
m_DEP(a)
m_GAP(a)
m_RULE(a)
m_HARM(a)
```

每个余量都已经减去预注册容差：

- `m_k <= 0`：该物理维度在允许的非退化 envelope 内；
- `m_k > 0`：该维度越过不可补偿安全边界。

统一 worst slack：

```text
s(a) = max_k m_k(a)
```

统一 recovery ranking utility：

```text
U_safe(a) = B(a) - λ [s(a)]_+
```

其中 `B(a)` 是 raw recoverability benefit，`[x]_+=max(x,0)`。

独立 hard veto 仍保留。连续 slack 用于边界附近的稳定排序，hard veto 用于不可补偿拒绝。它们不是三种 regime 的不同策略，而是同一个“收益必须建立在物理非退化上”的原则。

### 9.2 为什么这一语义同时适用于三种 regime

- Safe：nominal 通常已有充足 headroom。没有正 benefit 或存在任一安全越界时，recovery action 自动低于 nominal，因此保护正常驾驶。
- Near：允许安全 action 因提升 recoverability 获得正分，但要求 clearance/TTC/规则/风险等维度不越界。
- Contact：允许 brake/stabilize/escape 类 action 提升 recoverability，但不能通过牺牲 deployability、扩大 gap、增加 hard-rule 或 harm proxy 来换取表面 benefit。

这比识别 regime 后调用不同策略更统一，也更符合论文关于 deployable recovery headroom 的核心 novelty。

### 9.3 连续 factor margin supervision

v48.29 的五因子主要使用 BCE，能够判断边界哪一侧，却丢失“距离边界多远”。

v48.30 加入 signed margin regression：

```text
m_hat_k = temperature × component_logit_k
L_margin = SmoothL1(m_hat_k, m_k)
```

这样 factor heads 不只输出 harmful 概率，还能为统一 slack projection 提供有序的安全距离。

### 9.4 stage-2 恢复自然总体训练

v48.30 admission stage：

```text
GROUP_BATCH_STRATIFIED=false
GROUP_BATCHING_REPLACEMENT=false
```

每个 scene-time group 每个 epoch 最多出现一次。hard-negative、positive weight 等只改变 loss 权重，不再改变数据先验。

Stage 1 仍可做 factor-balanced sampling，因为它只学习 dense benefit/factor 边界，不学习 deployment recovery prevalence。

### 9.5 population-aware checkpoint

v48.30 新增 threshold-free `direct_population_safe_rank_risk`，在自然 validation population 上共同惩罚：

- safe top-1 regret；
- harmful recovery mass；
- false admission mass；
- safe opportunity recall shortfall；
- safe mass shortfall。

它不把 regime ID输入模型，只把 Near/Contact作为 worst-stratum 报告和鲁棒 checkpoint 选择。

### 9.6 主模型保留与移除

保留：

- top-3 frozen proposal；
- five factor heads；
- factor→admission 两阶段；
- bounded admission；
- categorical nominal+top-k；
- deployment-exact safe-utility regression；
- hardest-negative；
- independent hard veto；
- Natural certificate；
- fast physical shadow；
- legacy Noisy-OR 关闭。

主模型移除：

- stage-2 50% positive replacement sampling；
- unbounded admission；
- top-8 proposal；
- safe-utility listwise；
- frontier contrast；
- regime-specific residual/router。

## 10. v48.30 消融

八个任务仍可同时运行，GPU0 四个 Balanced、GPU1 四个 Precision：

1. `A_natural_population_reference`  
   自然 population + benefit-only，无连续 margin/slack/hard-negative。
2. `B_add_signed_component_margin`  
   A + 五因子 signed margin regression。
3. `C_add_safety_slack_projection`  
   B + unified safety-slack admission prior。
4. `D_full_slack_rank`  
   C + hardest-negative，v48.30 主设计。

该设计能回答：

- 自然总体训练是否解决过度准入；
- signed margin 是否改善 harm/frontier 表示；
- safety slack 是否同时提高 Near/Contact；
- hardest-negative 在正确 population 下是否产生真实增益。

## 11. Gate 与停止规则

Natural gate 不修改。

下一轮优先查看：

1. `TRAINING_CONTRACT.json` 是否 valid；
2. stage-2 是否自然采样且无 replacement；
3. population checkpoint metric 是否有限且跨 epoch 变化；
4. development precision LCB/harm UCB/recall；
5. certificate dev→verify gap；
6. Near/Contact selected teacher advantage；
7. physical shadow 的 clearance/TTC/free-space/NUP；
8. `PHYSICAL_TARGET_SUPPORT.json` 是否提示 Contact floor/ceiling saturation。

如果 v48.30 仍为 RC=20：

- 自然 population 后 precision 明显提升但 recall 不足：再调整 safe-positive loss weighting，不改变 sampling prior；
- dev 可通过而 certificate 失败：重点处理 scene-level泛化和 margin calibration；
- offline safe ranking 改善但 physical shadow 不改善：下一版才应预注册 candidate-level temporal physical teacher；
- oracle 不再可行：才重新讨论 proposal/label/certificate contract。

不能通过在当前 certificate 上事后调低阈值来制造 RC=0。

## 12. 本地验证

当前环境完成：

```text
265 passed, 5 warnings
compileall PASS
all shell bash -n PASS
```

当前环境没有真实 WOMD/Waymax 和两张 A30，因此不能预先声称 v48.30 已通过 gate 或达到 CCF-A closed-loop 目标。
