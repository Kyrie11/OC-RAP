## v48.55 — DCP-DRFC-BCDE-TCBC / COORDINATE-TYPED COMPONENT BOUNDARY CALIBRATION (2026-08-19)

**类别：由 v48.54 authoritative A/B 对 IPBD 的跨严重度负迁移直接触发的 component-correctness 2×2。主线仍是一个 regime-agnostic planning primitive；Safe / near-contact / contact 只是 normal→critical continuum 上的 dataset/evaluation strata。没有新增 regime identifier/router、regime-specific policy/threshold/loss/budget、proposal top-k、candidate family、learned admission residual、重采样、root-logit recalibration 或网络容量。**

### v48.54 工程状态：只有 Main 是预期行为，不是缺失消融

v48.54 是单轴 A/B：A 语义复用 v48.53-A，只有 IPBD Main/B 需要重新训练。`OC-RAP-v48.54-A-reference-reuse-contract.json` 为 `valid=true`、`errors=[]`，source checkpoint SHA、五个 canonical manifest SHA 与 gate semantic checks 全部匹配，且 `strategy_regime_conditioning=false`、`test_roots_read=false`。Main 为 authoritative `RC=20`、`pipeline_valid=true`、certificate/Natural gate真实执行；没有 RC30、Traceback、OOM 或缺失必需 artifact。因此本轮结果可做可靠 B-A 归因。

### v48.54 可靠归因：IPBD 作为统一主机制失败，physical-margin distillation family STOP

Precision 核心 B-A：

| regime | metric | A q-hard reference | B IPBD | B-A |
|---|---:|---:|---:|---:|
| Near | certificate recall | **0.333** | **0.000** | -0.333 |
| Near | harmful UCB90 | **0.042** | 0.247 | +0.205 |
| Near | candidate safe-positive AUC | **0.448** | 0.320 | -0.128 |
| Near | proposal safe-positive AUC | **0.491** | 0.384 | -0.108 |
| Near | development recall | **0.375** | 0 | -0.375 |
| Near | development joint sign | **4/19** | 0/19 | -4/19 |
| Near | dev DRS harmful false-safe | **0.419** | 0.706 | +0.287 |
| Contact | certificate recall | **0.050** | 0 | -0.050 |
| Contact | harmful UCB90 | 0.351 | **0.340** | -0.011 |
| Contact | candidate safe-positive AUC | 0.632 | **0.736** | +0.103 |
| Contact | proposal safe-positive AUC | 0.611 | **0.688** | +0.077 |
| Contact | dev DRS harmful false-safe | 0.789 | **0.560** | -0.229 |
| Contact | dev exact/native positive | **7/37** | 4/37 | -3/37 |

IPBD 因而不是“final centering 遮住了一个已正确的 upstream mechanism”。它确实证明 selected-option physical margin 含有 Contact-specific discrimination/specificity 信息，但同一个 training-only privileged signal 通过共享 witness 注入后会让 Near ranking、DRS specificity、joint sign 与 final evidence 同时崩塌。**未满足进入 Boundary-Complete Evidence Centering 的先决条件。**

**REJECT / STOP：**

- IPBD 作为 unified Main STOP；不再调 IPBD weight/temperature、selected-margin sampling 或分桶权重；
- physical-margin distillation family 整体 STOP（teacher-only PSA、student/native physical hard DRS、symmetric CSE 已在 v48.52/v48.53 STOP）；
- 不用 Contact 的局部收益为理由引入 regime-conditioned physical loss/router；
- 不进入 final evidence centering，直到 upstream component/native geometry 在 Near+Contact 同时形成 Pareto improvement；
- root-logit recalibration 继续 STOP；v48.54 B-A 的 root logits/source checkpoint均冻结一致，不能解释该负效应。

### v48.50→v48.54 的收窄链：应保存 decision invariant，而不是复制 privileged realization

1. **v48.50**：Exact-only transport 会丢失 local order；证明 exactness 不是越多越好。
2. **v48.51**：BC-FC 支持 `hard/discontinuous sign + smooth local order` 的职责分离；这是截至目前最稳定的主机制原则。
3. **v48.52**：teacher-only physical sign（PSA）系统性负向；否定“只把 teacher 物理化就更正确”。
4. **v48.53**：student/deployment physical replacement 与 symmetric CSE 仍失败；否定“内部 certificate computation 必须 structural imitation”。
5. **v48.54**：即便 deployment q-hard invariant完全不改，training-only physical margin distillation仍造成 Contact gain / Near collapse；否定“privileged physical signal 只要不进入 deployment 就天然无害”。

因此论文主线不再强调 *physical certificate realization*，而收敛为：

> **Invariant-Preserving Boundary-Complete Decision Equivalence：部署的 material boundary 必须保持 decision-equivalent；不同数学类型的证书坐标应按其类型校准，而 privileged physical geometry 只能在证明跨严重度 Pareto 安全后进入共享表示。**

当前仍支持：`q-hard` material sign、smooth q local order、observation-class execution、shared regime-agnostic policy。需要继续被验证的是 **heterogeneous component coordinates 的 calibration geometry**。

### 新 dominant bottleneck：coordinate-typed component supervision geometry / cross-regime scale consistency

v48.54 的 component readout显示共享 factor head存在明显坐标过旋转：同一个 IPBD 对 Contact 的 DRS specificity/ranking有利，却让 Near DRS、ranking、opportunity/pred-adv同时恶化；certificate component 中 GAP 甚至出现极端 safe-positive false-veto。当前三个被支持的 component margin 又被统一使用 raw SmoothL1 magnitude regression，但其数学类型并不相同：

- DRS 是离散/量化的 root-mass boundary coordinate；
- deployability 与 gap-quality 是连续非线性坐标；
- pooled adaptation-train 的典型 RMS 约为 DRS `0.419`、DEP `0.223`、GAP `0.356`，尺度明显不一致。

因此下一步不再改 hard certificate，而验证：**discontinuous component 是否只应承担 boundary/sign supervision；continuous components 是否应使用 regime-free、train-only、零点/顺序保持的尺度规范化。**

### v48.55 严格 2×2：Coordinate-Typed Component Boundary Calibration (TCBC)

两个 causal factors：

- **X = DRS boundary-only magnitude contract**：DRS 保留原 component BCE / hard veto / q-hard deployment，但从 explicit continuous SmoothL1 magnitude regression 中移除。没有删除 DRS sign supervision。
- **Y = continuous DEP/GAP pooled-RMS linear canonicalization**：只从 pooled adaptation-train Near+Contact teacher index计算 component RMS；DEP/GAP target 使用 `0.10 * raw_margin / pooled_RMS_k`。DRS在 Y 轴保持 identity（scale=0.10）；zero crossing、component内 order 与 hard veto不变。没有 `tanh`、clipping、regime权重或 dev/certificate/test拟合。

四臂：

- **A:** X=0, Y=0 — 当前 q-hard BC-FC + smooth-NAP raw-component reference；优先语义复用 v48.53-A。
- **B:** X=1, Y=0 — DRS boundary classification only；DEP/GAP raw regression。
- **C:** X=0, Y=1 — DRS raw regression；DEP/GAP pooled-RMS-linear canonicalization。
- **D/Main:** X=1, Y=1 — TCBC。

所有 arm 固定：BC-FC=true、smooth NAP=true、BC-NAP=false、PSA/CSE/IPBD=false、physical native DRS=false、root logits不训练、hard component veto不变、proposal top-k=5、source/data/gate不变、`strategy_regime_conditioning=false`、`test_roots_read=false`。

### v48.55 preregistered readout / go-stop

读取顺序必须是 **B-A → C-A → D-B-C+A**，而不是先看 D recall。

1. **B-A / DRS typed supervision**：若 Near DRS harmful false-safe / false-veto改善或不退化，同时 Near recall/ranking保持，并且 Contact不显著损失，则支持“discontinuous DRS 不应做 continuous magnitude regression”。
2. **C-A / DEP-GAP canonicalization**：若 Near/Contact 的 deployability/gap false-veto 与 harmful false-safe同时向 Pareto方向移动，并保持 candidate/proposal ranking，则支持 pooled scale mismatch 是 causal bottleneck。
3. **D interaction**：只有 D 的绝对 Near/Contact native/component geometry、harmful UCB 与 ranking均回到/超过 A，才可称 TCBC complementary；不能用 floor arithmetic 的正 interaction代替绝对改善。
4. **若 component/native geometry 跨 Near+Contact Pareto 改善，但 opportunity/pred_adv仍系统性负偏**：此时才正式进入 **Boundary-Complete Evidence Centering**。
5. **若 X/Y/D 都不能改善 component Pareto**：STOP component-normalization family；下一步改为 teacher component correctness audit（重新核对 DEP/GAP target定义/normalization/source labels），之后才做 root-uncertainty diagnosis。仍不重开 root-logit recalibration。
6. **Safe**：仅 D/Main authoritative RC0 后才执行 scene-disjoint paired Safe non-inferiority + stress/closed-loop；RC20 不放宽 gate。

### v48.55 runtime / engineering contract

- A reference reuse继续使用 semantic identity：source checkpoint SHA + 五个 canonical manifest SHA + gate semantics + factor contract；忽略 transient protocol-seal byte timestamp。
- 正常情况下只新跑 B/C/D：B、C 分别独占一张 GPU并行，D 再用两张 GPU并行 Balanced/Precision。
- Y 轴 scale artifact由 train teacher index生成并绑定 index SHA；`strategy_regime_conditioning=false`、`test_roots_read=false`。
- `V48_55_TCBC_CONTRACT.json` 在 certificate access 前 fail-closed 验证 target mode、regression reliability 与 scale artifact；任何 mismatch -> RC30 engineering failure，不允许做算法归因。

### 延续 stop signals

继续禁止：IPBD/physical-margin distillation、teacher-only PSA、student/native physical hard DRS、symmetric CSE、hard q/margin AND/OR transport、BC-NAP、old hard-magnitude DEFC、MC-NCP tolerance search、exact-only NAP、root-logit recalibration、threshold relaxation/grid densification、top-k/candidate/macro expansion、aggressive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、learned admission residual、one-sided component penalty、unbounded factors、v48.40 frontier-tanh、v48.41 full component factorization、broad encoder fine-tuning，以及任何 regime-conditioned policy/router/threshold/budget。


## v48.54 — DCP-DRFC-BCDE-IPBD / INVARIANT-PRESERVING PHYSICAL BOUNDARY DISTILLATION (2026-08-19)

**类别：由 v48.53 authoritative 2×2 对 CSE 的否证与 C-arm 的局部正信号共同触发的单轴机制实验。主线继续是一个 regime-agnostic planning primitive；Safe / near-contact / contact 仅是 normal→critical continuum 上的 dataset/evaluation strata。没有新增 regime identifier/router、regime-specific policy/threshold/loss/budget、proposal top-k、candidate family、learned admission head、重采样或网络容量。**

### v48.53 的可靠归因：CSE Main 不成立，physical hard certificate replacement 全家族停止

v48.53 A/B/C/D attribution contract valid，四臂 dataset manifest、source checkpoint、gate protocol identity 一致，均为 authoritative `RC=20`、`pipeline_valid=true`、certificate/Natural gate 执行且 `test_roots_read=false`。历史 v48.52 A/B 因 protocol-seal byte SHA变化被 fail-closed 拒绝，因此本轮 fresh 四臂完全可做 B-A / C-A / interaction 归因。

Precision 核心结果：

| regime | metric | A q-hard | B teacher-physical | C student-physical | D symmetric physical |
|---|---:|---:|---:|---:|---:|
| Near | cert recall | **0.333** | 0.222 | 0.222 | 0.111 |
| Near | harmful UCB90 | 0.042 | 0.074 | **0.039** | 0.044 |
| Near | candidate safe-positive AUC | 0.448 | 0.397 | **0.535** | 0.427 |
| Near | dev recall | **0.375** | 0.125 | 0.250 | 0 |
| Near | dev DRS harmful false-safe | **0.419** | 0.610 | 0.603 | 0.868 |
| Contact | cert recall | **0.050** | 0 | 0 | 0 |
| Contact | harmful UCB90 | **0.351** | 0.473 | 0.473 | 0.591 |
| Contact | candidate safe-positive AUC | 0.632 | 0.540 | **0.655** | 0.608 |
| Contact | dev exact/native positive | **7/37** | 0/37 | 2/37 | 1/37 |
| Contact | dev DRS harmful false-safe | **0.789** | 0.856 | 0.917 | 0.984 |

存在若干正 arithmetic interaction（例如 Contact recall interaction `+0.05`、exact/native-positive interaction `+0.162`），但 **D/Main 绝对表现不恢复 A，且 specificity 单调恶化**。因此不能把这些 interaction 解读为 CSE 成功。

**REJECT / STOP：**

- teacher-only physical hard sign（PSA）继续 STOP；
- student/deployment-only margin-physical hard DRS 作为 Main STOP；
- teacher+student/deployment symmetric margin-physical CSE 作为 Main STOP；
- 不再做 q-hard/margin-hard 的 AND/OR、混合 hard DRS、CSE weight/temperature/threshold search；这些仍属于已被 v48.53 否定的 hard-certificate replacement / transport family。

### v48.53 推进后的方法结论：decision equivalence 是 invariant preservation，而不是 structural imitation

截至 v48.53，最稳定的原则仍是：

1. **Observation consistency** 保持必要；
2. **hard/discontinuous q-based certificate 负责 material decision sign**；
3. **smooth q geometry 负责 hard-equivalence class 内 continuous order**；
4. deployed hard/exact boundary 必须进入 upstream learning/calibration；
5. 但 **teacher/student/deployment 不需要、也不应被强制做逐结构的 margin-physical imitation**。

v48.53 反驳了上一版提出的强命题“certificate computation 必须 structural-equivalent”。更准确的 CCF-A 论文主线应推进为：

> **Physical Boundary-Complete Decision Equivalence requires preservation of decision-relevant invariants, not identity of internal certificate realization.**

`q-hard DRS` 与 selected-root physical margin 分别携带不同有用信息：A 的 q-hard realization提供更好的 harmful specificity / recall；C 的 physical-student factor在 Near/Contact 都提高 candidate safe-positive discrimination，并把 DRS safe-positive false-veto降到 0，但同时把 harmful false-safe显著推高。说明 physical margin 是有价值的 **privileged boundary signal**，但不适合拥有 deployment hard sign。

### 新 dominant bottleneck：privileged physical boundary information 未被“无损”注入 q-hard witness

当前问题不再是 teacher/student structural equivalence，也还不能进入 final evidence centering。最有信息增益的问题变为：

> 如何利用 selected-option physical margin 的边界信息改善 latent recovery witness/ranking，同时保持部署 q-hard DRS 的 specificity 与 decision semantics 完全不变？

C 的局部正结果表明 physical margin 确实有可迁移信息；C/D 的 harmful false-safe恶化表明错误发生在“让 physical margin 接管 hard DRS”，而不是“physical margin本身无信息”。因此 v48.54 只允许 training-time privileged distillation，不再改变 teacher/student/deployment hard certificate。

### v48.54 单一新因素：Invariant-Preserving Physical Boundary Distillation (IPBD)

共同 reference（A/B 完全相同）：

- BC-FC=true；smooth NAP=true；BC-NAP=false；exact-only NAP=false；MC-NCP=false；old DEFC=false；
- teacher hard sign=`q_hard_proxy_drs_exact_pcd`；
- student/deployment hard sign=`hard_qbest_ge_zero_root_mass_exact_pcd`；
- smooth order=`smooth_boundary_drs_smooth_pcd`；
- `physical_teacher_sign_alignment=false`；`physical_student_sign_alignment=false`；`native_physical_student_drs=false`；
- root logits / encoder / decoder继续冻结；无新 head/parameter/threshold/regime input。

唯一 B-A 因素：

1. teacher OC-MERO `q` 在每个 valid root/class 内选择 observation-consistent option；
2. 只取这个 teacher-selected option 的 `m_star` 物理 zero crossing；
3. 对同一 option 的 predicted margin 加 class-balanced、teacher-root-probability-weighted BCE：`target = 1[m_star_selected >= 0]`；
4. zero boundary固定为 0，不使用新的 threshold；温度复用现有 frontier sign temperature `0.08`；loss weight复用现有 frontier sign weight `0.50`，不做搜索；
5. privileged physical target **不参与 option routing，不进入 native DRS，不改变 deployment/admission**。

这不是 one-sided component penalty、generic pairwise/listwise、learned admission residual，也不是 CSE hard certificate replacement；它只测试一个新命题：**physical boundary information 应作为 privileged invariant distillation，而非 deployed hard-coordinate imitation。**

### v48.54 A/B 与 preregistered go/stop

- **A/reference**：优先语义复用 v48.53-A；若 source checkpoint SHA、5 个 canonical manifest SHA、gate semantic protocol、factor contract任一不一致，则 fail-closed fresh A。
- **B/Main**：A + IPBD。

优先判断 B-A：

1. **Near**：certificate recall应至少保持 `>=0.25`（目标恢复 A 的 `0.333`），harmful UCB90不应明显高于 A（优先 `<=0.05`）；candidate safe-positive AUC希望高于 A `0.448`；development joint sign期望不低于 A 的 `4/19` 量级。
2. **Contact**：candidate safe-positive AUC应至少不低于 A `0.632`，proposal safe-positive AUC不应明显跌破 A `0.611`；DRS harmful false-safe不能出现 C/D 型恶化；certificate recall首先要求恢复/保持 A 的 `0.05`，harmful UCB90不应高于 A `0.351`。
3. **若 IPBD 改善 ranking/native boundary diagnostics且保持 q-hard specificity，但 opportunity/pred_adv仍系统性负偏**：此时才触发下一单轴 **Boundary-Complete Evidence Centering**。
4. **若 IPBD 无法形成上述 Pareto 改善**：STOP physical-margin distillation family；严格转入 v48.53 已预注册的 DEP/GAP teacher normalization、component correctness 与 root-uncertainty diagnostic。仍不重开 root-logit recalibration。
5. **Safe**：Main authoritative RC0 后才执行 gate-generated Safe paired non-inferiority + stress/closed-loop；RC20 不放宽 gate。

### v48.54 工程 / runtime 修复

v48.53 telemetry仍显示 A30 峰值显存约 537MB，GPU mean utilization约 19%/24%，pipeline继续受 Python/I/O/小粒度 inference限制。保留 standard-calibration prediction cache，不引入新的数值近似。

v48.53 reference reuse再次因 `V48_45_PROTOCOL_SEAL.json` byte SHA变化而失败；该 seal包含 transient creation metadata，导致语义相同 protocol也会被拒绝并重复运行。v48.54 reference reuse改为 **semantic identity**：source checkpoint SHA + 5 canonical dataset manifest SHA + preregistered gate protocol semantics + factor contract。仍 fail-closed，但不再把 transient seal byte identity当作算法 identity。

### 延续 stop signals

继续禁止：teacher-only PSA、student/native physical hard DRS、symmetric CSE、hard q/margin AND/OR transport、BC-NAP、old hard-magnitude DEFC、MC-NCP tolerance search、exact-only NAP、root-logit recalibration、threshold relaxation/grid densification、top-k/candidate/macro expansion、aggressive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、learned admission residual、one-sided component penalty、unbounded factors、broad encoder fine-tuning、以及任何 regime-conditioned policy/router/threshold/budget。


## v48.53 — DCP-DRFC-BCDE-CSE / CERTIFICATE STRUCTURAL EQUIVALENCE (2026-08-18)

**类别：由 v48.52 PSA authoritative A/B 负结果与代码结构审计共同触发的严格 2×2 falsification experiment。主线仍是一个 regime-agnostic planning primitive；Safe / near-contact / contact 只作为同一 normal→critical continuum 上的 dataset/evaluation strata。没有新增 regime identifier/router、regime-specific policy/threshold/budget/loss、proposal top-k、candidate family、learned admission head、重采样或网络容量。**

### v48.52 的可靠归因：teacher-only PSA 明确 reject，但 BC-FC 的 hard-sign / smooth-order 原则未被推翻

v48.52 A/B attribution contract valid，A 与 B protocol/source/gate identity 一致，均为 authoritative `RC=20`、`pipeline_valid=true`、certificate/Natural gate 实际执行、`test_roots_read=false`。历史 v48.51-B 因 protocol seal SHA mismatch 被正确拒绝复用，因此 A 是 fresh、byte/protocol-compatible 的 BC-FC + smooth-NAP reference；B-A 只隔离 teacher-side Physical Sign Alignment。

Precision 关键结果：

| regime | metric | A / q-proxy teacher | B / teacher PSA | B-A |
|---|---|---:|---:|---:|
| Near | certificate recall | **0.333** | 0.222 | -0.111 |
| Near | harmful-selected UCB90 | **0.042** | 0.074 | +0.032 |
| Near | candidate safe-positive AUC | **0.448** | 0.397 | -0.051 |
| Near | development recall | **0.375** | 0.125 | -0.250 |
| Near | development joint sign | **4/19** | 1/19 | -3/19 |
| Contact | certificate recall | **0.050** | 0 | -0.050 |
| Contact | harmful-selected UCB90 | **0.351** | 0.473 | +0.122 |
| Contact | candidate safe-positive AUC | **0.632** | 0.540 | -0.093 |
| Contact | development exact-positive | **7/37** | 0/37 | -7/37 |
| Contact | development joint sign | 0/37 | 0/37 | 0 |

PSA 因此不是 neutral experiment，而是**系统性负向**。Near/Contact 的 recall、ranking 与 physical/exact sign 都没有形成 Pareto improvement；Contact latent exact-positive geometry甚至从 7/37 清零。

**STOP: teacher-only PSA.** 不再通过调 sign weight、temperature、margin anchor、PSA strength 或 threshold 尝试“救回”v48.52-B。

但这个结果**不推翻 v48.51 的 BC-FC 结论**。A 本身就是 v48.51-B 的 fresh reference：Near recall 0.333 / harmful UCB 0.042，继续支持 hard/discontinuous coordinate负责 material sign、smooth q geometry负责 hard-equivalence class 内 local order/magnitude。v48.52 推翻的是更强命题：“只要把 teacher hard sign 改成 physical certificate，student/deployment 保持 q-hard realization 也会更好”。

### v48.52 负结果揭示的新 dominant bottleneck：teacher–student certificate structural mismatch

代码审计确认 v48.52-B 的两侧不是同一个事件：

- **teacher PSA**：`teacher q -> 选择 observation-consistent option -> selected m_star >= 0 判 root physical success -> teacher root mass 聚合 DRS`；
- **student/deployment**：`predicted q_best >= 0 -> predicted root mass 聚合 DRS`。

`q` 是 observation-compatible lower-tail robust recovery value；`m_star(selected)` 是被选 action 在具体 root 上的 physical margin。它们拥有相关但不同的语义。v48.52 只物理化 teacher、没有物理化 student/deployment，相当于让 BCE 监督一个 student representation不能结构同构实现的 target，同时 smooth order channel仍继续优化 q-depth。这会在共享 `margin_head` 上产生 target/representation conflict，能够解释 PSA 同时伤害 Near 与 Contact。

因此当前第一瓶颈**不是 final evidence centering**。v48.52 没有满足上一版预注册的“physical sign correctness先改善、final learned sign仍负”条件，不能进入 Boundary-Complete Evidence Centering。当前第一瓶颈改为：

> **Certificate Structural Equivalence：teacher 与 student/deployment 的 hard certificate composition 必须先同构，才能讨论 learned evidence 是否仍有 centering error。**

predicted-root reliability 仍可能是绝对性能共同瓶颈，但 v48.52 A/B 的 `root_logit_head` 冻结且 source checkpoint一致，所以它不能解释 PSA 的 B-A 负效应。历史 root-logit recalibration STOP 继续有效，禁止重开该路线。

### v48.53 方法假设：Certificate Structural Equivalence (CSE)

将 *Physical Boundary-Complete Decision Equivalence* 收敛为三个同一原则下的必要条件，而不是继续增加模块：

1. **Observation consistency**：hidden future/root identity不能进入 recovery decision；
2. **Boundary completeness**：hard certificate只负责 material sign，smooth q geometry负责 hard-equivalence class 内 local order/magnitude；
3. **Certificate structural equivalence**：teacher、student训练坐标与 deployment hard certificate采用相同的组合结构。

v48.53 新增的 student/deployment physical hard DRS 为：

`predicted q -> 选择 observation-consistent option -> selected predicted margin >= 0 判 root physical success -> predicted root mass 聚合 DRS`。

训练 forward 使用相同 hard zero crossing；backward只在**被 q 选中的 predicted margin**上使用 sigmoid straight-through gradient。q-side option selection保持 hard，不引入 soft router；smooth q-based DRS/PCD order channel保持 v48.51 byte-semantics不变。最终 native/deployment DRS coordinate 0 同样采用 q-selected predicted-margin physical success。没有新增 head、参数、threshold、regime输入或策略分支。

物理 margin zero crossing固定为 0；q 的 `gamma` 只服务 option-selection boundary/tie semantics，不能平移 physical margin boundary。

### v48.53 严格 2×2

两个 causal factors：

- **X = teacher physical sign alignment**（v48.52 PSA）；
- **Y = student/deployment physical certificate alignment**（v48.53 新因素）。

四臂：

- **A:** X=0, Y=0 — q-proxy teacher + q-hard student/deployment（v48.52 A reference）；
- **B:** X=1, Y=0 — teacher-only PSA（v48.52 B，已知负向）；
- **C:** X=0, Y=1 — student/deployment-only physical certificate；
- **D/Main:** X=1, Y=1 — teacher/student/deployment structurally equivalent physical certificate。

所有 arm 固定 BC-FC=true、smooth NAP=true、BC-NAP=false、exact-only NAP=false、MC-NCP=false、old DEFC=false、ROCT/top-k/source/data/calibration/risk protocol不变；所有 arm `strategy_regime_conditioning=false`。

读取顺序必须是：**B-A（已知） -> C-A -> D-B-C+A**。如果 B、C 单边都负但 D 有显著正 interaction，这将是最强的 CSE 机制证据；不能只看 D 最终 recall。

### v48.53 runtime / reference design

默认 fail-closed 检查并复用现有 v48.52 A/B：要求 current protocol seal、source checkpoint SHA、gate semantics、authoritative RC0/20、factor contract、Balanced/Precision witness contracts全部一致。通过后只新跑 C/D，分别占 GPU0/GPU1，每个 arm内部 Balanced/Precision串行；若任何 identity不一致则自动回退 fresh A/B -> C/D 两波。

继续保留 v48.52 standard-calibration prediction cache；不启用 AMP/TF32、跨 scene/group 合批、candidate reorder等可能改变数值或集合语义的加速。

### v48.53 预注册 go / stop

优先看 component/native geometry，而不是先追 RC0：

1. **Structural-equivalence evidence**：D 相对 B 必须恢复/改善 physical DRS geometry；Near 至少不能继续 PSA 型 collapse，重点看 DRS safe-positive false-veto、harmful false-safe、exact/native sign 与 candidate/proposal safe-positive AUC。
2. **Near mechanism screen**：D Precision certificate recall期望恢复到 `>=0.25`，harmful-selected UCB90 `<=0.15`；development joint sign期望至少回到 A 的 `4/19` 量级。此处是机制 screen，不是最终论文 gate。
3. **Contact mechanism screen**：首先要求 D 不再出现 v48.52-B 的 `0/37` exact-positive collapse；优先看 exact/native physical-positive fraction是否恢复到 A 的 `7/37` 量级、candidate/proposal ranking是否不再明显低于 A，以及 harmful UCB是否下降。certificate recall `>=0.05` 视为恢复 A 信号，`>=0.10` 才视为下一阶段明显进展。
4. **若 D 改善 physical/native geometry，但 final opportunity/pred_adv 仍明显负偏**：这时才触发下一单轴 **Boundary-Complete Evidence Centering**，因为 teacher/student/deployment correctness 已被支持。
5. **若 D 仍不能改善或强负 interaction**：STOP BC-FC/CSE/transport family；转入 DEP/GAP teacher normalization、teacher component correctness 与 predicted-root reliability的**诊断审计**。仍禁止重新训练 root logits；是否需要新的 root uncertainty representation必须另立机制假设，而不能把已拒绝的 root-logit recalibration换名字重做。
6. **Safe**：继续只在 D/Main authoritative RC0 后执行 gate-generated scene-disjoint Safe paired non-inferiority + stress/closed-loop。RC20 不放宽 gate。

### 新增 / 延续 stop signals

- **teacher-only PSA：STOP**；不做 PSA strength/weight/temperature search。
- **BC-NAP Main：STOP**；保持 smooth NAP。
- **old hard-magnitude DEFC：STOP**；hard coordinate不回归 continuous magnitude。
- **root-logit recalibration：STOP**；v48.45.6/46 已有负结果，v48.52 A/B也无法把其作为 causal解释。
- **threshold relaxation / grid densification：STOP**；不能为 Safe 解锁而改 Natural gate标准。
- 继续禁止：MC-NCP tolerance调节、exact-only NAP、top-k/candidate/macro扩张、aggressive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、learned admission residual、one-sided component penalty、unbounded factors、broad encoder fine-tuning、以及任何 regime-conditioned policy/router/threshold/budget。

### v48.53 代码/工程合同

- `boundary_complete_frontier_calibration_loss` 新增 student-side physical factor；缺 `pred_margins` 时 fail-closed。
- student physical hard DRS 的 option selection与 teacher相同；physical zero crossing固定 `selected_margin>=0`；backward只对 selected margin用 STE。
- `OCRAPModel` native certificate新增 `direct_recovery_evidence_physical_student_drs`；只有 coordinate 0 DRS 改为 physical composition，DEP、smooth boundary mass、gap-quality保持不变。
- checkpoint/config/inference reconstruction、model contract、factor-cache identity、v48.47 nested witness stage、v48.53 stage contract全部接线新 flag。
- 新增 v48.53 CSE witness checker、A/B reuse checker、2×2 comparator、two-GPU launcher与 RC0-only post-gate wrapper。
- C/D final model与训练 witness的 student physical flag必须一致；任何 stale/asymmetric contract在 calibration前提升为 RC30，禁止归因为算法结果。

## v48.52 — DCP-DRFC-BCDE-PSA / PHYSICAL SIGN ALIGNMENT (2026-08-18)

**类别：由 v48.51 authoritative 2×2 与代码语义审计共同触发的单轴 correctness experiment。主线继续是一个 regime-agnostic planning primitive；Safe / near-contact / contact 只作为 normal→critical continuum 上的数据/评测 strata。没有新增 regime identifier/router、regime-specific policy/threshold/loss/budget、proposal top-k、candidate family、learned admission head、重采样或模型容量。**

### v48.51 的可靠结论：BC-DE 原则部分成立，但 BC-NAP 不应进入 Main

v48.51 四臂 attribution contract valid，均为 authoritative RC20 / pipeline-valid algorithm rejection，certificate/Natural gate 实际执行且 test roots 未读取。因此 B-A、C-A 与 interaction 是有效机制证据。

Precision 关键结果：

| Arm | 机制 | Near cert recall | Near harmful UCB90 | Near dev joint sign | Contact cert recall | Contact harmful UCB90 | Contact dev joint sign |
|---|---|---:|---:|---:|---:|---:|---:|
| A | smooth NAP reference | 0.222 | 0.112 | 0/19 | 0 | 0.567 | 0/37 |
| B | A + BC-FC | **0.333** | **0.042** | **4/19** | **0.050** | 0.351 | 0/37 |
| C | A + BC-NAP | 0.222 | 0.130 | 0/19 | 0 | 0.356 | 3/37 |
| D | BC-FC + BC-NAP | **0.333** | 0.100 | **4/19** | 0 | **0.240** | **6/37** |

**Accept BC-FC principle.** B 是唯一同时提高 Near certificate recall、降低 harmful UCB、打开 development sign 的单因素机制。这继续支持“部署真正消费的 hard/exact decision boundary 必须进入 upstream calibration；hard/discontinuous coordinate 负责 sign，smooth boundary geometry 负责 continuous magnitude/order”的主线。

**Reject BC-NAP as a Main transport.** C 只带来部分 candidate/proposal ranking 改善，没有 development sign；D 与 B 组合后 Near candidate/proposal safe-positive ranking 下降，而且 Contact certificate recall 从 B 的 0.05 退回 0。BC-NAP 因此保留为诊断机制，不再进入后续 Main。原 v48.49 smooth NAP 保留。

### dominant bottleneck 已从 boundary quantization 下移到 physical sign correctness

v48.51 已经部分解决“boundary quantization vs local ordering”冲突。当前更上游、优先级更高的问题来自代码级语义审计：BC-FC 的 teacher hard DRS 在 v48.51 仍由 `teacher_q_best >= gamma` 构造；但 Natural-gate/evaluator 的物理 teacher DRS 实际是：

`teacher q -> 选择 observation-consistent legal option -> 在该 option 上用 m_star >= 0 判 root 物理恢复成功 -> root-probability aggregation`。

因此 v48.51-B 虽然证明 hard-sign/smooth-order 分工有效，但其 **teacher sign supervision 仍是 q-hard proxy，而不是最终被 gate 消费的 physical certificate**。继续调 frontier loss weight、temperature、threshold 或 downstream transport 会把 teacher-error 与 centering-error 混在一起，违反 v48.51 已预注册的 stop rule。

### v48.52 单一新因素：Physical Sign Alignment (PSA)

v48.52 保留 v48.51-B 的 BC-FC + smooth NAP，明确关闭 BC-NAP、exact-only NAP、old v48.50 DEFC、MC-NCP。唯一新因素是 teacher-side sign definition：

- **Student/deployed sign coordinate 不变**：model predicted q 上的 hard `q_best>=gamma` DRS，forward exact / backward STE；
- **Teacher option selection 不变**：teacher OC-MERO q 选择 observation-consistent legal recovery option；
- **Teacher physical success 改正**：所选 option 的 root 是否成功由对应 `m_star>=0` 判定，而不是再次用 q>=0；
- **Order/magnitude channel 完全不变**：smooth boundary DRS + smooth PCD 继续做 continuous SmoothL1；
- sign channel 仍只做 balanced sign BCE；没有让 hard DRS 做 continuous magnitude regression。

这把 BC-DE 的方法原则收敛为：**decision equivalence 不只要求 student 的部署坐标对齐；supervision 的 material sign 也必须与最终 physical certificate 语义同构。** 论文可称为 *Physical Boundary-Complete Decision Equivalence*，PSA 是该原则的 correctness realization，而不是新增一个独立 policy module。

### 为什么现在不直接加 final evidence centering

v48.51 Contact 已经显示 latent physical/exact sign 与 final learned opportunity/pred-adv 之间仍存在 centering gap；但在 teacher sign correctness 修正前直接改 evidence centering 会混入 upstream label mismatch，无法得到干净因果结论。因此 v48.52 先做 PSA。

- 若 PSA 显著改善 DRS safe-positive veto / physical joint sign，但 final opportunity/pred-adv 仍压在负侧，则下一 dominant bottleneck 才可可靠定位为 **Boundary-Complete Evidence Centering**；下一版只做该单轴。
- 若 PSA 连 physical sign consistency 都不能改善，则停止 BC-FC/transport 微调，转入 DEP/GAP teacher normalization、root-probability reliability 与 teacher/native component correctness audit。

### Safe / Near / Contact 的统一目标不变

- **Safe**：不降低 Natural-gate 标准来强行获取 RC0。Safe standard calibration valid 只是必要条件；critical gate 真正 RC0 后才授权 scene-disjoint paired non-inferiority + closed-loop。
- **Near-contact**：v48.51-B 已把 recall 拉到 0.333 且 harmful UCB 到 0.042，说明 recovery signal 存在；当前主要缺口是 precision/centering 与 component false-veto，而不是 proposal availability。
- **Contact**：hard safe-benefit certificate 继续作为“是否授权 intervention”的最终 material criterion，但不是唯一 dense ranking target。模型应先学会对 prefix action 的 continuous post-contact recoverability potential 排序，再由 exact physical certificate 决定是否达到可授权改善。当前数据没有独立 secondary-collision probability 标签，因此 v48.52 不凭空加入该 loss；二次碰撞/re-contact 更适合作为 RC0 后 closed-loop endpoint。

### v48.52 A/B 与历史 reference 复用

为减少无效计算，本版不再做四臂。严格单轴：

- **A/reference**：v48.51-B = BC-FC + smooth NAP + q-hard teacher-sign proxy；
- **B/Main**：A + PSA，唯一差异是 physical teacher sign。

默认尝试复用已有 `ocrap_v48_51_dcp_drfc_bcde_ablation_B`。只有 historical authoritative status、protocol-seal SHA、source checkpoint SHA、gate semantics、factor contract、两 variant BC-FC witness stage 全部一致时才授权复用；否则 fail-closed 自动回退 fresh A/B。复用不改变算法归因，只避免重复训练完全相同的 A。

### 数值语义不变的 runtime 优化

v48.51 telemetry 显示两张 24GB GPU 峰值仅约 919MB、平均利用率约 20%，大部分采样低于 30%；pipeline 明显受 Python/NPZ/I/O/小粒度 inference 与重复 calibration forward 限制。v48.52 新增 **standard-calibration prediction cache**：

- cache 位于当前 calibration 原子临时目录，不跨 run 持久化；
- cache identity 绑定 checkpoint SHA256 + inference-config SHA256；
- pooled `near+contact` pass 生成的原始 float score 后续 Near/Contact 标准 calibration 原样复用；
- cache miss 时仍走原 `predict_sample`；all-hit 时模型甚至无需再次 deserialize；
- threshold/delta/split bookkeeping 不进入 inference signature，因为它们不改变 model forward；
- 不启用 AMP/TF32、跨 scene 合批、候选重排或其他可能改变数值/集合语义的优化。

### v48.52 预注册 go/stop

不偷看 test roots，优先比较 B(Main)-A(reference)：

1. **Near**：至少不应丢掉 v48.51-B 的 certificate recall 0.333 / harmful UCB 0.042 量级；重点看 DRS safe-positive false-veto、development joint sign 与 precision/centering是否改善。
2. **Contact**：首先看 teacher/native physical-sign alignment 是否改善，而不是只看最终 recall；若 physical sign改善但 final learned sign仍差，触发下一版 evidence-centering，而不是再改 certificate transport。
3. **Safety**：只有 Main authoritative RC0 才执行 gate-generated `NEXT_COMMANDS.txt`；RC20 继续 fail-closed，不人为放宽 Natural gate。
4. **Attribution**：A/B identity 不一致、reference reuse contract失败且 fresh A未完成、PSA factor/stage contract不完整，均视为 engineering failure，禁止算法归因。

### 明确新增 stop signal

- **BC-NAP Main path：STOP**。不再重复 hard-material-sign/smooth-deadband downstream transport；保留 smooth NAP。
- **BC-FC weight/temperature search：PAUSE**。在 PSA correctness 未验证前不搜索 sign weight、order weight、temperature。
- **RC20 threshold relaxation：STOP**。不能为解锁 Safe而降低 precision/harm gate。
- 继续禁止：MC-NCP、exact-only NAP、top-k/candidate/macro 扩张、aggressive oversampling、hardest-negative population distortion、generic pairwise/listwise、learned admission residual、regime-conditioned policy/router/threshold/budget、broad encoder fine-tuning，以及 changelog 已记录的其余无效路线。

### 代码落地与工程合同

- `boundary_complete_frontier_calibration_loss` 新增 PSA flag；只有 boundary-complete 分支接收 `teacher_m_star`，legacy v48.50 DEFC 接口保持不变。
- PSA teacher DRS 复用 evaluator 相同 semantics：q 选择 option、m_star 判 physical success、root probs 聚合。
- checkpoint training cfg、witness stage contract、factor cache identity 与 model/training contract 全部接线 PSA flag。
- 新增 `check_v48_52_psa_contract.py`，在 certificate 前 fail-closed 校验 Balanced/Precision witness checkpoint 与 stage metadata。
- 新增 `check_v48_52_reference_reuse.py`、A/B comparator、RC0-only post-gate wrapper 与 two-GPU launcher。
- standard calibration 新增 checkpoint/config-SHA prediction cache，测试要求 cache hit 不再次 model-forward，并保持 calibration 数值语义。

## v48.51 — DCP-DRFC-BCDE / BOUNDARY-COMPLETE DECISION EQUIVALENCE (2026-08-18)

**类别：由修复后、可归因的 v48.50 A/B/C/D authoritative 2×2 直接触发的机制升级。继续保持一个 regime-agnostic planning primitive；Safe / near-contact / contact 只作同一 normal→critical continuum 上的数据与评测分层。没有增加 regime identifier/router、regime-specific policy/threshold/loss/budget、proposal top-k、candidate family、learned admission head 或重采样。**

### v48.50.1 工程事故已经被本次新结果覆盖；本轮四臂可做算法归因

本次新上传结果是在 `ComponentVetoTolerances` hotfix 后完整重跑得到。A/B/C/D 均为 authoritative `RC=20`、`pipeline_valid=true`、certificate executed、Natural gate evaluated、`test_roots_read=false`；2×2 attribution contract valid，四臂 calibration/source/protocol identity 一致。因此本条目的 B-A、C-A、D-B-C+A 都是算法证据，不能再套用 v48.50.1 那次 RC30 的工程结论。

v48.50 Precision certificate / development 的核心结果：

| Arm | 机制 | Near cert recall | Near harmful UCB90 | Near dev joint sign | Contact cert recall | Contact harmful UCB90 | Contact dev joint sign |
|---|---|---:|---:|---:|---:|---:|---:|
| A | old DRFC + smooth NAP | **0.222** | 0.112 | 0/19 | 0 | 0.567 | 0/37 |
| B | A + DEFC | **0.222** | **0.042** | **4/19** | **0.050** | 0.382 | 0/37 |
| C | A + exact NAP | 0 | 0.064 | 0/19 | 0 | 0.362 | **3/37** |
| D | DEFC + exact NAP | 0.111 | **0.035** | 1/19 | **0.050** | 0.292 | **4/37** |

四臂都仍为 RC20；这里的“改善”只表示机制信号，不表示已经通过 Natural gate。

### v48.50 的正确归因：exactness 必须分工，不能整体替换 smooth geometry

#### 1. DEFC：保留机制原则，但拒绝当前“hard coordinate 同时承担 sign + magnitude”实现

B-A 在 Near 给出真实正向信号：development recall `0 -> 0.375`、joint semantic eligible `0/19 -> 4/19`，certificate harmful-selected UCB90 `0.112 -> 0.042`；Contact 也首次选到 `1/20` positive，UCB90 `0.567 -> 0.382`。这证明 **deployed exact boundary 必须进入 upstream calibration**。

但 B 不是 clean win：Near candidate safe-positive AUC `0.527 -> 0.431`，precision `0.049 -> 0.026`；Contact candidate/proposal safe-positive AUC `0.681/0.627 -> 0.617/0.595`。更重要的是，DEFC 把 Near harmful DRS false-safe 从约 `73.5% -> 37.5%` 的同时，在 certificate Near 引入 `7/16` DRS safe-positive false-veto；Contact development DRS safe-positive false-veto也从 `0/37 -> 11/37`。因此 v48.50 DEFC 的问题不是“exact boundary 没用”，而是 **把量化的 hard DRS 同时拿来回归连续 magnitude/order，发生 over-rotation**。

结论：accept **exact sign supervision**；reject **hard-coordinate magnitude regression**。

#### 2. Exact NAP：作为 full downstream replacement 明确 reject，但 hard exact sign 本身有信息

C-A 让 Near certificate recall `0.222 -> 0`，Contact candidate safe-positive AUC `0.681 -> 0.579`、proposal safe-positive AUC `0.627 -> 0.464`。这直接证明 v48.49-C 的 smooth boundary NAP 并非“错误 proxy”；它提供了 hard DRS equivalence class 内真实有用的 local ordering/tie resolution。

同时 C 的 Contact development final positive sign从 `0/37 -> 3/37`，D 到 `4/37`；A 自己的 Contact exact native sign 也已有 `3/37`，只是 smooth final transport把它压回 0。说明 **hard exact coordinate 对 material sign 有信息，但不应独占 ranking**。

结论：reject **Exact-NAP full overwrite**；retain **smooth NAP ordering**，并把 hard exact certificate降为 material-sign anchor。

#### 3. D/Main：没有证明“两个 exact 机制叠加就会互补”

Near 的 interaction 对 development joint sign 为负：B 已有 `4/19`，D 只剩 `1/19`；Contact 虽 D 到 `4/37`，但 certificate UCB90 仍 `0.292 > 0.25`，recall 仅 0.05。D 不支配 B/C，说明 exact-only downstream transport会抵消 upstream 的一部分收益。v48.51 不再直接复用 D。

### 当前 dominant bottleneck

1. **Boundary quantization vs local ordering conflict（首要机制瓶颈）**：hard DRS 对 deployed sign 是正确坐标，但它把同一 hard state 内的 q-depth 全部压平；smooth DRS 有 ranking 信息，却不能单独保证 material decision sign。当前模型没有显式把这两个职责分开。
2. **Benefit precision / centering 是当前直接的 Natural-gate bottleneck**：四臂 Precision development 的 `min_precision_lcb` 都失败。最好的 Near development 是 B：raw precision `0.088`、LCB90 `0.043`，距离协议要求的 fit `0.50` / verify `0.40` 很远；Contact development 四臂 precision 都为 0。也就是说当前不是简单“recall 不够”，而是大量 neutral/non-beneficial candidate 仍跨过共享 admission rule。
3. **Contact sign/centering 仍是最大 regime-level empirical gap**：最好 Contact certificate recall 只有 0.05；development joint sign最多 4/37，远低于前一版预注册的第一阶段 `>=6/37`；Main certificate harmful UCB90 仍 0.292。
4. **DEP/GAP transport 仍存在 sensitivity/specificity trade-off**：Contact safe-positive 的 deployability/gap-quality false-veto 仍高，DEFC 只在部分坐标上改善 harmful false-safe，没形成同时降低两类错误的完整 Pareto improvement。
5. **proposal availability 不是当前主因**：所有 arm development 都报告 proposal oracle feasible，Near 有 9 个、Contact 有 20 个 safe-positive proposal groups。继续扩大 top-k/candidate/macro width 会违反已有 stop signal。
6. **Safe 论文证据仍未解锁**：standard calibration valid，但所有 Main 都是 RC20，因此 scene-disjoint paired Safe non-inferiority 与 authoritative closed-loop 仍被 fail-closed post-gate 阻断。不能把“Safe calibration valid”写成“Safe policy non-inferior”。

### v48.51 方法主线：Boundary-Complete Decision Equivalence (BC-DE)

论文主线升级为：

`recovery-sufficient roots -> observation-consistent legal recovery -> OC-MERO certificate -> boundary-complete decision-equivalent transport -> non-compensatory calibrated admission`。

**Observation consistency** 约束 recovery decision 不使用 hidden future/root identity；**Boundary-complete decision equivalence** 进一步要求 transport 同时保留两类 decision-sufficient information：

- **material boundary sign**：由部署时真正消费的 hard/exact certificate 决定；
- **within-equivalence-class order**：当 hard certificate本身无法区分候选时，由与同一零边界一致的 smooth boundary geometry提供局部排序。

这不是“hard + smooth 两个模块加权融合”，而是一个职责分解：**hard owns sign; smooth owns unresolved order**。Safe/Near/Contact 不参与该分解，三者共享同一个 primitive、相同参数和相同 materiality boundary。

### 因素 X：BC-FC / Boundary-Complete Frontier Calibration

只更新既有 `margin_head`，没有新增参数。把 v48.50 DEFC 的一个 loss channel拆成两个语义通道：

- **sign channel**：hard predicted DRS（forward exact + backward STE）、`sigmoid(R_dep)`、`exp(-relu(gap))` 与 exact PCD；只做 balanced sign BCE，确保物理 zero crossing / material boundary 不被 proxy 改写；
- **order channel**：boundary-resolved smooth DRS、相同 DEP/GAP 与 smooth PCD；做 symmetric SmoothL1 magnitude regression，保留 q-depth 和 candidate-relative ordering。

因此不再要求 discontinuous hard DRS 的数值幅值拟合 teacher 幅值；它只承担它真正可靠的职责——decision sign。旧 v48.50 `decision_equivalent_frontier_calibration_loss` 在 v48.51 2×2 中明确关闭，不与 BC-FC 堆叠。

### 因素 Y：BC-NAP / Boundary-Complete Native Advantage Preservation

保留 v48.49 smooth NAP，并新增 parameter-free material-sign/deadband transport。令：

- `d_exact = V_exact(candidate)-V_exact(nominal)`；
- `d_smooth = V_smooth(candidate)-V_smooth(nominal)`；
- `g = positive_gain = 0.015`（沿用既有全局物理 materiality boundary，不新增可调阈值）。

规则为：

- 若 `d_exact >= g`：hard certificate 已明确 materially positive，`d_BC=max(d_exact,d_smooth)`，禁止 smooth 把正号改负；
- 若 `d_exact <= -g`：hard certificate 已明确 materially negative，`d_BC=min(d_exact,d_smooth)`，禁止 smooth 把负号改正；
- 若 `|d_exact| < g`：exact certificate位于 material equivalence band 内，本身分辨率不足，令 `d_BC=d_smooth` 保留 local ordering；
- 最终 `benefit_margin=d_BC-g`。

BC-NAP 不增加 head/residual/参数；deadband 直接复用已有 `positive_gain`，因此不是新 threshold search。它专门针对 v48.50 观测到的“Near 需要 smooth tie-resolution，而 Contact 有部分 exact positive sign 被 smooth erase”的矛盾。

### v48.51 严格 2×2

- **A:** v48.50-A reference = old DRFC + smooth NAP。
- **B:** A + BC-FC，只测 upstream boundary-complete calibration。
- **C:** A + BC-NAP，只测 downstream hard-sign/smooth-order transport。
- **D/Main:** BC-FC + BC-NAP。

所有 arm 固定：NCP=true、smooth NAP=true、MC-NCP=false、v48.50 exact-only NAP=false、v48.50 old DEFC=false、ROCT/top-k/source/dataset/calibration/risk protocol 不变。`D-B-C+A` 仍只在 attribution contract valid 时解释。

### v48.51 预注册 go/stop

第一阶段目标仍以 development/certificate 为准，不偷看 test：

- **Near:** D Precision certificate recall `>=0.25`，harmful-selected UCB90 `<=0.25`；development joint sign至少不低于 B(v48.50) 的 `4/19`，且 candidate safe-positive AUC 不应再出现 v48.50-B 的明显塌陷。
- **Contact:** development joint sign至少 `>=6/37`；certificate recall `>=0.10`、harmful UCB90 `<=0.25`。这是进入 Safe/closed-loop 前的最小 mechanism screen，不是最终论文目标。
- **Safe:** standard calibration valid 是必要条件；只有 D/Main authoritative RC=0 且自动生成 `NEXT_COMMANDS.txt` 后，才允许 scene-disjoint paired Safe non-inferiority + stress/closed-loop。
- 若 **BC-FC(B)** 仍出现 v48.50-B 型 DRS safe-positive veto/排序塌陷，停止 frontier loss 权重/temperature 搜索，直接转入 predicted-root probability calibration 与 teacher/native component correctness audit。
- 若 **BC-NAP(C)** 不能在保留 A 的 Near recall/AUC 的同时提高 Contact sign，则说明 exact-vs-smooth transport不是主瓶颈，停止 admission/value transport变体，转入 teacher PCD decomposition、DEP/GAP normalization 与 recovery-witness/root-probability calibration。
- 若 B/C 各自有正向主效应但 D 仍有强负交互，优先检查共同 `margin_head` 对 DRS/DEP/GAP 的 gradient conflict；不要再通过新 router/regime conditioning 规避冲突。

### 明确不重复的历史路线

继续禁止：MC-NCP tolerance 微调、exact-only NAP、旧 v48.50 hard-coordinate magnitude DEFC、threshold-grid densification、proposal top-k/candidate/macro width 扩张、aggressive positive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、full joint Stage-2、learned admission residual、one-sided safe-positive component penalty、unbounded factors、frontier-tanh、full component factorization/partial pooling/rank skip、POET、SOWR、DWOK、broad encoder fine-tuning，以及任何 Safe/Near/Contact-conditioned policy/router/threshold/budget。

### 代码落地

- 新增 `boundary_complete_frontier_calibration_loss`，把 exact sign supervision 与 smooth magnitude/order regression分离；train fast-path 与 general path 均接通。
- 新增 `direct_recovery_evidence_native_boundary_complete_advantage_preservation`；parameter-free，要求 NAP 已启用，并与 exact-only NAP fail-closed 互斥。
- checkpoint metadata、inference reconstruction、model contract、training shell、factor-cache settings、v48.47 nested witness stage isolation全部接通新 flag。
- 新增 `run_v48_51_dcp_drfc_bcde_ablation_arm.sh`、两 GPU launcher、2×2 comparator 与 RC0-only post-gate wrapper；输出目录独立为 `ocrap_v48_51_dcp_drfc_bcde_*`。
- comparator额外发布 diagnostic-only `boundary_complete_adv_*`，但不读 test roots、不进入 policy。
- 新增 v48.51 regression：BC-NAP material sign不能被相反 smooth ranking翻转；hard-equivalence band 内必须保留 smooth ordering；BC-NAP不增加参数；BC-FC 对同一 hard pattern仍保留 boundary-depth magnitude信息；2×2 factor isolation 与 non-regime contract fail-closed。

## v48.50.1 — CALIBRATION NATIVE-DIAGNOSTIC ENGINEERING HOTFIX + INFERENCE-ONLY SPEEDUP (2026-08-18)

**类别：纯工程修复与数值语义不变的推理开销优化。DCP-DRFC-DE 算法、v48.50 A/B/C/D 2×2 因素、模型参数、loss、dataset/split、proposal top-k、risk budget、threshold、calibration protocol、Natural gate、输出目录与双 GPU 执行指令均不变。**

### 本次上传的 v48.50 A/B/C/D 全部禁止做算法归因

四个 arm 都是 authoritative `RC=30`、`pipeline_valid=false`，统一停在 `certificate` stage；Balanced/Precision 的训练/适配路径已经返回 RC=0，但 near/contact development diagnostic 在 calibration 产物写出前异常退出，导致 `calibration/direct_value_risk_{near,contact}_v48.json` 缺失。A/B/C/D、Balanced/Precision、Near/Contact 的 traceback 完全同型，因此这不是 DEFC/E-NAP 的算法负结果。

唯一一致异常为：

```text
TypeError: 'ComponentVetoTolerances' object is not subscriptable
  tools/calibrate_policy_risk_v48.py:755-757
```

### 根因与修复

`ComponentVetoTolerances` 是 frozen dataclass，字段为 `.drs / .deployability_gate / .gap_discount / ...`。v48.50 新增的 native-certificate diagnostic 却误写为 `component_tolerances[0/1/2]`。该段只做 candidate-vs-nominal native component margin 诊断，但它位于 calibration JSON 原子提交之前，所以 Python 异常被 fail-closed controller 正确提升为 RC30，并连带使四臂都看起来像 `ENGINEERING FAILURE`。

修复为 named-field access：

- hard DRS margin 使用 `component_tolerances.drs`；
- deployability margin 使用 `component_tolerances.deployability_gate`；
- gap-quality margin 使用 `component_tolerances.gap_discount`。

新增 v48.50 regression，禁止 calibrator 再出现 `component_tolerances[...]`，并要求三个 named field 同时存在。修复不改变任何 certificate 数学定义或阈值。

### 工程与逻辑审计结论

- v48.50 factor contract 仍是严格 2×2：A=old DRFC+smooth NAP，B=+DEFC，C=+Exact NAP，D=DEFC+Exact NAP；MC-NCP 四臂均关闭。
- D/Main model/factor contract 继续 fail-close `direct_recovery_value_regime_conditioning=false`、`strategy_regime_conditioning=false`、`test_roots_read=false`。Safe/Near/Contact 只作 dataset/evaluation strata。
- calibration controller 的临时目录、必需 artifact 检查、shared-rule SHA、Balanced/Precision 子进程隔离与最终目录原子替换逻辑保持不变；unexpected RC 继续规范化为 RC30。
- post-gate wrapper 重新做 synthetic contract：RC20 被拒绝且不执行 `NEXT_COMMANDS.txt`；RC0 + valid D factor contract 才授权执行。
- 当前根执行指令仍写入原 v48.50 输出目录：`ocrap_v48_50_dcp_de_ablation_A/B/C` 与 `ocrap_v48_50_dcp_de_main`；没有改输出目录。
- 搜索当前执行路径未发现第二种 traceback/failure signature；上传四臂的所有 Python traceback 都收敛到同一 dataclass 下标错误。

### 数值语义不变的运行优化

纯推理入口 `predict_samples` / `predict_sample` 从 `torch.no_grad()` 改为 `torch.inference_mode()`。模型、batch/group 组织、候选顺序、算子、dtype 和输出转换均不变；同一 synthetic candidate set 上与原版逐字段 bitwise identical，`r_dep/r_orc/gap/q/root_probs/c_star/margins/direct value/std` 最大绝对差均为 0。没有启用 TF32、AMP、跨 scene 合批、候选重排或其他可能改变结果的优化。A30 上具体吞吐提升需实机测量。

### 修复后验证

- v48.47--v48.50 algorithm-focused regression：`36 passed`（包含本次新 regression）。
- inference hotpath 定向测试：`1 passed`；原版 vs `inference_mode` synthetic 输出 bitwise-equivalent。
- `python -m compileall -q src tools tests`：PASS。
- 仓库全部 `*.sh` 共 115 个：`bash -n` PASS；v48.50 根双 GPU 指令：`bash -n` PASS。
- post-gate synthetic：RC20 blocked / RC0 authorized：PASS。
- 先前已完成的 active terminal/stage-isolation suites 共 `93 passed, 1 skipped`；一次把更多 slow/historical tests 合并在当前容器执行时触及 120 s 工具上限，因此仍不声称 full historical pytest 全通过。历史缺失脚本/旧版本测试债务不在 v48.50 active path。
- 当前环境没有用户 `/data0/...` datasets 与 A30，因此不能在此做真实训练/calibration end-to-end；真实结果必须删除旧四个结果目录后在目标机器重跑原 v48.50 双 GPU 指令。

### 重跑与归因规则

删除旧四个 v48.50 结果目录后，继续执行根目录 `OC-RAP-v48.50-DCP-DRFC-DE-two-GPU-run-commands-ZH.txt`。只有 arm authoritative RC∈{0,20}、`pipeline_valid=true`、certificate/gate executed 且 factor identity valid 时，才允许解释 `B-A`、`C-A`、`D-B-C+A`；不要从本轮 RC30 结果推导任何 DEFC/E-NAP 算法结论。

## v48.50 — DCP-DRFC-DE / DECISION-EQUIVALENT CERTIFICATE TRANSPORT (2026-08-17)

**类别：由 v48.49 authoritative 2x2 直接触发的机制升级。模型仍为 DCP-DRFC；没有增加 learnable head、Safe/Near/Contact identifier、regime router、regime-specific threshold/loss/budget、proposal top-k、candidate family 或重采样。所有 arm 继续使用同一 observation-class policy/certificate。**

### 为什么不是继续调 v48.49，而是修“decision equivalence”

v48.49 的新 authoritative 结果已经触发上一版预注册 stop rule：C/NAP 有排序收益，但 C/D 仍不能稳定形成 safe-positive 的最终 joint sign；B/MC-NCP 明确失败。因此本版停止继续扩大 selector/OCAF capacity，转入 **teacher PCD correctness + predicted-vs-teacher native-coordinate calibration**。

代码级审计发现三个可证伪的 semantic mismatch：

1. 当前 teacher/evaluator 的精确 PCD 是 `DRS_hard × sigmoid(R_dep) × exp(-max(gap,0))`，其中 `DRS_hard` 来自 `q_best>=0` 的 deployable recovery success；但 v48.49 NAP 的实现实际使用 native certificate 第 3 维 `shared_feasible_mass`（smooth boundary DRS）而不是第 1 维 hard DRS。它与 teacher **共享 zero crossing，但不是同一个 decision coordinate**。
2. v48.49 MC-NCP 同样把 paper/deployment hard DRS 替换成 smooth boundary mass，并直接把 native gap-quality 送入 non-compensatory veto。新结果表明“共享 zero boundary”不足以保证 candidate-vs-nominal margin 的 sign/order；B/D 的 DRS 与 gap false-veto 都明显恶化。
3. v48.47/49 DRFC witness 的 predicted DRS 是 smooth sigmoid-q aggregation，而且使用 **teacher root probabilities**；部署 native DRS 则是 hard `q_best>=0`、使用 **model-predicted root probabilities**。旧 loss 也没有显式校准 exact gap-quality 与 exact PCD advantage。因而训练目标与最终部署坐标仍存在 train/inference semantic gap。

因此 v48.50 把论文/算法原则收敛为：**same zero crossing is necessary but not sufficient; the coordinates consumed by admission must be decision-equivalent to the deployed/teacher certificate.**

### 论文主线：Observation-Consistent + Decision-Sufficient Recoverability

统一链条保持为：

`recovery-sufficient roots -> observation-consistent legal recovery -> OC-MERO certificate -> decision-equivalent certificate transport -> non-compensatory calibrated admission`。

Observation consistency 解决 hidden-future identity 不可用于 recovery choice；Decision sufficiency/decision equivalence 解决同一个物理 certificate 在 learned/proxy transport 中不能被再次改写 sign、ordering 或 boundary geometry。Safe/Near/Contact 只是 normal-to-critical continuum 上的数据分层、evaluation strata 与 error analysis，不是三套 policy。

### 新基线：冻结 v48.49-C，永久关闭本轮被否定的 MC-NCP

四臂全部固定：NCP=true、DRFC=true、NAP=true、MC-NCP=false、ROCT/top-k/source/calibration/risk protocol 不变。A 直接复现 v48.49-C 的 smooth-NAP reference。这样不会把已知失败的 B 继续带入 Main。

### 因素 X：Decision-Equivalent Frontier Calibration (DEFC)

只更新既有 `margin_head`，不新增参数。forward coordinate 与部署完全一致：

- `DRS_pred = Σ_i p_pred(i) * 1[max_l q_pred(i,l) >= gamma]`；
- `DEP_pred = sigmoid(R_dep_pred)`；
- `GAPQ_pred = exp(-max(gap_pred,0))`；
- `PCD_pred = DRS_pred * DEP_pred * GAPQ_pred`。

teacher 侧使用对应的 hard DRS / teacher root probabilities / teacher R_dep,R_orc。对每个 candidate-vs-nominal group 同时校准三个 harmful margins 与 exact PCD benefit margin，并继续使用 symmetric SmoothL1 + balanced sign BCE。hard DRS 的 **forward value 保持 exact hard event**；梯度仅使用 straight-through sigmoid surrogate，因此训练不会把部署坐标偷偷改回 smooth DRS。

与旧 DRFC 的关键差异不是“更大 loss”，而是 **predicted root weights、hard forward DRS、gap-quality、exact PCD 都与 inference/teacher 同语义**。

### 因素 Y：Exact Native Advantage Preservation (E-NAP)

v48.49-C 的 NAP 保留为 A/B baseline；Y 只改变 final benefit transport：

`V_exact = DRS_hard * sigmoid(R_dep) * exp(-max(gap,0))`

`benefit_margin = V_exact(candidate) - V_exact(nominal) - 0.015`。

仍然是 deterministic overwrite，不增加 head/residual。这个轴专门回答：v48.49-C 的排序收益是否能在改成 teacher/evaluator exact PCD 后保留，并把 absolute sign/centering 拉回正确语义。若 exact 版本单独变差，说明 smooth boundary mass 确实提供了有用 tie/ranking resolution；那下一步只能做 **sign-preserving refinement**，不能再让 smooth coordinate决定 admission sign。

### 严格 2x2

- **A:** v48.49-C reference = old DRFC + smooth NAP。
- **B:** A + DEFC。只测 upstream decision-equivalent calibration。
- **C:** A + E-NAP。只测 downstream exact PCD transport。
- **D/Main:** A + DEFC + E-NAP。

`D-B-C+A` 只在四臂 authoritative RC∈{0,20}、pipeline_valid、factor identity 一致时解释。

### 新增 diagnostic-only readout

v48.50 在 proposal rows 额外发布 `[hard DRS, sigmoid(R_dep), smooth boundary DRS, gap quality]` 的 native certificate、candidate-vs-nominal native component margins、`native_exact_adv_margin` 与 `native_smooth_adv_margin`。这些字段 **不进入 policy**，只用于直接回答：

- DEFC 是否降低 predicted native DRS/DEP/GAPQ 的 safe-positive false-veto；
- hard-vs-smooth PCD 在 safe-positive 上有多少 sign disagreement；
- E-NAP 失败时究竟是 exact semantic 本身没有排序力，还是 upstream coordinate 仍未校准。

### 预注册 go/stop

- **Near:** D 的 Precision recall 首先要求 `>=0.25`（目标 0.30 左右），harmful-selected UCB90 `<=0.25`；native deployability false-veto 应从 v48.49-C 的 `8/16` 降到 `<=6/16`，且 candidate/proposal safe-positive AUC 不低于 C。
- **Contact:** 第一阶段不是盲目追 selected count，而是 development safe-positive joint sign 至少 `>=6/37`、certificate recall `>=0.10`、harmful UCB90 `<=0.25`；native deployability false-veto目标 `<=14/31`。最终论文目标再推进到 recall 0.20--0.30 与 post-contact closed-loop metrics。
- **Safe:** standard calibration 继续必须 valid；只有 D/Main authoritative RC=0、Natural gate 真通过并自动生成 `NEXT_COMMANDS.txt` 后，才能跑 scene-disjoint paired Safe non-inferiority + stress/closed-loop。
- 若 B/D 仍不能把 Contact development sign 拉起来，**停止 selector/OCAF/threshold/top-k 搜索**，按 stop rule 转入 teacher margin/constraint normalization、root-probability calibration、recovery-option coverage 与 teacher PCD recomputation audit。

### 工程落地与防串线

- 新增 `decision_equivalent_frontier_calibration_loss`；train/witness path、config plumbing、checkpoint reconstruction、model contract、factor-cache identity 已接通。
- 新增 `direct_recovery_evidence_native_exact_advantage_preservation`；要求 NAP 已启用，且 parameter-free。
- nested DRFC witness 显式把 NCP/MC-NCP/NAP/E-NAP 全部 stage-locally 关闭，防止父级 DCP flag 再次泄漏造成 v48.49.1 类 RC30。
- 初版 v48.50 工程审计发现并修复一个真实版本串线：arm 的默认 `OUTPUTDIR` 仍指向 `ocrap_v48_49_*`，launcher 却按 `v48_50_*` 读取。现已改为完全独立的 `ocrap_v48_50_dcp_de_*`，scheduler/runtime/comparator schema 也统一改为 v48.50，并加入 regression guard。
- full D/Main 后续闭环由 `run_v48_50_postgate_if_authorized.sh` fail-closed：MC-NCP 必须 off、DEFC/E-NAP 必须 on、strategy_regime_conditioning=false、test_roots_read=false、authoritative RC 必须为 0。
- DEFC prediction-side validity mask 只依赖部署阶段可用的 root/option validity；teacher finite/NaN 只在 target side 生效，防止 teacher missingness 泄漏。
- model contract 新增 `direct_recovery_value_regime_conditioning=false` 的 fail-closed 核验；v48.50 新 loss/model/policy 路径不读取 Safe/Near/Contact bucket 作为策略输入。
- 开发中一次文本替换曾误伤 legacy v48.47 `teacher_mask` scope，已修复并由 v48.47 regression 覆盖；不保留该临时错误。
- 最终局部验证：v48.47--v48.50 algorithm regression `35 passed`；terminal/stage-isolation `19 passed`；`compileall` PASS；114 个 `scripts/*.sh` 全部 `bash -n` PASS；RC20/RC0 post-gate fail-closed synthetic contract PASS。
- release `SHA256SUMS.txt` 在最终代码落地后重新生成并校验，避免沿用 v48.49 的 stale hashes。

### 明确继续禁止的方向

MC-NCP 本轮进入 reject list，不再继续调 smooth boundary DRS/gap tolerance；同时继续禁止历史 changelog 已经有 stop signal 的 POET、SOWR、DWOK、unbounded factors、learned admission residual、top-k/candidate expansion、aggressive oversampling、generic pair/listwise stacking、broad encoder fine-tuning、regime-conditioned router/policy/threshold/budget。

## v48.49.2 — AUTHORITATIVE DCP-DRFC RESULT ADDENDUM / SUPERSEDES v48.49.1 ALGORITHMIC EVIDENCE (2026-08-17)

**类别：对用户当前上传的 v48.49 A/B/C/D 新结果做 authoritative 归因。该结果与 v48.49.1 记录的历史 witness-stage RC30 不是同一轮；v48.49.1 仍作为工程事故记录保留，但不再代表当前 DCP 算法结果。**

### 四臂当前都可归因

A/B/C/D 均为 authoritative `RC=20`、`pipeline_valid=true`、certificate executed、Natural gate evaluated、`test_roots_read=false`，2x2 attribution contract valid。四臂都属于 **Natural gate 的算法失败**，不是 OOM、stage crash、缺文件或 flag isolation failure。

Precision certificate 核心结果：

| Arm | Near recall | Near harmful UCB90 | Contact recall | Contact harmful UCB90 | 结论 |
|---|---:|---:|---:|---:|---|
| A | 0.111 | 0.043 | 0 | 0.310 | v48.49 reference |
| B / MC-NCP | 0 | 0.591 | 0 | 1.000 | 明确负向 |
| C / NAP | **0.222** | **0.112** | 0 | 0.567 | 唯一正向机制信号 |
| D / MC-NCP+NAP | 0 | 0.465 | 0 | 0.770 | 被 MC-NCP 拖垮 |

Balanced 也给同方向：只有 C 在 Near 选到 `1/9` positive（recall 0.111）；A/B/D Near recall 都为 0，Contact 四臂仍为 0。

### MC-NCP：reject，不再“调强一点”

B 相对 A：Near safe-positive DRS false-veto `0/16 -> 7/16`，gap `5/16 -> 14/16`；Contact DRS `0/31 -> 4/31`，gap `6/31 -> 28/31`。harmful DRS false-safe 虽下降，但代价是 safe-positive 排序/准入整体坍塌；D 与 B 同方向，说明 negative effect 不是 NAP 可以补偿的。**因此 MC-NCP 的“smooth coordinate 直接替换 deployed coordinate”假设被结果否定。**

### NAP：accept mechanism signal，但不把当前公式当 final

C 把 Near Precision recall 从 `0.111 -> 0.222`，candidate safe-positive AUC `0.469 -> 0.527`；Contact 虽 recall 仍 0，但 candidate safe-positive AUC `0.620 -> 0.681`、proposal safe-positive AUC `0.529 -> 0.627`。这说明 OC-MERO native recovery value 携带比旧 learned benefit proxy 更有用的 **ordering information**。

但 current NAP 没有解决 absolute sign：development safe-positive `opportunity>=0.5` 从 A 的 Near `11/19`、Contact `8/37` 反而变成 C 的 Near `1/19`、Contact `0/37`；joint semantic sign 仍是 Near `0/19`、Contact `0/37`。所以 C 的正确解读是 **ranking-positive / centering-negative**，不是“benefit 已修好”。

### 当前 dominant bottleneck

proposal availability 仍不是问题：development 有 Near 9 个、Contact 20 个 safe-positive groups。真正瓶颈是 certificate transport 的两侧同时未闭环：

- safety side：v48.49 A/C 的 hard DRS 已做到 0 safe-positive false-veto，但 deployability 仍 Near `8/16`、Contact `18/31`；gap learned proxy 仍 Near 3--6/16、Contact 6/31 false-veto。Near development safe-positive 的 harm<=0.5 仅 A `0/19`、C `1/19`。
- positive side：C 的 native smooth PCD 有排序力但 absolute opportunity sign 偏负；Contact safe-positive opportunity `0/37`。
- Contact 是论文最明显性能缺口：四臂 recall 都为 0；C 虽提升 AUC，却没有把任何真实 positive 转成稳定 controlled-risk capture。

这正好满足 v48.49 的 stop condition：下一版转向 predicted-vs-teacher exact native-coordinate calibration，而不是继续添加 selector capacity。

## v48.49.1 — DCP DRFC WITNESS-STAGE FLAG-ISOLATION ENGINEERING HOTFIX (2026-08-17)

**类别：纯工程修复。v48.49 DCP/DRFC 算法、2x2 因素定义、模型参数、loss、dataset/split、proposal top-k、risk budget、threshold、calibration protocol、Natural gate 与原双 GPU 执行指令均不变。**

### 本次上传结果不能做 v48.49 新算法归因

- A 为 authoritative `RC=20`、`pipeline_valid=true`，certificate/Natural gate 正常执行；这只能说明 v48.48-D reference 路径本轮仍可完整运行。
- B/C/D 均为 authoritative `RC=30`、`pipeline_valid=false`，Balanced 与 Precision 都在 adaptation 的 `v48_47_recovery_frontier` 阶段、模型构造前失败。
- 三臂的唯一一致 failure signature 是 `ValueError: v48.49 native decision-complete transport requires native certificate preservation`。因此 B-A、C-A、D-B-C+A 都不成立，**禁止把本轮 B/C/D 当作 MC-NCP/NAP/DCP 的算法负结果**。

### 根因：历史 witness isolation 只关闭了 NCP，没有关闭 v48.49 新增的依赖 flag

`run_v48_49_dcp_ablation_arm.sh` 的外层 factor contract 是正确的：B/C/D 都保持 `native_certificate_preservation=true`，并分别打开 MC-NCP/NAP。问题发生在嵌套的 `adapt_ocrap_v48_47_dsofr_witness_stage.sh`。该历史 stage 为了只训练 paper-native `margin_head`，会显式设置 `EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=false`，但 v48.49 新增后没有同步清零：

- `EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION`；
- `EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION`。

于是 B/C/D 从父级环境继承 DCP=true，而 witness 子阶段又把 NCP=false，形成非法组合 `DCP=true + NCP=false`。`OCRAPModel.__init__` 的 dependency guard 正确 fail-closed，因此这是 **stage-local environment/config isolation bug**，不是算法、数据、GPU 或训练稳定性失败。A 两个 DCP flag 都为 false，所以不会触发。

### 修复

1. 在 v48.47 witness 子进程入口统一 `export` 三个 downstream native transport flag 为 false：NCP、MC-NCP、NAP。该脚本本身是 child process，因此不会把 false 泄漏回父级 v48.49 arm。
2. 在真正调用 `train_ocrap_v48_trac_sr.sh` 的 stage-local environment 中再次显式把三者一起置 false，形成双层 fail-closed isolation。
3. `V48_47_WITNESS_STAGE.json` 新增三个 false 字段，明确记录该 witness checkpoint 的 native-transport isolation contract。
4. 新增 v48.49 regression：任何 nested stage 若显式关闭 NCP，就必须同时关闭两个依赖 DCP flag，避免未来新增 factor 后再次发生同类父级环境泄漏。

### 其他工程审计

- B/C/D Balanced/Precision failure signature 完全一致；未发现 OOM、NaN、segfault、缺失 dataset/source checkpoint 或 GPU lease 争用的第二故障源。
- v48.49 新 flag 已覆盖 train config、checkpoint metadata、inference reconstruction、model contract 与 factor-cache contract；当前失败点只位于 DRFC witness isolation 边界。
- 原 `OC-RAP-v48.49-DCP-DRFC-two-GPU-run-commands-ZH.txt` 未修改。删除本次不完整 run 后，可直接使用原指令重跑。

### 验证

- v48.49/v48.48/v48.47 focused regression：通过。
- stage-isolation、stage-transfer、terminal-state、idempotent terminal 与既有 engineering-hotfix regression：通过。
- OCAF regression：通过（保留既有 1 skipped）。
- `compileall`：PASS；全部 `scripts/*.sh` 与原 v48.49 根执行指令 `bash -n`：PASS。
- 本 hotfix 没有任何算法参数或实验因素改动，因此在完整四臂重跑前继续冻结 v48.49 算法判断。

## v48.49 — DCP-DRFC / DECISION-COMPLETE NATIVE CERTIFICATE PRESERVATION (2026-08-17)

**类别：基于用户最新上传、已经通过 terminal-contract hotfix 后重跑的 v48.48 A/B/C/D authoritative 2x2 的算法升级。没有新增 Safe/Near/Contact identifier、router、regime-specific policy/threshold/loss/budget；proposal top-k、risk budget、calibration split 与 observation-class execution semantics 均保持不变。**

### 先纠正 v48.48.1 历史状态：本次上传的四臂已经可做算法归因

v48.48.1 条目记录的是 hotfix 之前那次 A/B `RC=30`、C/D 未执行的历史工程失败；**本次用户上传的是 hotfix 后完整重跑的新结果，不能套用旧结论。** 当前四臂 A/B/C/D 均为 authoritative `RC=20`、`pipeline_valid=true`，certificate 与 Natural gate 已执行，`test_roots_read=false`，且上传结果的 2x2 attribution contract `valid=true`。因此 B-A、C-A 与 D-B-C+A 都是有效算法证据；RC20 表示 Natural gate miss，不是工程失败。

### v48.48 给出的强机制证据：NCP 方向成立，但只保留两个坐标还不完整

Precision certificate 的关键变化：

- **Near:** A 的 deployability/DRS safe-positive false-veto 为 `14/16`、`11/16`；D/Main (NCP+DRFC) 变为 `8/16`、`0/16`。harmful-selected UCB90 也从 `0.2274` 降到 `0.0425`。
- **Contact:** A 的 deployability/DRS false-veto 为 `30/31`、`20/31`；D/Main 变为 `18/31`、`0/31`。
- DRFC witness 本身确实学得动：C/D Precision frontier validation objective 约 `0.785 -> 0.460`。因此“upstream frontier 可学习，但旧 proxy interface 阻断传导”的 v48.48 诊断得到更强支持。
- 2x2 interaction 很大：Near deployability false-veto `A14,B12,C15,D8`，interaction `-5`；Contact `A30,B27,C30,D18`，interaction `-9`。DRS interaction 更强（Near `-8`，Contact `-24`）。这支持 **NCP 与 DRFC 是互补而非简单并列堆叠**。

但最终 recovery capture 仍不够：D/Main Near certificate recall 仍 `0.1111`，Contact仍 `0`；development safe-positive `pred_adv>=0` Near `0/19`、Contact `1/37`。Natural gate 的 dominant failure layer 是 `development_rule_fit`。

### 新瓶颈不是 proposal，而是 v48.48 NCP 的“决策不完整”

v48.48 D/Main 暴露两个新的接口缺口：

1. **Hard-DRS saturation / local geometry loss.** v48.48 为了严格对齐 paper-facing DRS，把每个 observation row 的 `q_best>=0` 变成硬 indicator 后再聚合。它保留零边界，但把边界两侧的 margin depth 全部压平。结果 D/Main 的 DRS false-veto 降到 0，却同时让 harmful false-safe 激增到 Near `73.9%`、Contact `96.5%`。这不是应该继续“放松 DRS”，而是说明 admission 需要保留同一零边界附近的连续几何。
2. **Gap 与 positive advantage 没有被 native-preserve.** D/Main 中 gap 从过去不主导变成 Near `5/16`、Contact `6/31` safe-positive false-veto。更关键的是当前训练/校准的 benefit teacher 本身就是 signed PCD advantage，`PCD = DRS × sigmoid(R_dep) × exp(-gap)`；v48.48 NCP 只覆盖 DRS/R_dep 两个 harm component，positive-side benefit 仍由 learned proxy 自由重编码，所以 component veto 改善并没有自动产生正的 admission evidence。

proposal generation 继续不作为主修改方向：此前 top-5 safe-positive oracle availability 与 any-hit 已经证明好动作大多存在，本轮也没有相反证据。继续禁止 top-k/candidate/macro width 扩张与 aggressive resampling。

### v48.49 主线：Decision-Complete Preservation (DCP)

论文故事进一步收敛为：

`recovery-sufficient roots -> observation-consistent legal recovery -> DRFC-calibrated OC-MERO native certificate -> decision-complete native transport -> non-compensatory admission`。

Observation consistency 仍回答“哪些 recovery 在部署时是合法可执行的”；DRFC 回答“OC-MERO frontier 是否直接对齐决策边界”；DCP 回答“该 certificate 到 final admission 时，**负向 veto 与正向 advantage 两侧是否都保留会改变决策的 native sign/order/local geometry**”。三种 regime 仅作为同一策略从 normal 到 critical continuum 的 evaluation strata。

### 因素 X：Margin-Complete Native Preservation (MC-NCP)

仍保留 v48.48 paper-facing hard DRS 作为报告指标，不改论文 DRS 定义；但 final component transport 不再只使用 hard indicator：

- `boundary_DRS = Σ_i p_i sigmoid(q_best_i / tau_q)`，其中 `tau_q=0.35` 直接复用已经冻结、全局共享的 ROCT option temperature；它与 hard DRS 使用相同 `q_best=0` 边界，但保留边界距离。这里把它定义为 **boundary-resolved native recovery-success coordinate**，不冒充新的 paper DRS。
- `DEP_native = sigmoid(R_dep)` 保持不变。
- `GAP_quality = exp(-max(gap,0))` 与 teacher/component target 同方向。
- final harmful margin 固定为 `nominal_native - candidate_native - tolerance`，分别覆盖 DRS/DEP/GAP 三个 non-compensatory component。

X 不增加 learnable parameter、不新调 temperature、不读 regime 标签；其目的不是放宽 hard veto，而是让 admission 恢复可校准的 local ordering，并阻止 gap 继续由第二层 proxy 重编码。

### 因素 Y：Native Advantage Preservation (NAP)

positive-side 不再让 learned benefit proxy 任意改变 OC-MERO recovery value 的符号。直接构造：

`V_native = boundary_DRS × sigmoid(R_dep) × exp(-max(gap,0))`

`benefit_margin = V_native(candidate) - V_native(nominal) - positive_gain`

并使用已有 `benefit_margin_temperature` 映射为 benefit logit。当前 `positive_gain=0.015`，与现有 factor teacher 的 safe-benefit boundary 对齐。**这里是 overwrite learned benefit proxy，而不是 residual/新 head**，因此与 v48.37 HAF、v48.38/39 benefit-range 修改不同，不重复历史失败方向。

### 新的严格 2x2：所有 arm 都固定 v48.48-D 的 NCP+DRFC

- **A:** v48.48-D reference：NCP + DRFC，X=off，Y=off。
- **B:** A + MC-NCP (X)。回答 hard DRS/gap interface 是否是新的 safety-side bottleneck。
- **C:** A + NAP (Y)。回答 positive advantage proxy 是否是 `development_rule_fit` 的主瓶颈。
- **D/Main:** A + X + Y = full DCP。`D-B-C+A` 检验 safety-side 与 benefit-side native preservation 的交互。

所有 arm 继续固定 `observation_class` train/eval semantics、ROCT、top-k=5、source checkpoint、calibration protocol 与 risk budget，并在 `V48_49_FACTOR_CONTRACT.json` 明确 `strategy_regime_conditioning=false`、`test_roots_read=false`。

### v48.49 go/stop 读法

- **Near:** 优先要求 development safe-positive sign 从当前 `0/19` 真正抬起，同时 certificate recall 至少越过 `0.20`；harmful UCB90 保持 `<=0.25`。MC-NCP 应显著降低当前 D/Main 的 DRS harmful false-safe `0.739` 与 gap false-veto `5/16`，且不要把 deployability false-veto重新推回 `>8/16`。
- **Contact:** development sign 至少朝原预注册 `>=6/37` 推进，certificate recall `>=0.10` 且 UCB90 `<=0.25`；MC-NCP 必须显著修复 DRS harmful false-safe `0.965` 和 gap false-veto `6/31`。若只增加 selected 数而 safe-positive recall 不涨，视为无效。
- **Safe:** 不引入 Safe-specific policy。只有 authoritative RC=0 / Natural gate pass 后，才执行自动生成的 Safe paired non-inferiority 与 stress/closed-loop 命令。
- 若 B 修复 component geometry、C 修复 development benefit sign、D 同时改善且通过 gate，则 DCP 是非常强的 mechanism-complete story；若 C/D 仍无法产生正 benefit sign，则停止 selector/OCAF 修改，转向 **teacher PCD correctness、margin/constraint normalization 与 predicted-vs-teacher native-coordinate calibration**，而不是继续加网络。

### 数据集审计对本轮设计的额外启示

12 个 train/val/calibration/test × safe/near/contact 报告均 `failure_count=0`、scene leakage count=0，并有 Waymax runtime support。关键 critical-regime shift 是：train Near/Contact `R_dep` mean 都约 `-1.79`、negative-deployable fraction `~0.55/0.54`，而 calibration/test 明显更靠近边界（Near calibration/test mean约 `-0.51/-0.69`；Contact约 `-0.35/-0.57`），同时 calibration/test alias incompatibility 与 option diversity 较高。这个 shift 进一步支持“保留统一的 physical sign/order + 做 calibration”而不是增加 regime-specific capacity。

Safe report 中大量 warning 主要来自“缺少 targeted futures”的 validator 规则：Safe 数据本身只含 replay/reactive futures，且 artifact/alias incompatibility 为 0；这些 warning 需要在论文/数据报告中解释为 **regime-appropriate future-source contract warning**，不要误写成数据失败。建议后续单独修 validator 的 Safe contract，以免审稿材料出现 20k 级 warning 噪声，但本轮不把它混入算法因素。

### 闭环授权与 `LAUNCH_RC`

本次四臂的最终 `logs/v48_48_launcher.rc` / runtime authoritative RC 都是 `20`，不是 `0`。局部 adaptation/launcher 子步骤出现 `0` 只表示该子步骤正常，不代表 Natural gate 通过。`NEXT_COMMANDS_STATUS.json` 明确 `reason=natural_gate_failed`、`generated=false`。因此 **当前 v48.48 不应执行论文-authoritative closed-loop/test**。

v48.49 新增 `scripts/run_v48_49_postgate_if_authorized.sh`：它只有在 D/Main `AUTHORITATIVE_RUN_STATUS.authoritative_exit_code==0`、pipeline/certificate/gate contract 完整、full DCP factor contract 成立、且 controller 已生成 `NEXT_COMMANDS.txt` 时才执行后续 Safe paired non-inferiority + stress/closed-loop；RC20 会 fail-closed 拒绝。

### 工程落地与防回归

- 新 flags 已打通 train config、checkpoint metadata、inference reconstruction、model contract 与 run provenance：`native_margin_complete_preservation`、`native_advantage_preservation`、`native_gap_tolerance`、`native_positive_gain`。
- DCP 在模型层不增加参数；v48.49 regression test 验证 state_dict key 与 v48.48-D 一致、hard paper DRS coordinate 保持、MC-NCP/NAP 单调性、无 regime/bucket 输入。
- 2-GPU launcher 沿用 v48.48.1 已修复的 GPU lease + arm 并行/variant 串行 + subshell errexit 隔离，不重新引入已知 OOM/terminal-state bug。
- 当前实现额外输出 native certificate/component/benefit diagnostics，便于下一轮直接判断是 safety-side 还是 benefit-side 未传导。

### 继续禁止的历史失败方向

继续不重复 threshold-grid densification、top-k/candidate/macro expansion、aggressive positive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、full joint Stage-2、learned admission residual、one-sided safe-positive tail、unbounded factors、frontier-tanh、full component factorization、partial pooling/rank skip、POET、joint SOWR、generic obs->margin SOWR、DWOK、broad encoder fine-tuning，以及任何 regime-conditioned router/policy/threshold/budget。v48.49 的两条轴都是 **删除/替换 proxy 的确定性 native transport**，不是这些旧方向的变体。


## v48.48.1 — SERIAL CERTIFICATE TERMINAL-CONTRACT ENGINEERING HOTFIX (2026-08-17)

**类别：纯工程修复；NCP/DRFC 算法、模型参数、loss、dataset/split、top-k、risk budget、threshold、Natural gate 与 2x2 factor matrix 均不变。**

### 上传的 v48.48 A/B 为什么不能做算法归因

- A/B 均为 authoritative `RC=30`、`pipeline_valid=false`；两者 adaptation 的 Balanced/Precision exit code 都是 `0`，certificate controller 已进入证书计算，但 terminal status contract 因缺失 `dedicated_recalibration_status.json` 与 `NEXT_COMMANDS_STATUS.json` 返回 raw RC=4，随后被 controller 归一化为 RC30。
- 因为 A/B 非 pipeline-valid，且原 2x2 launcher 在第一波发现 engineering failure 后立即 `break`，C/D 根本未执行。因此本轮不满足 v48.48 预注册的四臂归因条件，**禁止把 A/B 当作 NCP 的算法负结果，也禁止做 B-A / C-A / interaction 结论。**

### 根因：serial 模式下 Bash `errexit` 状态从函数泄漏到 caller

`calibrate_v48_36_shared_certificate_pool.sh` 的 `calibrate_variant()` 为了区分 certificate 自然拒绝与工程失败，会在函数内部多次切换 `set +e` / `set -e`，并在 Near/Contact 都是有效 Natural-gate miss 时返回 `20`。v48.48 新增 `SERIAL_VARIANTS_ON_ONE_GPU=1` 后，caller 原本先 `set +e` 再直接调用函数并捕获 `$?`；但 Bash 函数与 caller 共享 shell option，函数末尾的 `set -e` 会把 parent 的 errexit 重新打开。于是 Balanced 返回 `20` 的瞬间，shell 在执行 `S0=$?` 前退出：Precision 不运行，final status writer 不运行，最终缺少上述两个 terminal JSON，status contract 再把它误分类为 RC30。

该机制与上传结果完全一致：A/B adaptation 都成功；GPU lease 时显存充足，不是 OOM；controller log 在 Balanced certificate 完成后中止；两臂都缺失相同 terminal JSON。

### 工程修复

1. serial worker 路径统一隔离 Bash shell-option 状态：certificate 的 `calibrate_variant` 与 adaptation 的 `run_variant` 都在独立 subshell 中执行并由 parent 捕获 exit code。这样函数内部 `set -e` 只能影响 subshell；Natural certificate RC20 或某个 adaptation RC30 都不会在 exit code 被记录前意外终止 parent，Balanced 失败也不会阻止 Precision 的诊断执行。对于本次 A/B，关键修复是 certificate 路径，因此 `dedicated_recalibration_status.json` / `NEXT_COMMANDS_STATUS.json` writer 能继续运行。
2. v48.48 2x2 launcher 不再在 A/B 第一波出现工程失败时 `break`。它仍然严格禁止在任一 arm 非 RC0/20 时运行 comparator/算法归因，但会继续尝试 C/D，以便一次实验获得四臂完整工程诊断，避免“第一波坏掉导致第二波完全没有证据”。
3. runtime telemetry summary 无论 2x2 是否有 engineering failure 都会生成；comparator 仍只在四臂工程有效时执行，因此不会降低因果归因门槛。
4. 新增 v48.48 回归测试：静态验证 serial 两 variant 都使用 subshell；执行一个复现 Bash `errexit` 语义的最小测试，要求两个自然 RC20 都被捕获；验证 launcher 第一波工程失败不再阻止第二波，同时 attribution 仍 fail-closed。

### 下一轮执行与解释

- 建议从 clean output 重新跑完整 A/B/C/D；不要把这次 RC30 目录做算法复用或拼接归因。shared v48.45 source 仍保持 hash-sealed，不需重训。
- 只有四臂 `authoritative_exit_code in {0,20}`、`pipeline_valid=true` 且 `OC-RAP-v48.48-NCP-DRFC-2x2-audit.json` 的 attribution contract valid，才恢复 NCP/DRFC 算法归因。
- 本 hotfix 不改变任何算法因素，所以修复前后的 RC 差异只能解释为 pipeline correctness，不应计入算法增益。

## v48.48 — NCP-DRFC / NATIVE CERTIFICATE PRESERVATION + CLEAN FRONTIER CAUSAL TEST (2026-08-14)

**类别：基于 v48.47 authoritative/engineering-mixed 结果的算法主线升级 + GPU 资源隔离修复。没有新增 Safe/Near/Contact identifier、router、regime-specific policy/threshold/loss/budget。**

### v48.47 结果状态与可归因边界

- A、B、D/Main 均为 authoritative `RC=20`、`pipeline_valid=true`，certificate 与 Natural gate 已执行，`test_roots_read=false`。因此它们是有效算法负结果。
- C 为 authoritative `RC=30`、`pipeline_valid=false`，停止在 adaptation；Balanced 的 DRFC witness 在首个训练 backward 时发生 `torch.OutOfMemoryError`，certificate/gate 未执行，因此 **C 不能用于 DRFC-alone 算法归因**。
- C OOM 不是 DRFC 数值爆炸：失败时当前进程 PyTorch 仅约 `124 MiB` allocated、`~50 MiB` reserved，但同一 A30 已有另一个进程占用约 `20.86 GiB`，另有 Precision 约 `1.68 GiB`，23.60 GiB 卡只剩 `24.44 MiB`。D/Main 的同一 DRFC stage 在另一卡完整完成并得到 RC20，证明本次 C 是 GPU 资源租约/调度工程故障。
- persistent tensor cache 已命中；10,015 train + 3,526 val 样本的 materialization 仅约 `1.25 s`，因此此前 NPZ/IO 已不再是当前主要 runtime 瓶颈。现在主要是实际 witness/factor/certificate compute 与 GPU contention。

### v48.47 最可靠算法结论

1. **DWOK 不吸收。** B 把 observation validation loss 从约 `0.7284` 降到 `0.6462`，但 Precision Near/Contact 的 DRS/deployability safe-positive false-veto 基本不动（Near DRS `11/16`、deployability `14/16`; Contact DRS `20/31`、deployability `30/31`），certificate recall 也未改善。说明更精细的 observation-kernel weighting 可以学到自己的目标，却不是当前 final admission 的主导瓶颈。
2. **当前 DRFC 实现不直接吸收，但保留其“直接对齐 decision frontier”的方向。** D 的 frontier validation loss 从约 `0.8166` 降到 `0.4538`，Contact development sign 从 reference `1/37` 提升到 `3/37`，certificate 得到 `1/20` positive；但 Near deployability false-veto反而 `14/16 -> 16/16`，Contact仍 `30/31`，Precision Contact harmful UCB90 约 `0.342`，风险不可接受。由于 C 工程失败，本轮仍缺少 clean DRFC-alone 主效应。
3. **proposal generation 仍不是主瓶颈。** v48.46 final certificate 已证明 top-5 proposal 对 Near safe-positive groups `9/9`、Contact `20/20` 可达，Contact positive-group any-hit约 `96.9%`、oracle-best hit约 `90.6%`；v48.47 没有出现相反证据。继续禁止 top-k/candidate-width/macro-count 扩张。
4. **更强接口诊断：frontier 学得动，但 final component proxy 不跟。** v48.47-D 明确显示 upstream `margin_head -> OC-MERO` 的 DRFC objective 大幅改善，而最终 certificate 中 DRS/deployability false-veto 几乎不移动。当前 factor/OCAF 链把 paper-native OC-MERO 证书压成 detached compact signature，再由 learned component head 重新预测 DRS/deployability harmful coordinates；这个 proxy-of-proxy 接口允许 sign/scale 再次坍缩。这比“继续加 downstream residual/head”更符合现有证据。

### 论文主线升级：Decision-Sufficient Observation-Consistent Recoverability + Native Certificate Preservation

- **保留 observation-consistency 作为必要的 deployability semantics，而不再把 observation aliasing 叙述为 Near/Contact 当前经验 failure 的主要来源。** 它解决 hidden branch identity leakage 的规范性问题。
- **decision-sufficiency 的可检验含义升级为 certificate preservation：** selector 真正使用的关键恢复坐标（DRS、deployability）必须从 OC-MERO 到 non-compensatory CRISP/OCAF admission 保持物理方向与零边界，不能被第二个任意 learned proxy 重新编码后反转。
- Safe/Near/Contact 仍只作为 evaluation strata。同一 OC-MERO、同一 NCP mapping、同一 risk budget 和同一 selector 在三种 regime 下运行。

### 新算法因素 X：Native Certificate Preservation (NCP)

NCP **不增加任何可学习参数**。在 final ordinal-evidence/component-veto path 中，DRS 和 deployability 两个核心 coordinate 不再由 downstream component head 自由重预测，而直接由 model 的 paper-native OC-MERO 预测证书构造：

- `DRS_native(a)`：从 observation-conditioned `q[i,l]` 取每个 row 的 best option，并以 `q_best>=0` 的**硬预测 DRS**按 root probability 聚合；与 paper-facing observation-class predicted DRS 使用同一零边界。
- `DEP_native(a)=sigmoid(R_dep(a))`：与 component teacher deployability target 同一尺度。
- 对每个 candidate，以 nominal 为 anchor 构造 harmful physical margin：
  `H_drs = DRS_native(nom)-DRS_native(cand)-eps_DRS`，
  `H_dep = DEP_native(nom)-DEP_native(cand)-eps_DEP`。
- 两个 margin 以已有 slack temperature 映射到 non-compensatory component logits；gap 与其它独立证据保持原机制。

因此 NCP 具有显式单调性：提高 candidate 的 native DRS/R_dep 绝不会增加对应 harmful logit；benefit head 也不能补偿一个 positive native veto。它删除 proxy，而不是新增 residual。

### 新一轮严格 2x2

四臂全部固定 `observation_class` training/evaluation semantics、同一个 rebuilt source、dedicated calibration protocol、top-k=5、dual-ROCT、harm budgets 和 Natural gate：

- **A:** v48.47 paper-consistent reference；NCP=off，DRFC=off。
- **B:** A + NCP；DRFC=off。`B-A` 直接检验 certificate-to-proxy interface 是否为主瓶颈。
- **C:** A + **clean DRFC-alone**；NCP=off。补回 v48.47-C 因 RC30/OOM 缺失的 DRFC main effect。
- **D/Main:** NCP + DRFC。`D-B-C+A` 检验 DRFC 是否只有在 native certificate 被 downstream preservation 后才能传到最终 admission。

强解释规则：若 B 显著降低 false-veto 且 UCB 不恶化，则 proxy interface 是主瓶颈；若 C 的 witness objective 改善但 final frontier仍不动、而 D 显著优于 C，则构成很强的“DRFC 被旧 proxy 衰减”证据；若 B/C/D 都不能移动 false-veto，则停止 downstream selector 优化，转向 teacher margin normalization、constraint scale 与 continuous recovery-option teacher coverage 审计。

### v48.48 screening / stop rule

- **Near-contact:** Precision deployability false-veto从 `14/16` 优先降到 `<=10/16`，DRS false-veto从 `11/16` 降到 `<=8/16`；certificate recall至少 `>=0.20`，harmful-selected UCB90 `<=0.25`。进一步 paper-readiness 目标仍是 recall约 `0.25-0.33`、precision LCB `>=0.40`，并由 paired/bootstrap CI 支持 TTC/clearance 改善。
- **Contact:** deployability false-veto从 `30/31` 降到 `<=24/31`，development safe-positive `pred_adv>=0` 至少 `>=6/37`；certificate recall至少 `>=0.10` 且 UCB90 `<=0.25`。进一步目标为 recall `0.20-0.30`、secondary-collision absolute reduction约 `>=2 pp`、post-contact TTC约 `+0.2 s`，并改善 stable-stop/rejoin/impact severity。
- **Safe:** 不新增 Safe-specific policy。当前仅有 standard calibration，最终论文必须补 scene-disjoint paired closed-loop non-inferiority：nominal utility/progress/comfort 不显著下降、intervention/FRA 受控。
- 上述是项目内部 CCF-A readiness bar，不是任何 venue 官方阈值。

### 工程与性能修复

1. **修复 v48.47-C GPU OOM 根因。** 2x2 launcher 仍保持 `A@GPU0+B@GPU1`、`C@GPU0+D@GPU1` 两 arm 同时运行，但默认 `V4848_VARIANT_MODE=serial`：每个 arm 内 Balanced/Precision 串行，保证每张卡同一时间只有一个训练 variant。
2. 启动每个 arm 前通过 `nvidia-smi` 建立 GPU capacity lease；serial 默认要求 `>=12,000 MiB` free，parallel debug 模式默认 `>=20,000 MiB`，不足则等待而不是盲目启动。超时按 RC30 engineering fail 处理并保存 preexisting compute-app memory provenance。
3. 2x2 默认要求 `GPU0 != GPU1`；只有显式 debug 开关才允许共享同一 GPU id，避免两个 arm 对同一卡同时“成功租约”的竞态。
4. 继续保留 persistent tensor mmap cache、selective NPZ decoding、witness fast path、30s只读 GPU/host telemetry；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 仅用于减少 allocator fragmentation，不把它当作容量修复。
5. NCP flag/tolerances 已加入 train config、checkpoint metadata、inference reconstruction、factor-cache key 与 model contract；v48.47 DRFC witness stage显式强制 NCP off，防止 D 的 downstream factor泄漏进 C/DRFC witness，从而保持2x2因果隔离。
6. NCP native DRS 使用 hard `q_best>=0` paper-facing predicted DRS，不再在 certificate interface 引入另一个 soft-DDS/DRS proxy。

### 继续禁止的历史失败方向

不重复 threshold-grid densification、top-k expansion、candidate/macro width 扩张、aggressive positive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、full joint Stage-2、learned admission residual、v48.38 one-sided tail、v48.39 unbounded factors、v48.40 frontier-tanh、v48.41 full component factorization、v48.42 partial pooling/rank skip、v48.43 POET free alias transport、v48.45 joint SOWR、v48.46 generic staged witness、v48.47 DWOK、broad encoder fine-tuning，以及任何 regime-conditioned router/policy/threshold/budget。

## v48.47 — DS-OFR / DECISION-SUFFICIENT OBSERVATION & RECOVERY-FRONTIER CALIBRATION (2026-08-13)

**类别：基于 v48.46 authoritative 2x2 的算法升级 + attribution-safe 执行优化。没有新增 Safe/Near/Contact identifier、router、regime-specific policy/threshold/loss/budget。**

### v48.46 authoritative OC-SWIC 归因：四臂均为有效算法负结果

本轮上传的 v48.46 A/B/C/D(Main) 四个 arm 均为 authoritative `RC=20`，且
`pipeline_valid=true`、certificate 已执行、Natural gate 已评估、`test_roots_read=false`；v48.46 comparator
的 attribution contract 通过。四个 arm 共享完全相同的 source checkpoint、dedicated calibration protocol、
final `observation_class` evaluation semantics、dual-ROCT、top-k=5、risk budgets 和 gate protocol。因此本轮
可以正式解释 B-A、C-A 与 D-B-C+A；RC=20 是有效 Natural-gate miss，不是 pipeline failure。

Precision selector 的核心结果：

- **A / paper-consistent reference:** Near certificate recall `0.1111`、harmful UCB90 `0.2361`，deployability
  safe-positive false-veto `14/16`；Contact recall `0`、UCB90 `0.2921`，false-veto `30/31`，development
  safe-positive `pred_adv>=0` 仅 `1/37`。
- **B / observation-class training semantics:** Near recall 仍 `0.1111`，false-veto 仍 `14/16`；Contact recall
  仍 `0`，false-veto 仍 `30/31`，development sign 仍 `1/37`。Near/Contact candidate safe-positive AUC 仅
  `+0.0022/+0.0041`，UCB90 小幅下降到 `0.2274/0.2773`。因此 v48.46 直接否定了“历史 global-one-option
  是当前 Near/Contact false-veto 的主导根因”这一更强性能假设。`observation_class` 仍作为论文正确语义保留，
  但不再作为有证据的性能模块。
- **C / sequential obs->margin witness:** Precision Contact development recall `0.0588 -> 0.1176`，safe-positive
  `pred_adv>=0` `1/37 -> 3/37`，certificate harmful UCB90 `0.2921 -> 0.2618`；但 Contact certificate recall
  仍 `0`，Near recall `0.1111 -> 0` 且 deployability false-veto `14/16 -> 16/16`。Balanced Contact 虽得到
  `0.05` recall，却伴随 UCB90 `0.3329`。因此 staged witness 只有局部 sign/risk 信号，不能整体吸收。
- **D/Main / observation-class + staged witness:** Precision candidate safe-positive AUC 达到 Near `0.3653`、
  Contact `0.5086`，Contact certificate 得到 `0.05` recall，deployability false-veto `30/31 -> 29/31`；但
  Near/Contact harmful UCB90 恶化到 `0.2959/0.3278`。两因素在 discrimination/capture 上有弱正 interaction，
  在 risk 上出现强负 interaction，故 D/Main 仍不能吸收为主算法。

Balanced selector 不改变上述结论：Near certificate recall 四臂均为 `0`；Contact certificate 最好只有 C 的
`0.05`，而风险不稳定。Safe 只完成统一 calibration（A/B gamma `1.6269`，C `1.4829`，D `1.4558`），没有
registered scene-disjoint Safe policy certificate，因此本轮不能宣称 paper-level Safe closed-loop non-inferiority。

### 最可靠的新瓶颈结论：proposal 已覆盖，失败发生在 learned witness -> frontier/admission

final certificate 的 proposal-oracle 证据显示：Near top-5 proposal 对 9 个 safe-positive opportunity group 为
`9/9` 可达；Contact 对 20 个 safe-positive group 为 `20/20` 可达，positive-group any-hit 约 `96.9%`、
oracle-best hit 约 `90.6%`。因此当前主瓶颈不是 candidate family、top-k 或 recovery macro “没有生成好动作”，
而是好 candidate 已存在却在 learned recovery-value / deployability witness、rerank/admission 中被错误否决。

component audit 更具体地把错误定位到 **DRS/deployability physical frontier 的 sign/scale collapse**：Contact
safe-positive 中 teacher deployability degradation margin 往往已经明显在安全侧（约 `-0.24~-0.38`），但 learned
component margin 被压在约 `+0.02~+0.06` 的危险侧附近，导致 `30/31` safe-positive 被 deployability veto；Near
同样为 `14/16`。DRS 也有明显 false-veto。相比之下 gap/hard-rule/harm-proxy 不是当前 safe-positive veto 主因。

因此下一步明确**不**扩大 top-k、不增加 recovery macro 数量、不放松 harm budget，也不继续给 downstream
classifier 堆 generic residual/head。现有 factor stage 已开启 dense signed component-margin regression；重复 v48.38
one-sided safe-positive component penalty 没有科学依据，继续禁止。

### 论文 motivation 的严谨升级：从 observation-consistent 到 decision-sufficient observation-consistent

v48.46 不否定论文的核心结构命题：oracle branch-wise recoverability 与 deployed observation-conditioned
recoverability 不等价，OC-MERO 的 observation-consistency 仍是必要的 deployment semantics。但两轮结果
（v48.43 POET、v48.46-B）不支持把“observation ambiguity / global-option semantics 是实际 false-veto 的主导原因”
作为更强经验主张。

v48.47 将方法论升级为 **Decision-Sufficient Observation-Consistent Recoverability**：

1. **物理 observation equivalence 定义不变。** 不修改 `y_obs`，不按 regime 改 observation threshold。
2. **Decision-weighted observation identifiability.** 对 recovery decision 不敏感的 root pair 不需要与关键 pair
   获得相同梯度预算；若两 root 各自存在 recovery、但没有相同 option 同时具有高 support，则错误合并/分离该 pair
   会直接改变可部署 recovery decision，应被优先校准。
3. **Recovery-frontier identifiability.** pointwise margin/Q reconstruction loss 降低并不等于 selector 的
   candidate-relative DRS/deployability 零边界正确；必须直接对齐最终 non-compensatory admission 使用的 frontier。

这不是替换 OC-MERO，而是把“结构上可部署”升级为“结构正确且决策边界可识别/可校准”。三个 regime 仍仅作为
training/evaluation strata，同一性质和同一算法作用于 Safe/Near/Contact。

### 新算法因素 X：Decision-Weighted Observation Kernel (DWOK)

对 teacher recovery margin 定义 smooth option support
` s_{i,l}=sigmoid((m*_{i,l}-gamma)/tau) `，pair 的 recovery-decision conflict 为
`kappa_ij = max_l s_i,l * max_l s_j,l - max_l(s_i,l*s_j,l)`，截断到 `[0,1]`。
物理 observation BCE 的 label **完全不改变**，只把 pair 权重改为 `1 + lambda*kappa_ij` 并归一化保持整体 loss scale。
高 `kappa` 同时覆盖两类真正与 OC-MERO decision 有关的错误：物理不可区分但 recovery-incompatible 的 pair 被错误
分开会产生 oracle false-safe；物理可区分且需要不同 recovery option 的 pair 被错误合并会产生 false-veto。
该权重不读取 Safe/Near/Contact label。

### 新算法因素 Y：Direct OC-MERO Recovery-Frontier Calibration (DRFC)

margin witness 不再用 v48.46 的 generic deployability/oracle/option auxiliary bundle。它仅更新 `margin_head`，直接
从 differentiable OC-MERO `q[i,l]` 构造 observation-class DRS，并在同一 scene-time candidate group 内对齐两条
selector 原生 frontier：

- deployability degradation: `sigmoid(R_dep(nominal))-sigmoid(R_dep(candidate))-eps_R`；
- DRS degradation: `DRS(nominal)-DRS(candidate)-eps_Q`。

loss 为双坐标**对称** SmoothL1 + balanced sign BCE；safe/harm 两侧同时校准，不是 v48.38 的 one-sided tail patch，
也不是已失败的 generic pairwise/listwise ranking。保留小权重 absolute margin anchor 防止 candidate-relative training
破坏全局 signed margin scale。root logits、encoder、root decoder、direct policy/OCAF/ROCT heads 全冻结。

### v48.47 scientific 2x2

四臂 training/evaluation semantics 全部固定为 paper-consistent `observation_class`：

- **A:** reference；无 DWOK、无 DRFC。
- **B:** A + DWOK（只更新 `obs_embed_head`）。
- **C:** A + DRFC（只更新 `margin_head`）。
- **D/Main:** A + DWOK，然后 freeze obs，再做 DRFC。

因此 `B-A` 是 decision-weighted observation identifiability 主效应；`C-A` 是 direct recovery-frontier calibration
主效应；`D-B-C+A` 是 interaction。四臂共享 source/protocol/gate/dual-ROCT/top-k5/risk budget。D 在 fail-closed
identity contract 通过时复用 B 的 DWOK checkpoint，只节省重复训练时间，不改变 D 的模型状态或输入。

### v48.47 pre-registered screening / stop rule

不以 train/val loss 下降作为吸收标准：

- **Near:** Precision deployability false-veto 优先从 `14/16` 降至 `<=10/16`；DRS false-veto 从 `11/16`
  明显下降；certificate recall 至少向 `>=0.20` 移动，同时 harmful-selected UCB90 `<=0.25`。
- **Contact:** deployability false-veto 从 `30/31` 降至 `<=24/31`；development safe-positive `pred_adv>=0`
  至少从 `1/37` 移动到 `>=6/37`；certificate recall 至少 `>=0.10`（主算法目标继续向 `0.15-0.20` 推进），
  harmful-selected UCB90 `<=0.25`。
- **D/Main:** 只有同时优于对应单因素的 capture/frontier 且不产生 risk interaction 才吸收；若 C 单因素最好，
  主算法只吸收 C，不为“模块完整性”强行保留 B/D。
- **Safe:** 必须保持 standard calibration 有效，并监控 gamma drift；真正 paper-level Safe 结论仍要求后续 paired
  closed-loop non-inferiority，不新增 Safe-specific route。

若 v48.47 的 C/DRFC 仍不能显著移动 DRS/deployability false-veto，则 stop rule 是进入 teacher-margin calibration、
recovery constraint normalization、option continuous-parameter teacher coverage 审计；不再增加 downstream capacity。
若 B/DWOK 不能改善 observation-conflict 子集而 frontier 也不动，则停止继续优化 observation kernel，把它保留为
结构语义模块而非性能抓手。

### Execution-equivalent 加速与工程 contract

- 两张 GPU 默认同时跑两个 ablation：`A@GPU0 + B@GPU1`，完成后 `C@GPU0 + D@GPU1`。
- 基于用户已观察到单 ablation 显存有余量，每个 arm 的 Balanced/Precision 默认可在分配到的同一 GPU 并发；若
  telemetry 显示 compute saturation/OOM，可设 `V4847_VARIANT_MODE=serial`，但 arm-level 双 GPU 并行保持不变。
- 继续复用 v48.46 persistent tensor mmap cache；cache key 只依赖输入 tensor geometry/manifest，算法因素不会造成
  无意义 cache miss。D 通过 source SHA、checkpoint SHA、train/val mix、group-index SHA 和 DWOK 超参的 fail-closed
  contract 复用 B 的 obs stage，避免重复一整个 witness stage。
- witness stage 新增严格 `witness_fast_path`：DWOK 只执行 root/observation 图与加权 observation BCE，DRFC 只执行
  root/margin/observation 图、OC-MERO 与 frontier loss；utility/direct-policy/tournament/delta/group auxiliary 等全部为零权重且
  冻结的计算不再前向。所有冻结 subtree 强制 `eval()`；回归验证各 stage 实际消费的 active outputs（DWOK: `root_logits/obs_embeddings/c_star`；DRFC: `root_logits/margins/obs_embeddings/c_star`）
  与 full path 逐元素 bit-exact，且 forward 后 RNG state 完全一致，因此该优化不改变 witness 的样本、
  loss、梯度、optimizer step 或随机轨迹。
- runtime telemetry 每次 launcher 开始前清空，避免多次运行日志混合；30 s 只读采集 GPU util/memory/power 与 host
  memory/load，结束后自动汇总，不改变 CUDA/sampler/random state。
- 修复 v48.46 launcher runtime 记账偏差：父进程先 `wait` 左 arm 会把已提前完成的右 arm 空等时间错误计入右 arm
  wall time。v48.47 在每个后台 arm 自己退出时原子记录 end/RC，再由父进程校验；这只影响性能诊断，不影响训练。
  对本轮上传结果回溯可见 D 实际约 `56.3 min` 已完成，而旧 logger 因等待 C 错记为 `120.8 min`。
- comparator 继续 fail-closed 验证 source checkpoint、protocol seal、gate manifest、factor matrix 和 no-test/no-regime
  contract；并记录 Safe calibration gamma 供跨臂 drift 诊断。

### 继续禁止的历史失败方向

v48.47 不重复：threshold-grid densification、top-k expansion、aggressive positive oversampling、hardest-negative
population distortion、generic pairwise/listwise stacking、full joint Stage-2、learned admission residual、v48.38
one-sided tail losses、v48.39 unbounded benefit/harm factors、v48.40 frontier_tanh、v48.41 full component factorization、
v48.42 partial-pooling/rank-skip、v48.43 POET free alias transport、v48.45 joint SOWR、v48.46 uniform generic
obs->margin witness、broad encoder fine-tuning，以及任何 regime-conditioned router/policy/threshold/budget。


## v48.46 — OC-SWIC / OBSERVATION-CLASS SHARED-WITNESS IDENTIFIABILITY CALIBRATION (2026-08-12)

**类别：有效算法迭代 + execution-equivalent 性能优化。没有新增 Safe/Near/Contact identifier、router、regime-specific policy/threshold/loss/budget。**

### v48.45.6 authoritative 2x2 attribution

本轮上传的 v48.45.6 A/B/C/D(Main) 四个 arm 均为 authoritative `RC=20`，且
`pipeline_valid=true`、certificate 已执行、Natural gate 已评估、`test_roots_read=false`。
四个 arm 的 dedicated-calibration protocol seal、Balanced/Precision source checkpoint
SHA256、gate protocol 与所有 development/certificate manifest SHA256 完全一致；因此这是
v48.45 SOWR 第一轮真正可进行 B-A、C-A、D-B-C+A 因果归因的有效负结果，而非工程失败。

Precision selector 的主要结果：

- A/Near certificate recall `0.1111`，harmful-selected UCB90 `0.2361`；deployability
  harmful-vs-safe-positive AUC `0.4510`，safe-positive false veto `14/16`。
- A/Contact certificate recall `0`，UCB90 `0.2921`；deployability AUC `0.5252`，false veto
  `30/31`；development safe-positive `pred_adv>=0` 仅 `1/37`。
- B(root+margin) 将 Near/Contact candidate safe-positive AUC 分别提高约 `+0.026/+0.027`，
  Contact certificate 首次得到 `1/20` positive recall `0.05`，false veto `30/31 -> 29/31`；
  但 Near/Contact certificate harmful UCB90 分别恶化到 `0.2980/0.3151`，不满足风险约束。
- C(obs-only) 的主要正信号是 certificate harmful UCB90：Near `0.2361 -> 0.1532`、Contact
  `0.2921 -> 0.2368`；Contact development `pred_adv>=0` 从 `1/37 -> 2/37`。但 Near
  certificate positive capture 从 `1` 降为 `0`，无法单独作为主算法。
- D/Main 基本退化为 B；它没有同时保留 C 的 risk suppression 与 B 的 capture signal。
  Near/Contact certificate UCB90 为 `0.2980/0.3189`，Contact development sign 又回到 `1/37`。
  因而 v48.45 joint SOWR 不吸收为成功模块。

Balanced selector 给出相同方向的弱证据：B/C/D 可把 Contact development recall 从
`0.1176` 提到 `0.1765`，但 certificate 仍只有 `0.05`，Near 仍极弱，且 risk/generalization
不稳定。结果支持“上游 witness/option semantics 有问题”，不支持继续堆 downstream capacity。

### SOWR 组件归因与保留/拒绝

v48.45.6 witness validation 显示 B/D 的 margin、deployability、oracle、option-q/admission/best-option
loss 都下降，但 `root_loss` 反而略升；C 的 observation loss 明显下降，同时有 certificate risk
suppression。由此：

1. **拒绝 root-logit recalibration。** 它没有独立正证据，且与 margin 更新绑定时 root objective
   反向漂移。v48.46 所有新 witness stage 都 byte-level 冻结 `root_logit_head`。
2. **保留 observation recalibration 作为风险侧候选。** 它只能被视作局部正信号，不能单独宣称
   解决 positive capture。
3. **保留 margin recalibration 作为 capture 侧候选。** B 的弱 AUC/Contact-positive 改善与
   margin/option loss 改善一致，但必须摆脱 root update 并验证不再用 harmful risk 换 recall。
4. **拒绝 simultaneous joint SOWR。** D 没有正 interaction，因此 v48.46 改为
   `observation -> freeze -> margin -> freeze` 的顺序识别，不再联合更新三个 witness heads。

### 根本算法语义审计：历史 global-one-option 与论文 OC-MERO 不一致

本轮对论文、`src/ocrap/algorithms/ocmero.py` 与历史 v48.5+ DRS 辅助路径进行逐项审计后发现：
核心 OC-MERO 一直正确实现为对每个 post-prefix observation-conditioned row `q[i,l]` 先
`max_l`，再在 roots 上做 outer LCVaR。也就是说，不可区分的 roots 通过 compatibility-weighted
lower-tail `q[i,l]` 被迫共享 compatible option，但**不同的可区分 post-prefix observation class
可以选择不同 recovery option**。

历史 v48.5 的“exact policy contract”却把它解释成“整个候选所有 roots 只能选择一个 globally
shared option”，并把这一更强约束扩散到 best-option/DRS auxiliary loss、teacher-PCD、calibration
和部分 execution diagnostics。这个约束不是论文 Eq. OC-MERO 所要求的 observation consistency。
当 Safe 的 best-option diversity≈1 时影响极小；Near/Contact 的 best-option diversity≈1.4–1.6，
它会系统性制造 false veto。当前 Precision certificate 的 `14/16` Near 和 `29–30/31` Contact
safe-positive deployability false veto 与这一语义错配高度一致。

v48.46 不通过降低 harm budget 或放松 observation consistency 来修复，而是新增显式、可审计的
两种训练监督语义：`global`（历史对照）与 `observation_class`（论文一致）。新的
`best_observation_consistent_option_indices()` / `predicted_observation_consistent_option_success()`
直接从 OC-MERO `q[i,l]` 做 row-wise option selection；class-specific success/best-option losses
只移除对**可区分 classes**的全局绑死，compatible roots 仍被 q 的 observation-kernel lower-tail
共同约束。

### v48.46 scientific 2x2：固定同一评价尺子

为了避免“改变算法同时改变 certificate ground-truth 定义”的混杂，四个 arm 的最终
calibration/certificate/closed-loop **全部固定为论文一致的 `observation_class` execution
semantics**。2x2 的因素只改变训练监督和 staged witness：

- **A:** legacy `global` option-witness/teacher supervision；无 staged witness。
- **B:** `observation_class`-aligned option-witness/teacher supervision；无 staged witness。
- **C:** legacy `global` supervision + sequential `obs -> margin` witness。
- **D/Main:** `observation_class`-aligned supervision + sequential `obs -> margin` witness。

因此 B-A 是“训练语义与 OC-MERO 对齐”的主效应；C-A 是“顺序 witness identifiability”的主效应；
D-B-C+A 是二者 interaction。A/B/C/D 共享同一个 paper-consistent gate protocol、dataset labels、
risk budgets、source checkpoint、dual-ROCT、top-k=5 和 shared continuous rule；comparator 对任何
identity mismatch fail-closed。

顺序 witness 的 obs stage 只允许 `obs_embed_head` 更新，margin stage 只允许 `margin_head` 更新；
`root_logit_head`、shared encoder/root decoder/direct policy/OCAF/ROCT heads 全部冻结。每一 stage
复用 v48.45.6 source-architecture/isolation contract，禁止 hidden partial checkpoint load。

### Pre-registered v48.46 go/no-go

本轮不以“loss 下降”作为成功条件：

- **Near:** 优先要求 certificate safe-positive deployability false veto 从 `14/16` 明显下降，
  screening target `<=10/16`；deployability harmful-vs-safe-positive AUC 向 `>=0.56` 移动；
  verify recall 至少接近 `0.20`，并保持 harmful-selected UCB90 `<=0.25`。
- **Contact:** development safe-positive `pred_adv>=0` 必须从 `1–2/37` 显著增加，screening target
  约 `>=10/37`；certificate recall 向 `0.15–0.20` 移动，harmful-selected UCB90 `<=0.25`，
  false veto 必须显著低于 `29/31`。
- **D/Main:** 必须同时保留 B 的 capture/frontier 改善和 C 的 risk suppression；若 interaction
  再次为负，不吸收 sequential witness package。
- **Safe:** 只有 RC=0 后才进入 paper-level paired closed-loop non-inferiority；不因为 Safe 数据简单
  而添加 Safe-specific strategy。

内部 CCF-A readiness 仍不是 venue 官方阈值：Near verify recall 目标约 `0.25–0.33`、harmful UCB
`<=0.25`、precision LCB `>=0.40`；Contact recall `0.20–0.30`、UCB `<=0.25`，并需要 secondary
collision、post-contact TTC、stable-stop/rejoin 等 closed-loop 改善；Safe 需要严格 nominal utility /
progress / intervention / FRA non-inferiority。当前 v48.45.6 与这些目标仍有明显距离。

### 不重复的历史失败方向

遵守 v48.45 stop rule，本轮明确**不**增加 ROCT width/scale，不放松 harmful budgets，不 densify
threshold grid，不扩大 top-k，不做 positive oversampling，不再堆 generic pairwise/listwise rank
loss，不恢复 generic harm residual/unbounded factor，不重做 v48.42 partial pooling/rank skip，
不恢复 v48.43 POET free alias transport，不 broad fine-tune encoder，也不加入 regime routing。

若 v48.46 仍失败，下一步才进入 recovery-option taxonomy/continuous-parameter coverage、margin teacher
calibration 和 observation-class identifiability 的数据/teacher 审计，而不是再加 downstream head。

### Execution-equivalent speed path

上传 v48.45.6 四个 arm 实际串行 wall time 约：A `59.2 min`、B `97.3 min`、C `71.6 min`、D
`67.6 min`。主要时间仍在 factor/witness training；同一 train/val NPZ 在不同 variant/stage 中重复
materialize 约 90–250 s/次，是剩余可去除的 I/O 开销。

v48.46 的加速不改变样本、epoch、batch、loss、seed、sampler、top-k 或 gate：

1. `OCRAPSampleDataset` 增加跨进程/跨 stage 的 **persistent decoded-tensor cache**。key 只绑定
   dataset manifest/path、tensor geometry 和 feature-construction settings；model head/optimizer/ROCT/
   option-semantic设置不改变 tensor，因此不再错误地制造不同 cache。`fcntl` lock + atomic replace
   保证 A/B 与 Balanced/Precision 并发 warm-up 时不会产生损坏 cache。
2. A/B **同时运行**，分别占 GPU0/GPU1；完成后 C/D 同时运行。每个 ablation 内 Balanced 与
   Precision 默认在该 arm 的同一 GPU 上并发，因为用户已确认单 arm 显存有 headroom。
3. 若单卡同时跑 Balanced+Precision 出现 OOM/吞吐下降，只设置 `V4846_VARIANT_MODE=serial`；
   两个 ablation 仍分别在两张 GPU 上并发，科学协议不变。
4. v48.45.6 已将单 A 从约 2.13 h 降到约 59 min；v48.46 进一步消除跨 stage 重复解压。按上传
   v48.45.6 wall time 粗略上界，A/B + C/D pairing 在无 contention 时可把四臂串行约 296 min
   降到约 169 min，再叠加 persistent cache 收益。真实值必须由 A30 实测，不在代码审计环境中宣称。
5. persistent cache schema v3 优先以 `weights_only + mmap` 读取，多个 arm/variant 可共享 OS page cache，
   避免每个 trainer 再复制一整份 decoded tensors 到 host RAM；旧 PyTorch 自动 fallback。cache 若损坏/截断
   会在 flock 内视为 miss 并原子重建，不再把缓存介质问题升级成实验 RC=30。
6. certificate config 同时显式写入 `training/calibration/evaluation.option_execution_semantics`，避免未来 config
   优先级或默认值改变时把 paper-consistent evaluation 静默退回 legacy global 语义。

### Legacy architecture debt kept out of this 2x2

共享 v48.45 rebuilt source 仍含历史 `DELTA_REGIME_EXPERTS=true`/bucket-conditioned internal geometry。
v48.46 **不新增也不扩大**这一 legacy 路径，并在 factor contract 中声明
`strategy_regime_conditioning=false`。为了保持 2x2 单因素可归因，本轮不同时重建 source 去除该
历史结构。若 v48.46 得到明确正结果，论文最终版前应单独做一次 controlled source rebuild，移除
legacy bucket-conditioned policy internals，并把 Appendix 的 “Regime-conditioned recovery admission”
改写为连续 deployable headroom + observation compatibility + harm envelope；该结构清理必须作为
独立实验，不能和 v48.46 混在一起。

### Engineering / attribution contracts and local validation

- 新增 `test_v48_46_ocswic.py`，验证 global-vs-observation-class 最小反例、class loss gradients、
  persistent cache key/mmap/损坏重建、explicit evaluation semantics、telemetry summary 与 2x2 launcher/factor contract。
- `compare_v48_46_ocswic_2x2.py` 要求四臂 source checkpoint SHA、protocol seal、**完整同一 gate
  protocol**、development/certificate manifest 完全一致；RC 必须为 authoritative `0/20`。
- 四臂最终 evaluation semantics 固定 `observation_class`；training semantics 才是可变因素，避免
  change-of-metric confounding。
- 当前本地完成回归：v48.42–v48.46 + v48.45 engineering/protocol 组合测试 `70/70`；
  v48.37–41 `29/29`；v48.36 OCAF `14 passed / 1 skipped`；v48.36 transfer/terminal/idempotence
  `18/18`。去重后合计 **131 passed / 1 skipped**。
- `python -m compileall -q src tools` PASS；全部 **103/103** shell scripts `bash -n` PASS。
- 本地环境没有用户 A30/WOMD 数据，因此未声称真实 v48.46 GPU/certificate 结果。


## v48.45.6 — SOWR STAGE-ISOLATION + SOURCE-ARCHITECTURE + EXACT-I/O HOTFIX (2026-08-12)

**类别：工程/执行性能修复；SOWR、dual-ROCT、shared rule、risk/gate 算法语义不变。**

### Uploaded v48.45.5 evidence

- Arm A is **not** a pipeline failure. It completed certificate and gate with `pipeline_valid=true`, authoritative `RC=20`; it is a valid negative algorithm result.
- Arms B/C/D(Main) all terminate before SOWR training with normalized `RC=30`, both Balanced and Precision failing in `shared_option_witness_recalibration` with the same constructor exception: `ValueError: ROCT requires component-head physical evidence`.
- The dedicated-calibration protocol introduced in v48.45.5 is valid in all four arms. The shared protocol seal is identical and no test root was read.
- Source checkpoint hashes are valid and identical across arms: Balanced `070218c66e506d66f25a12bf53b4127581992d75481bbb635d7ea658f4cfd352`, Precision `8f7528b76ce4b2424c5f153fe3109844de27a08a976a067b1292baee393f768d`.

### Root cause 1 — downstream ROCT environment leaked into witness-only SOWR

`run_v48_45_sowr_ablation_arm.sh` exports downstream v48.44-D `EVIDENCE_ROCT_BENEFIT=true` and `EVIDENCE_ROCT_DEPLOYABILITY=true`. B/C/D execute the optional SOWR stage before downstream factor adaptation. v48.45.5 called the generic trainer without stage-local overrides, while the generic trainer defaults `EVIDENCE_COMPONENT_HEADS=false`. The witness model therefore received `ROCT=true && component_heads=false` and correctly failed its constructor guard. A has no SOWR stage, so it does not hit this leak.

The repair does **not** remove the ROCT guard and does **not** fabricate reliability for the observed constant `hard_rule`/`harm_proxy` components. During SOWR only, all downstream OCAF/ROCT/component-evidence flags are explicitly disabled. After SOWR, downstream factor adaptation still runs the original dual-ROCT configuration with component heads enabled.

### Root cause 2 — hidden source-architecture drift after fixing root cause 1

The rebuilt v48.45 source uses a fixed architecture (`preference_head=false`, set tournament enabled 48/4/0.05 with replacement, `delta_mode=ordinal_evidence`, delta hidden 48/dropout .02, regime experts/policy features enabled). Generic trainer defaults differ materially. v48.45.5 SOWR did not pin the source architecture, so a simple ROCT-only fix could have produced partial `strict=False` checkpoint loads and silently replaced frozen direct-policy layers.

v48.45.6 therefore pins the exact rebuilt-source architecture in the SOWR subprocess and adds fail-closed pre/post contracts:

- `check_v48_45_sowr_source_architecture.py` validates immutable source metadata and required state prefixes before training.
- `check_v48_45_sowr_stage_isolation.py` verifies architecture metadata preservation, key-set preservation outside the allowed witness prefixes, exact trainable-prefix declaration, no downstream evidence flags in the SOWR checkpoint, and no parameter change outside the intended B/C/D witness heads. Epoch-zero/identity is allowed because “no beneficial update” is a legitimate algorithm outcome.

### Execution-equivalent I/O acceleration

Uploaded valid Arm A took about **7683.7 s = 2.13 h** end-to-end. The initial contracts/index phase was about 3.5 min, factor adaptation about 85 min, and certificate/calibration about 34–38 min. Both A30 jobs show a synchronized epoch-14 stall (`~1188 s` Balanced, `~1203 s` Precision) while ordinary epochs are roughly 140–267 s, consistent with shared compressed-NPZ/storage/CPU data-pipeline contention rather than a single-GPU compute failure.

v48.45.6 adds only execution-equivalent fast paths:

1. NPZ member-selective loading: model/index/calibration readers materialize only fields actually consumed by the computation.
2. Optional `training.cache_samples_in_memory`: each training process decodes the final flat feature + recovery-label CPU tensors once and reuses them for all epochs; raw BEV/map/debug arrays are not cached.
3. v48.45 launcher defaults to `ABLATION_CACHE_SAMPLES_IN_MEMORY=true`, `ABLATION_NUM_WORKERS=3`, `ABLATION_PREFETCH_FACTOR=3`; batch size, epochs, sampler, loss, LR, seed, top-k, ROCT parameters and gate are unchanged.
4. `dataset_materialization_done` logs cache time and byte count so the next real GPU run can quantify the improvement.
5. Certificate/support-index paths also use selective NPZ loading. Setwise candidate scoring semantics are unchanged; no cross-scene batching was introduced.
6. Keep `MAX_PARALLEL_ARMS=1`: every arm already runs Balanced on GPU0 and Precision on GPU1. Arm-level concurrency >1 puts multiple training processes on each GPU and reintroduces I/O/GPU contention.

### Resume and attribution safety

- `prepare_v48_45_6_resume.py` can preserve an existing authoritative RC=0/20 arm only when its protocol seal and current source checkpoint SHA256 match. Pipeline-invalid RC=30 arms are deleted for a clean retry.
- Fast engineering recovery can therefore keep the uploaded valid A and rerun only B/C/D. For paper-quality final 2x2 attribution, rerun all four from the same v48.45.6 checkout (`FORCE_RERUN_VALID_ARMS=1`).
- No Safe/Near/Contact identifier, router, policy, threshold or risk budget was added. `test_roots_read=false` remains a hard contract.

### Local validation

- v48.45.6 new tests: **7/7 passed**.
- v48.44 + v48.45 focused regression: **43/43 passed**.
- v48.40–v48.45 regression: **74/74 passed** (warnings are existing PyTorch nested-tensor warnings).
- v48.36 stage/terminal/OCAF regression: **32 passed / 1 skipped**.
- v48.37–v48.39 regression: **18/18 passed**.
- `python -m compileall -q src tools tests`: PASS.
- **100/100** shell scripts pass `bash -n`; operator command passes `bash -n`.
- repository-wide same-`local` nounset RHS audit: **0 findings**.
- The delivery environment has no user WOMD files or A30 GPUs, so no new real training/certificate result is claimed.

## v48.45.5 — DEDICATED CALIBRATION PROTOCOL BOOTSTRAP HOTFIX (2026-08-12)

**Category: engineering-only. The v48.45 SOWR algorithm, A/B/C/D factors, shared source,
ROCT/top-k/shared-rule settings, harm budgets, certificate semantics and Natural gate are unchanged.**

### Uploaded failure diagnosis

The uploaded v48.45.4 shared source is now valid: `S1_SOURCE_POLICY_STATUS.json` records
`balanced_exit_code=0`, `precision_exit_code=0`, both rebuilt checkpoints pass the
SHA256 source contract, and the source-quality contract is valid.  The uploaded A/B/C/D
SOWR runs all fail before adaptation with the same authoritative engineering state:
`stage=dataset_root_contract`, `raw_exit_code=4`, `pipeline_exit_code=30`,
`certificate_executed=false`, `gate_evaluated=false`.

The dataset contract shows the exact missing roots.  Every arm resolved the legacy
dedicated protocol root
`$OCRAP_ROOT/calibration_v48_14_prism_4814` and expected six derived directories:
`evidence_adapt_{train,dev}_{near_contact,contact}` plus
`certificate_pool_{near_contact,contact}`.  All six canonical-path checks were true but
all six `*_exists` checks were false.  `calibration_safe` existed.

The operator command was the defect: it validated `train_*`/`val_*` for source rebuild
but never invoked the repository's deterministic
`partition_dedicated_calibration_v48_14.py` on the actual
`calibration_near_contact` and `calibration_contact` roots.  The v48.36 controller was
therefore correct to fail closed; the missing protocol bootstrap made all four arms
guaranteed RC=30.  This upload contains no new SOWR algorithm evidence.

### Engineering fixes

1. Added `scripts/prepare_v48_45_protocol.sh`.  Before any arm starts it deterministically
   derives the dedicated protocol from `calibration_near_contact/contact` with the
   preregistered seed `4814`, adaptation-train fraction `0.45`, adaptation-dev fraction
   `0.15`, and certificate fraction `0.40`.  `calibration_safe` remains the safe
   calibration root.  No `test_*` root is read.
2. Added `tools/check_v48_45_protocol_seal.py`, an independent fail-closed seal.  It
   verifies source split IDs are exactly `calibration`, all six derived role manifests
   are non-empty, role labels are exact, every referenced sample exists, Near and
   Contact train/dev/certificate scene sets are mutually disjoint, their union exactly
   recovers each source calibration population, and every scene assignment exactly
   matches the seed/fraction hash rule.  Source and role manifest SHA256 hashes are
   recorded for attribution.
3. Protocol preparation is resumable.  A valid existing protocol is independently
   revalidated and reused without repartitioning.  An invalid/partial protocol is moved
   aside, rebuilt, audited and sealed; failure rolls back to the previous directory.
4. `run_v48_45_sowr_2x2_parallel.sh` now prepares/verifies this one shared protocol
   before launching A/B/C/D.  Direct single-arm invocation also prepares/verifies it,
   so the previous four-way repeated dataset-root failure cannot recur silently.
5. The new operator command explicitly validates all train/val/calibration manifests,
   prepares the shared protocol before deleting failed arm outputs, reuses the already
   successful v48.45 source, and captures launcher RC so any future engineering failure
   prints each arm's `PIPELINE_FAILED.json` instead of being mistaken for algorithm
   evidence.
6. Every arm recomputes a fast independent protocol-seal check at its boundary and records
   `protocol_seal_sha256` in `ATTEMPT_STARTED.json`.  The v48.45 comparator now fails closed
   unless all four arms have the same protocol seal, source checkpoint SHA256 pair, gate
   protocol SHA256 and certificate/development manifest SHA256 identities.  This prevents
   B-A/C-A/D-B-C+A from being reported when shared inputs drift.
7. Arm implementation-version strings are advanced to v48.45.5 for provenance only;
   `OCRAP_ALGORITHM_VERSION`, SOWR factor definitions and all algorithm hyperparameters
   remain unchanged.

### Attribution boundary

Safe/Near/Contact remain dataset/evaluation strata only.  No regime identifier, router,
per-regime policy, per-regime threshold or relaxed risk budget is introduced.  The
strict 2x2 remains A=no SOWR, B=root/margin witness only, C=observation-kernel only,
D=both, with one shared source and one shared scene-disjoint calibration protocol.
Only authoritative arm RC 0/20 results are eligible for B-A, C-A and interaction
`D-B-C+A` attribution.

### Validation

- New protocol/bootstrap focused tests: **4/4 passed**; complete v48.45 focused set:
  **29/29 passed**.
- v48.42-v48.45 regression: **56/56 passed**.
- v48.36 controller/terminal suites: **32 passed / 1 skipped**.
- v48.37-v48.41 regression: **29/29 passed**.
- Combined completed v48.36-v48.45 validation: **117 passed / 1 skipped**.
- `python -m compileall -q src tools tests`: PASS.
- **100/100** shell scripts pass `bash -n`; the v48.45.5 operator command also passes.
- Repository-wide nounset same-`local` self-dependency scan: **0 findings**.
- Synthetic end-to-end arm smoke passes `source_checkpoint_contract`,
  `dataset_root_contract`, `dedicated_protocol_audit`, multigroup loss and OCAF bridge,
  then stops only at the expected CUDA preflight because the validation container has
  no training GPU.  Thus the exact uploaded `dataset_root_contract` failure is removed
  from the real controller path before any algorithm stage.

## v48.45.4 — S1 NOUNSET SOURCE-REBUILD ENGINEERING HOTFIX (2026-08-12)

**Category: engineering-only.  The v48.45 SOWR algorithm, 2x2 factors, source data mix,
ROCT/top-k/shared-rule settings, harm budgets, certificate and gate are unchanged.**

### Uploaded failure diagnosis

The uploaded `ocrap_v48_45_source_rebuild_s7` is not an algorithm result.  S0 completed
successfully (`50114` train samples, `12250` validation samples, `13` epochs, best epoch
`7`, best validation loss `5.2515695061`, and an existing `best.pt` on the run machine),
but the source rebuild terminated at `S1_source_policy_heads`.  The retained status log
is exactly `balanced=1 precision=1`; neither S1 candidate directory nor its
`train_summary.json` was created.  The outer source driver then normalized the failed
S1 stage to RC=30.  No v48.45 SOWR adaptation/certificate/gate result can therefore be
attributed from this upload.

The direct cause is Bash nounset evaluation in `rebuild_v48_45_shared_source.sh`:

```bash
train_source_variant() {
  local variant="$1" gpu="$2" run="$SOURCE_OUT/candidates/$variant"
  ...
}
```

The script uses `set -u`.  Bash expands all RHS expressions before the `local` builtin
establishes those local variables, so `$variant` in the third assignment is unbound.
Both background S1 functions abort before `mkdir -p "$run/logs"`, producing the observed
`balanced=1 precision=1` signature.  A minimal shell reproduction returns the same two
exit codes.

### Engineering fixes

1. `train_source_variant` now declares `variant gpu run` first and assigns them on
   separate commands; S1 additionally writes `S1_SOURCE_POLICY_STATUS.json` with both
   raw exits before the outer RC=30 mapping.
2. Resuming an unsealed source removes only stale failure/status markers.  A complete S0
   (`best.pt` + `TRAINING_COMPLETE.json`) is preserved and reused; it is not retrained.
3. The new operator commands no longer delete the entire incomplete `SOURCE_RUN`, so the
   already-completed S0 from the uploaded attempt can be reused on the original machine.
4. `run_v48_45_sowr_2x2_parallel.sh` now defaults to `MAX_PARALLEL_ARMS=1`.  Each arm
   already runs Balanced on GPU0 and Precision on GPU1 concurrently, so this uses both
   GPUs while avoiding 8 concurrent train processes on two GPUs.
5. The same nounset-local dependency class was removed from the latent v48.36 ablation
   task launcher and both v48.34 video launchers.  A repository-wide shell test now
   scans every script for this exact unsafe `local a=... b=...$a` pattern.
6. The new operator command runs source hash/quality contracts before A/B/C/D and builds
   `ocrap_v48_45_sowr_2x2_comparison.json` only after all four arms are authoritative
   RC=0/20 results.

### Attribution boundary

No new Safe/Near/Contact identifier, router, regime-specific threshold or regime-specific
policy is introduced.  Because the uploaded run stopped before S1 training, no further
algorithm change is justified by this result.  The next algorithmic decision remains the
pre-registered SOWR 2x2: A=no witness recalibration, B=root/margin witness only,
C=observation kernel only, D=both.  Only after these four arms complete with RC 0/20 may
we decide whether SOWR is effective.

### Validation

- v48.42-v48.45 focused regression: **52 passed**.
- v48.45 focused regression: **25 passed**.
- `python -m compileall -q src tools tests`: PASS.
- **99/99** scripts pass `bash -n`; v48.45.4 operator command also passes `bash -n`.
- Repository-wide nounset same-`local` self-dependency scan: **0 findings**.
- A larger v48.36-v48.41 regression batch was started and reached its environment time
  limit without reporting a test failure; it is not counted as a completed validation.

## v48.45.3 — SOURCE-REBUILD EMPTY-OVERRIDE ENGINEERING HOTFIX (2026-08-11)

**Category: engineering-only; v48.45 SOWR algorithm and v48.45.2 source-rebuild attribution protocol are unchanged.**

### Uploaded v48.45.2 failure diagnosis

The uploaded `ocrap_v48_45_source_rebuild_s7` did not complete S0.  Its shared-backbone
log scans the pooled Safe/Near/Contact train and validation roots successfully, then
crashes before epoch 1 while constructing `OCRAPModel`:

```text
ValueError: could not convert string to float: 'None'
... direct_recovery_evidence_component_reliability
```

The shell trainer intentionally passes optional string settings as explicit empty CLI
overrides, e.g. `--set model.direct_recovery_evidence_component_reliability=` and
`--set training.init_checkpoint=` in scratch mode.  The generic override parser used
`yaml.safe_load("")`, which returns Python `None`.  `train.py` then converted the
reliability value with `str(None)`, producing the literal `"None"`; the model attempted
`float("None")` and S0 terminated.  Since S0 produced neither `best.pt` nor
`TRAINING_COMPLETE.json`, S1 never produced Balanced/Precision source checkpoints.
The uploaded A/B/C runs therefore all fail later at `source_checkpoint_contract` with
RC=30.  Adaptation, certificate and gate were never executed, so those runs are not
algorithm-attribution evidence.

### Engineering fixes

1. CLI `--set key=` now preserves the explicit empty string.  YAML null remains
   available explicitly via `key=null` or `key=~`.
2. Train-time model and ordinal-evidence reliability plumbing normalizes optional
   `None` to the original unspecified/default semantics instead of the string
   `"None"`.
3. Model, loss and inference/checkpoint paths defensively accept `None`, empty string,
   and legacy textual null spellings as an unspecified reliability vector, retaining
   the pre-existing default of all-one component reliability.  Nonempty numeric CSVs
   are unchanged.
4. `rebuild_v48_45_shared_source.sh` now writes `SOURCE_REBUILD_FAILED.json` with the
   exact stage (`preflight`, teacher-index build, S0, S1, sealing, or final contracts)
   on any nonzero source-rebuild exit.  This prevents a primary S0 error from being
   visible only later as missing source checkpoints.
5. The v48.45.3 operator commands explicitly delete only an **incomplete** source
   (never a sealed `SOURCE_REBUILD_COMPLETE.json`), run an empty-override semantic
   precheck, fail immediately if source rebuild is nonzero or the manifest is absent,
   and only then launch A/B/C/D.

### Attribution boundary

No loss weight, S0/S1 dataset mix, architecture, SOWR switch, ROCT parameter, top-k,
shared Natural rule, risk budget, certificate, gate, or regime-specific policy is
changed.  After rebuilding one fresh sealed source, A/B/C/D remain attributable within
that source identity.  The failed uploaded v48.45.2 A/B/C runs must be discarded.

### Validation

- Focused v48.13/v48.36-v48.45 regression: **117 passed / 3 skipped**.
- v48.45.3 exact S0 shell-argument capture confirms scratch `init_checkpoint`,
  freeze/trainable prefixes, model reliability and ordinal reliability are all parsed
  as `""`, not `None`.
- `python -m compileall -q src tools tests`: PASS.
- **99/99** shell scripts pass `bash -n`; backslash-continuation/comment hazard scan: 0.
- Dynamic source-failure simulation writes `SOURCE_REBUILD_FAILED.json` with the
  correct `preflight` stage and preserves the original nonzero exit.

## v48.45.1 — SOWR ENGINEERING HOTFIX / SOURCE-RUN RESOLUTION (2026-08-10)

This revision changes **no algorithmic factor, loss target, shared rule, gate, top-k,
ROCT scale, regime handling, or risk budget**. It fixes the execution layer only.

The failed v48.45 A/B/C runs are authoritative engineering failures (`RC=30`,
`pipeline_valid=false`, `failure_stage=adaptation`, certificate not executed). Their
result packages contain no candidate directories or `adapt_balanced/precision.log`,
which means both variants returned before entering adaptation. The run metadata records
`source_run=runs/ocrap_v48_13_terra_proxy_4801` while the v48.45 command changes cwd to
the versioned checkout. In the successful v48.44 results, the actual source checkpoint
was `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_13_terra_proxy_4801/.../best.pt`. The
relative default therefore resolved against the wrong repository in v48.45. This is the
root cause; there is no OOM evidence.

Engineering fixes:

- v48.45 arm launcher resolves the immutable v48.13 source run from persistent
  `BASE_OUT` first, while preserving an explicit `SOURCE_RUN` override; the resolved
  path is made absolute before entering the dedicated controller.
- the dedicated controller now performs a source-checkpoint contract **before** dataset
  index construction or GPU adaptation, checking both Balanced and Precision source
  checkpoints and recording paths, size and SHA256 in `SOURCE_CHECKPOINT_CONTRACT.json`;
  a missing source now fails as `source_checkpoint_contract`, not as the ambiguous
  `both variants failed`.
- the per-variant fallback path now always writes an in-run adaptation log,
  `VARIANT_STAGE_FAILED.json`, `FAILURE_SIGNATURE_*.json`, and
  `ADAPTATION_FAILED_*.json` if a source/factor-cache prerequisite disappears after the
  global preflight.
- SOWR controls are explicitly passed across the dedicated-controller -> variant-process
  boundary instead of relying only on inherited export state.
- the internal SOWR witness stage now sets `SKIP_POST_TRAIN_CALIBRATION=1`; generic
  bucket calibration is not run inside witness recalibration, while the downstream authoritative
  v48.36 calibration/certificate path remains unchanged.
- the v48.45 2x2 launcher now treats RC=20 as a valid Natural-gate result, not an
  engineering crash; only non-{0,20} arm exits make the launcher fail. Launcher logs and
  exit codes are stored inside each run directory so result ZIPs retain the primary
  failure evidence.

Validation after the hotfix: compileall PASS; 98/98 shell scripts `bash -n` PASS; focused
v48.36-v48.45 regression 102 passed / 1 skipped; dedicated missing-source simulation
correctly terminates before dataset/adaptation; synthetic four-arm RC=20 launch returns
success and classifies all four arms as valid Natural-gate failures; synthetic RC=30
launch is correctly classified as engineering failure.

The existing v48.45 SOWR run commands remain valid when the repaired code is extracted
under the same `/home/senzeyu2/code/OC-RAP-v48.45-SOWR` path. Do not delete the immutable
source run `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_13_terra_proxy_4801` when cleaning
old v48.45 outputs.

## v48.45 — SOWR / SHARED-OPTION WITNESS RECALIBRATION (2026-08-10)

### v48.44 ROCT final gate attribution

All uploaded v48.44 A/B/C/D runs are authoritative Natural-gate algorithm results:
`authoritative_exit_code=20`, `pipeline_valid=true`, certificate executed, gate evaluated,
`gate_passed_false=true`, and no pipeline/calibration-failure marker. They are therefore
valid for algorithm attribution and must not be treated as engineering failures.

**B (deployability-side ROCT) is rejected as a Near frontier rotation.** Relative to A,
Near certificate deployability safe-positive false veto changes only 16/18 -> 15/18,
harmful-vs-safe-positive AUC 0.416 -> 0.418, while harmful false-safe rises
0.162 -> 0.301. Certificate recall remains 0.111. This misses the pre-registered
strong signal (roughly <=12/18 false veto and AUC >=0.56 without contamination).
Do not absorb B as evidence that the Near deployability bottleneck is solved.

**C (benefit-side ROCT) is rejected.** Contact development still has exactly **0/37**
safe-positive proposal rows with `pred_adv >= 0`; certificate recall falls from 0.10 to
0.05; candidate safe-positive AUC 0.544 -> 0.523 and proposal safe-positive AUC
0.555 -> 0.543. It therefore does not break the physical-sign collapse and must not
be absorbed.

**D/Main (dual ROCT) is not accepted as a successful shared-rule solution.** Precision
development gets one Near safe-positive but zero Contact safe-positive; Contact remains
0/37 on `pred_adv >= 0`. Balanced development does select one Near and two Contact
safe-positives under one shared rule, but harmful-selected UCB90 is 0.282 (Near) and
0.387 (Contact), both above the 0.22 fit budget, while precision LCBs are far below 0.50.
The two-sided positive-capture condition is therefore not achieved without harmful
contamination.

D nevertheless provides one useful *diagnostic* signal. It reduces certificate
deployability harmful false-safe to 0.076 (Near) / 0.068 (Contact) and improves Contact
deployability AUC to 0.651, while its Near development deployability AUC reaches 0.764.
But Near certificate AUC collapses to 0.487 and false-veto returns from 9/17 development
to 16/18 certificate. The useful structure is not stable across splits.

### Bottleneck re-diagnosis

The top-5 proposal oracle remains feasible in both Near and Contact, so candidate support
is not the first bottleneck. More importantly, the current direct-evidence adaptation
freezes the paper-matched `root_logit_head`, `margin_head`, and `obs_embed_head`; only
OCAF/ROCT-side evidence corrections are adapted. Yet source training already contains
teacher supervision for root assignment, signed recovery margins, observation
compatibility, deployability/oracle scores, option-resolved shared-recovery `q`, shared
option admission/success, and best shared option.

The D split gap shows why adding another residual or increasing ROCT width/scale is the
wrong next move: development can fit a useful deployability ordering, but the frozen
recovery witness does not carry that ordering stably to certificate. This is consistent
with stale/miscalibrated recovery-option margins, root probabilities, or observation
compatibility rather than a lack of downstream ranking capacity.

### v48.45 algorithm: SOWR

**SOWR = Shared-Option Witness Recalibration.** Before the existing factor/OCAF/ROCT
adaptation, insert a short, low-learning-rate, regime-agnostic witness stage that uses
only the existing training/validation teacher contract. It does **not** add a
Safe/Near/Contact identifier, regime router, regime-specific threshold, or regime-specific
policy.

The shared encoder and root decoder stay frozen. SOWR only permits the following
paper-matched semantic heads to move:

- margin-witness factor: `root_logit_head,margin_head`;
- observation factor: `obs_embed_head`.

The stage uses full OC-MERO forward semantics plus explicit root-assignment and
observation-equivalence losses. The explicit losses are required because lower-tail /
top-m OC-MERO aggregation alone does not guarantee dense gradients to root/observation
heads. Default SOWR settings are 8 epochs, patience 3, LR 5e-5, batch 72. It uses
train/validation only; held-out test roots remain unread. After SOWR, all witness heads
are frozen again and the **same v48.44-D dual ROCT, top-k=5, shared continuous rule and
risk budgets** are run unchanged.

### v48.45 strict 2x2

- **A:** v48.44-D dual-ROCT reference, no witness recalibration.
- **B:** A + root-probability/recovery-margin witness recalibration.
- **C:** A + observation-kernel recalibration.
- **D/Main:** A + both witness factors.

This 2x2 answers whether the frozen recovery witness is the current bottleneck without
confounding the experiment with another selector, larger ROCT scale, new rank loss, or
regime routing.

### Pre-registered go/no-go interpretation

**B / Near:** require a material certificate deployability frontier rotation, not only
lower training loss. Strong signal: safe-positive false veto <=12/18, deployability
AUC >=0.56, and no increase in harmful false-safe / harmful-selected UCB. The witness
stage should also reduce validation margin/option-q/best-option loss from its initial
checkpoint.

**C / Contact:** first requirement remains breaking the physical-sign collapse:
Contact development safe-positive `pred_adv >= 0` must become non-zero (strong signal
about >=25%), with candidate/proposal safe-positive AUC improving by roughly 0.05 and
recall moving toward 0.20 without a risk rebound. Lower observation BCE alone is not a
go decision.

**D/Main:** the same shared rule must have `positive_selected > 0` in both Near and
Contact development, retain B/C structural signals, and satisfy the existing harmful
budgets rather than relaxing them. A key generalization signal is that the large
Near development->certificate deployability AUC/FV gap should shrink materially.

If SOWR fails, **do not** increase ROCT scale/width, loosen harmful budgets, densify the
threshold grid, expand top-k, oversample positives, add generic pairwise/listwise rank
stacks, re-add generic harm residuals, or introduce regime-specific routing. The next
algorithmic audit should move upstream to recovery-option coverage/taxonomy, continuous
option parameter coverage, margin-teacher calibration and option-resolved witness
identifiability.

### Historical modifications explicitly not repeated

v48.45 intentionally avoids previously unsuccessful families: threshold densification,
top-k expansion, positive oversampling, generic rank stacking, generic harm residuals,
unbounded residuals/factors, full factorization, v48.42 partial-pooling/rank-skip,
v48.43 POET free alias transport, ROCT width/scale increases, broad encoder fine-tuning,
and regime-conditioned policy/threshold routing. The retained downstream stack remains
candidate-relative physical OCAF + dual-task bridge + bounded component veto/support
reliability + deterministic joint reserve + shared rule.

### Engineering changes and validation

- `scripts/train_ocrap_v48_trac_sr.sh`: source loss weights and direct-only fast path are
  now environment-overridable while legacy defaults remain exactly unchanged.
- New `scripts/adapt_ocrap_v48_45_sowr_stage.sh` implements the head-only witness stage.
- `scripts/adapt_ocrap_v48_36_ocaf_variant.sh` optionally inserts SOWR and binds the
  resulting checkpoint hash into the factor-cache contract.
- New A/B/C/D runners and `tools/compare_v48_45_sowr_2x2.py` report SOWR initial/best
  validation diagnostics plus development/certificate metrics without test-root reads.
- Focused regression audit after implementation: **95 passed / 1 skipped** across the
  retained v48.36-v48.45 contracts; v48.44+v48.45 targeted matrix **14 passed**.
- `compileall` PASS; **98/98** shell scripts pass `bash -n`; new v48.45 runtime scripts
  pass a static no-regime-routing and no-test-boundary check.

## v48.44 — ROCT / RECOVERY-OPTION COMPATIBILITY TRANSPORT (2026-08-10)

### v48.43 POET final 2x2 attribution

The uploaded v48.43 A/B/C/D runs are all authoritative Natural-gate algorithm results
(`authoritative_exit_code=20`, `pipeline_valid=true`).  They therefore support causal
algorithm attribution rather than engineering-failure diagnosis.

**B (harm-side POET) is rejected.**  Near deployability safe-positive false veto is
unchanged at 16/18 relative to A, while harmful-vs-safe-positive deployability AUC
falls from 0.416 to 0.409.  Near certificate harmful-selected UCB90 worsens from
0.546 to 0.640.  B therefore does not rotate the Near deployability frontier; it
mostly changes score level/cleanliness trade-offs and is not retained.

**C (benefit-side POET) is rejected.**  Contact candidate safe-positive AUC changes
only from 0.544 to 0.556, proposal safe-positive AUC falls from 0.555 to 0.526, and
certificate positive recall remains 0.10.  The development Contact safe-positive
rows still have 0/37 with `pred_adv >= 0`.  C does not convert post-prefix alias
geometry into physical benefit capture and is not retained.

**D/Main (dual POET) is rejected as an implementation, but supplies a structural
clue.**  It improves Contact candidate safe-positive AUC to 0.578 and deployability
safe-positive false veto from 26/31 to 17/31; DRS harmful-vs-safe-positive AUC rises
from 0.554 to 0.706.  However Near deployability false veto worsens to 17/18,
Contact deployability harmful false-safe rises from 0.148 to 0.255, Contact
certificate recall falls to zero, and the fitted shared development rule selects
zero candidates in both Near and Contact.  This is evidence that candidate-specific
post-prefix structure carries useful signal, but free dual context transport creates
incompatible scales/negative transfer across evidence coordinates.

A decisive development diagnostic is common to all A/B/C/D arms: Contact has 37
safe-positive proposal rows and **none has `pred_adv >= 0`**.  Since the shared rule's
semantic score domain starts at zero, no threshold-grid densification can recover
those positives.  The current bottleneck is therefore sign/structural identifiability,
not grid resolution.

### Bottleneck re-diagnosis

POET measures whether latent roots remain observation-aliased, but the paper's
recoverability definition is stricter: observation-equivalent roots must admit a
**shared compatible recovery option**.  Alias mass alone cannot distinguish benign
ambiguity (roots share a recovery maneuver) from harmful ambiguity (roots require
incompatible maneuvers).  Moreover v48.43 injected the same structural vector into
whole benefit/harm OCAF contexts, permitting one useful Contact coordinate to rotate
unrelated DRS/gap coordinates and damage the shared rule.

v48.44 therefore changes the structural statistic and the injection locality rather
than adding generic capacity, another ranking loss, more proposal candidates, or a
regime-specific policy.

### Algorithm: Recovery-Option Compatibility Transport (ROCT)

For every candidate prefix, ROCT reuses the frozen latent-root decoder, predicted
post-prefix observation kernel, and frozen recovery-option margin decoder.  It computes
the same OC-MERO geometry already used by the paper and forms a bounded four-coordinate
signature:

1. normalized deployable recoverability `0.5*(tanh(R_dep)+1)`;
2. bounded oracle-to-deployable gap `tanh(relu(R_orc-R_dep))`;
3. observation-weighted shared-option conflict pressure: aliased root pairs receive
   high pressure only when no recovery option has high common support for both roots;
4. root-probability mass that has a feasible observation-consistent recovery option.

Only candidate-minus-nominal evidence is learned.  The structural teacher is detached,
and both ROCT projections are bias-free and zero-initialized, preserving the exact A
forward pass at initialization and exact zero correction on nominal candidates.
Corrections are bounded with `tanh`; the v48.44 experiment uses a shared logit bound of
3.0 (at `tau_b=0.05`, at most 0.15 physical benefit-margin correction), avoiding the
historically failed unbounded-residual path.

ROCT is deliberately **semantically local**:

- benefit-side ROCT corrects only the unified benefit logit;
- safety-side ROCT corrects only component index 1 (deployability), which is the
  paper-matched observation-consistent recoverability coordinate;
- it does not rotate DRS, gap, hard-rule, harm-proxy, the shared OCAF bridge, proposal
  generator, or deployment thresholds.

No Safe/Near/Contact identifier, routing branch, per-regime threshold, per-regime loss,
or state machine is introduced.  Regimes remain evaluation slices only.

### v48.44 2x2 ablation

- **A:** v48.43-A retained shared dual-OCAF reference; no POET and no ROCT.
- **B:** A + deployability-side ROCT only.  Primary causal readout: Near deployability
  false-veto/AUC/false-safe, with Contact reported as a cross-regime consistency check.
- **C:** A + benefit-side ROCT only.  Primary readout: Contact safe-positive benefit
  AUC plus the development `pred_adv >= 0` sign-capture rate.
- **D/Main:** A + both semantically local ROCT corrections.  Tests whether one shared
  development rule can select safe-positive examples in both Near and Contact while
  preserving harmful-selection budgets.

Every arm force-disables v48.43 POET, v48.42 partial pooling/rank skip, unbounded
benefit/harm factors, full component factorization, and regime routing.  Proposal
top-k remains 5 and the shared development rule is unchanged.

### Pre-registered v48.44 go/no-go

- **B vs A:** Near deployability false veto must materially decrease (strong signal:
  <=12/18), harmful-vs-safe-positive AUC must increase (strong signal: >=0.56), and
  harmful false-safe / selected-UCB must not materially worsen.
- **C vs A:** Contact development safe-positive `pred_adv >= 0` must move from 0/37 to
  a non-trivial fraction (strong signal: >=25%), Contact candidate/proposal
  safe-positive or positive AUC should improve by about >=0.05, and certificate recall
  should move toward >=0.20 without worse harmful UCB.
- **D:** Near and Contact development must both have `positive_selected > 0` under the
  same shared rule.  Contact sign capture and B's Near deployability gain must both be
  retained.  A lower constraint deficit with zero positives is explicitly not a go.
- If B/C/D fail these structural criteria, do **not** increase ROCT width/scale or
  loosen shared-rule harm budgets.  The next diagnosis must target calibration of the
  frozen recovery-option/observation teacher or the definition/coverage of the
  recovery-option set.

### Engineering changes

- Added ROCT flags/hyperparameters to training, checkpoint, inference materialization,
  factor-cache identity, architecture records, model/training contracts, and
  stage-transfer approved prefixes.
- Direct-only and full forward paths both consume root/option validity masks when
  constructing ROCT evidence.
- Added v48.44 tests for exact zero-init identity, bounded/candidate-relative signature,
  nominal-zero correction, component-local deployability injection, benefit/harm
  separation, detached structural teacher, script cleanliness, and contract binding.
- Added `run_v48_44_roct_ablation_arm.sh`, `run_v48_44_roct_2x2_parallel.sh`,
  `run_v48_44_roct_dedicated.sh`, and a development/certificate-only v48.44 comparator
  with explicit Contact/Near sign-geometry diagnostics.

### Non-repetition list extended by v48.43

Do not repeat or simply amplify: POET alias-only free dual-context transport, POET
width/scale increases, threshold-grid densification, top-k expansion, positive
oversampling, generic rank-loss stacking, generic harm residuals, full component
factorization, unbounded factors, or regime-conditioned routing/threshold/policy.

Retain: candidate-relative physical OCAF, dual task bridge, bounded HAF/component veto,
support reliability, deterministic joint reserve, proposal top-k=5, and one shared
continuous development/deployment rule.

## v48.43 — POET / POST-PREFIX OBSERVATION-EQUIVALENCE TRANSPORT (2026-08-09)

### v48.42 final valid 2x2 attribution (supersedes the earlier RC30-only interpretation)

The newly uploaded v48.42/48.42.1 A/B/C/D results are all pipeline-valid algorithm
results: `authoritative_exit_code=20`, `pipeline_valid=true`,
`certificate_executed=true`, and `gate_evaluated=true`.  Consequently the earlier
v48.41-era statement that the rank-skip arms could not be evaluated no longer applies
to this experiment set.

All four Precision arms remain blocked at `development_rule_fit`, while the proposal
oracle remains feasible (Near top-5: 9 safe-positive groups; Contact: 20).  The failure
is therefore downstream of candidate generation.

**B (hierarchical partial-pooling harm residual) is rejected as a main-algorithm
component.**  In Near certificate, deployability safe-positive false veto changes only
from 16/18 to 15/18 while harmful-vs-safe-positive deployability AUC degrades from
about 0.482 to 0.420 and harmful false-safe fraction rises from about 0.106 to 0.184.
In Contact, B reduces deployability false veto from 27/31 to 21/31, but harmful
false-safe fraction rises and certificate harmful-selected UCB90 worsens from about
0.484 to 0.563.  This is primarily a score/calibration shift, not improved conditional
frontier discrimination.

**C (bounded rank-benefit skip) is rejected.**  Contact candidate safe-positive AUC
changes from about 0.582 to 0.580; proposal positive/safe-positive AUC does not improve;
certificate positive recall remains 0.10.  The strong frozen rank signal previously
observed is not converted into deployable physical benefit evidence by this skip.
Further ranking-objective/skip stacking is therefore stopped.

**D/Main (partial pooling + rank skip) is also rejected.**  It reaches Contact
certificate recall 0.15 versus A's 0.10, but harmful-selected UCB90 worsens to about
0.503 and Contact harmful false-safe rises sharply.  The nearest shared development
rule has fewer formal failures (4 versus 6) and lower aggregate deficit, yet selects
zero safe-positive examples in both Near and Contact development strata.  It has not
entered a statistically meaningful feasible window.

These results activate the pre-registered stop rules from v48.42: do not add more
generic harm residual capacity after B fails the Near deployability frontier, and do
not stack more ranking objectives after a valid C run fails to improve Contact benefit
capture.

### Bottleneck re-diagnosis

The retained v48.40/v48.42-A architecture already sends candidate-relative executable
physical action features to dual benefit/harm OCAF branches, but its observation
context is a broadcast of the nominal row's current observation features.  During the
`direct_only` evidence-adaptation path it does not explicitly expose the candidate's
predicted **post-prefix observation-equivalence geometry**, despite observation
consistency after the candidate prefix being the central recoverability object in the
paper.

v48.43 therefore treats the next problem as **structural identifiability**, not
additional classifier capacity.  The same learned latent-root/observation model is
used for every sample and every regime; no Safe/Near/Contact identifier, branch,
threshold, router, or loss is introduced.

### Algorithm: Post-prefix Observation-Equivalence Transport (POET)

For each candidate prefix `a`, the frozen latent-root decoder and post-prefix
observation embedding predict a compact four-coordinate structural signature:

1. normalized latent-root entropy;
2. probability-weighted off-diagonal observation-alias mass;
3. probability-weighted peak alias pressure;
4. maximum latent-root probability.

Let this bounded signature be `psi(a)` and let `a0` be the nominal candidate in the
same proposal group.  POET uses only the candidate-relative structural evidence

`delta_psi(a) = psi(a) - psi(a0)`.

Two optional bias-free linear projections inject this signal into the already
validated dual-OCAF task contexts:

`z_b' = z_b + s W_b delta_psi`,
`z_h' = z_h + s W_h delta_psi`.

`W_b` and `W_h` are zero-initialized.  Therefore enabling POET is exactly identical to
the shared dual-OCAF reference at initialization; the nominal candidate has exactly
zero transport; and the frozen root/observation teacher cannot be rotated by sparse
evidence gradients.  Only the tiny transport projection is learned.  The mechanism is
continuous, observation/action based, and regime agnostic.

### v48.43 2x2 ablation

- **A:** retained shared dual-OCAF reference; no POET.
- **B:** A + harm-side POET only.  Tests whether candidate-specific post-prefix
  observability improves Near deployability discrimination rather than merely shifting
  its calibration.
- **C:** A + benefit-side POET only.  Tests whether the missing Contact benefit signal
  is recoverable from candidate-specific post-prefix observability without another
  rank objective.
- **D/Main:** A + benefit and harm POET.  Tests complementarity and the shared-rule
  feasibility window.

Every v48.43 arm force-disables v48.42 partial pooling and rank-benefit skip, as well as
historical unbounded factors, factorized harm interaction, one-sided tail losses, and
regime-conditioned routing.  Proposal top-k remains 5 and the same shared Natural rule
is fitted.

### Go/no-go interpretation

The v48.43 mechanism is considered supported only if its causal readout improves
conditional discrimination, not merely score level:

- B should materially reduce Near deployability safe-positive false veto **and** raise
  harmful-vs-safe-positive deployability AUC without a comparable rise in harmful
  false-safe/selected UCB.
- C should materially improve Contact safe-positive/positive AUC and certificate
  positive capture without worsening harm control.
- D should retain B's Near harm cleanliness and C's Contact benefit gain, and the
  shared development rule must begin selecting safe-positive examples in both Near and
  Contact.  A lower numerical constraint deficit with zero positives is not sufficient.
- If POET fails these tests, do not increase its width/scale as the next reaction;
  inspect recovery-option conflict/teacher observability calibration instead.

### Engineering contract changes

- POET flags/scale are checkpoint-, inference-, training-, architecture-, factor-cache-
  and contract-bound.
- Stage-transfer approved-prefix logic now recognizes both POET projections, including
  non-reserve future paths.
- Runtime preflight checks signature shape/finiteness, nominal exact-zero transport,
  and full forward/backward finiteness.
- Added a v48.43 2x2 comparator that reads development/certificate artifacts only and
  explicitly sets `test_roots_read=false`.
- Two-GPU launcher supports four concurrent arms; `MAX_PARALLEL_ARMS=2` provides two
  waves on memory-constrained GPUs without changing experiment semantics.

### Non-repetition list extended by the valid v48.42 results

Do not repeat: threshold-grid densification, top-k expansion, aggressive positive
oversampling, hardest-negative population distortion, generic pairwise/listwise loss
stacking, full identity-stage factor updates, learned admission residual, v48.38
one-sided tail loss, v48.39 unbounded benefit/harm, v48.40 `frontier_tanh`, v48.41 full
component factorization, **v48.42 partial-pooling harm residual**, **v48.42 bounded
rank-benefit skip**, or any regime-conditioned routing/threshold/policy.

Retain: OCAF, factor preservation, bounded HAF semantics, dual task OCAF, component
veto, support reliability, aligned deterministic joint reserve, one shared continuous
deployment rule, and proposal top-k=5.

## v48.42.1 — HPFR METRIC-PROVENANCE HOTFIX (engineering-only, 2026-08-08)

### Scope

This is an **engineering-only** repair of the uploaded v48.42 HPFR main run. The algorithm remains `v48.42-HPFR`: no model tensor computation, loss, optimizer setting, dataset, candidate set, factor-cache semantic setting, Natural gate, threshold, Safe/Near/Contact treatment, or shared continuous deployment rule is changed. `src/` and `configs/` are byte-identical to v48.42.

### Uploaded RC=30 root cause

- Balanced and Precision adaptation both completed with RC=0 and valid training/model/stage-transfer/cache contracts.
- The controller entered certificate execution, but `check_v48_36_metric_calibration_contract.py` rejected the reserve-only final stage before policy-certificate verification.
- `materialize_v48_38_reserve_stage.py` had recorded factor provenance as repo-relative strings such as `runs/<run>/candidates/<variant>/factor_stage/model_v48_trac_sr/train_summary.json`.
- The checker incorrectly treated every relative provenance path as relative to the final model directory, producing a duplicated path of the form `.../model_v48_trac_sr/runs/<run>/.../factor_stage/...`.
- Both certificate workers therefore exited RC=1 with `metric source train summary missing`; the certificate controller normalized the artifact failure to RC=30. The Natural gate was never evaluated.

### Engineering changes

1. Metric-source provenance resolution is SHA256-pinned and anchor-aware. It accepts absolute paths, controller-CWD/repo-relative historical paths, genuine stage-parent-relative paths, and a relocation-safe `factor_stage/...` suffix. A candidate is accepted only when its bytes match the already-recorded SHA256; mismatches still fail closed.
2. New zero-update materialization writes canonical absolute source paths and explicit candidate-relative provenance fields (`provenance_path_schema_version=2`) so future runs do not depend on an implicit relative-path anchor.
3. Certificate ordering now executes adaptation-dev proposal extraction, pooled shared-rule fitting, and the metric-calibration provenance contract **before reading certificate samples**. This changes no score or gate semantics but prevents an engineering provenance error from consuming certificate I/O or leaving misleading partial verification artifacts.
4. Added `run_v48_42_1_hpfr_from_exact_factor_cache.sh`. It requires a new OUTPUTDIR and reuses old Balanced/Precision factor stages only through the existing exact factor-cache SHA/semantic contract; it never silently reuses a checkpoint. This avoids repeating the 20-epoch factor training after the engineering-only failure.
5. The implementation identifier is `v48.42.1-HPFR-METRIC-PROVENANCE-HOTFIX`; the algorithm identifier remains exactly `v48.42-HPFR`.

### Interpretation

The uploaded v48.42 RC=30 is **not an algorithm result**. Development diagnostics produced before the failed provenance contract must not be used as a Natural-gate outcome. The next valid scientific result is the first pipeline-valid RC=0 or RC=20 obtained from the byte-identical v48.42 HPFR factor checkpoint through this repaired certificate path.

# Algorithm Change Log

## v48.42 — HPFR / HIERARCHICAL PARTIAL-POOLING FRONTIER RESERVE (2026-08-08)

### v48.41 result audit and corrected attribution

The uploaded v48.41 result set is **not** four comparable RC=20 runs.  A and B are
authoritative, pipeline-valid `RC=20` Natural-gate rejections.  C and D/main are
`RC=30`, `pipeline_valid=false`, `failure_stage=adaptation`, with no certificate or
Natural-gate decision.  Their factor training completed, but the post-training
prefix verifier rejected the valid scalar parameter
`direct_evidence_rank_benefit_log_gain`: the state-dict key is exactly that string,
while the verifier accepted only dotted module descendants (`prefix + "."`).  The
stage-transfer checker used the same module-only assumption.  v48.42 changes both
contracts to exact-or-dotted matching and explicitly registers the scalar rank-gain
parameter.  C/D therefore cannot be used to claim that rank-benefit skip is effective
or ineffective; it is re-tested as a preregistered arm after the engineering repair.

A/B provide valid evidence about the v48.41 factorized-harm mechanism.  Full
component factorisation is **not retained**.  Precision Near certificate
harmful-vs-safe-positive conditional harm AUC falls from about 0.482 (A) to 0.447
(B); Contact falls from about 0.694 to 0.641.  B also drives the Precision selector
toward near-total abstention.  Component diagnostics explain the mixed signal:
DRS improves with specialization, but deployability remains the dominant false veto
and degrades in rare-frontier discrimination.  For Near certificate safe-positive
candidates, deployability is above the 0.5 harm frontier for 16/18 candidates in
both A and B, while B reduces DRS false-veto frequency.  Contact shows the same
pattern: deployability dominates most safe-positive vetoes.  Thus the strategic
priority—rare-frontier harm discrimination—remains correct, but removing all sharing
between DRS/deployability/gap is the wrong implementation.

Proposal oracle support remains feasible, so candidate generation is not the primary
bottleneck.  The valid A/B failures still occur downstream of proposal support at the
continuous benefit/harm evidence -> noncompensatory reserve -> one shared rule path.
Near benefit ordering remains useful while safety evidence rejects good recoveries.
Contact still has a dual bottleneck: benefit capture is weak and harmful false-safe
control remains insufficient.  The frozen Contact `rank_adv` signal remains much
stronger than the learned opportunity signal (about 0.73 AUC on the audited
certificate rows), so the rank-benefit hypothesis remains worth a clean test after
the checker repair.

### Algorithm change 1: shared-base, detached component frontier residuals

HPFR returns to the empirically stronger **shared harm OCAF + shared harm calibrator**
from v48.40-B / v48.41-A.  It adds only a small component-specific residual readout
for supported physical factors.  Each residual consumes a **detached** copy of the
same shared, regime-free harm evidence.  Therefore a DRS/deployability/gap residual
can specialize without sending its gradient back through the shared OCAF bridge.
This is hierarchical partial pooling rather than full component factorisation.

The residual heads are zero-initialised, so enabling HPFR is exactly identical to the
shared-harm reference at initialization.  Their correction is bounded with tanh and
a preregistered raw-logit scale of 0.50.  This deliberately avoids the falsified
v48.39 unbounded-factor route.  Global component reliability and the exact
noncompensatory max-veto remain unchanged.  No regime id, regime head, or
regime-specific threshold is introduced.

### Algorithm change 2: clean re-test of bounded monotone rank-benefit skip

The v48.41 rank skip is not modified conceptually.  The positive softplus gain from
frozen `rank_adv` remains inside the bounded HAF benefit residual.  v48.42 fixes the
engineering contract that previously prevented C/D from reaching certificate and
re-runs it as an independent arm.  It is not promoted to a proven mechanism until
C has an authoritative RC=0/20 result and improves the preregistered benefit/frontier
metrics.

### Diagnostic change

Calibration proposal rows now include the complete ordered teacher component-veto
term vector (`DRS, deployability, gap, hard, harm_proxy`) in addition to the maximum
veto margin and predicted component diagnostics.  This is diagnostic-only and does
not change selection, labels, thresholds, or gate logic.  It permits direct
false-veto / false-safe attribution by physical component in the next round.

### Preregistered v48.42 2x2

- A: shared dual-task OCAF + shared harm calibrator, no component residual, no rank skip.
- B: A + detached bounded component-specific harm residuals.
- C: A + engineering-fixed bounded monotone rank-benefit skip.
- D/main: B + C.

Interpretation is fixed before results are read: B>A supports partial-pooling harm
specialization; C>A supports underused frozen rank evidence; D>B,C supports
complementarity.  If B does not improve deployability-dominated Near false vetoes,
stop adding generic harm capacity and move to deployability observation/teacher
identifiability and data support.  If C fails after the engineering repair, do not
restack ranking losses.  If learned evidence approaches proposal-oracle behavior but
Near still fails only through minimum support / Wilson bounds, treat statistical
power as the limiting factor rather than continuing post-hoc algorithm tuning.

### Explicit non-repetition

Still excluded: threshold-grid densification, top-k expansion, aggressive positive
oversampling, hardest-negative population distortion as the main mechanism, generic
pairwise/listwise restacking, full joint stage-2, learned admission residual,
v48.38 one-sided tail losses, v48.39 unbounded benefit/harm, v48.40 frontier-tanh,
v48.41 full component-factorized harm, and any Safe/Near/Contact routing.


## v48.41 — FCFR / FACTORIZED COMPONENT FRONTIER RESERVE (2026-08-07)

### Evidence-based diagnosis of v48.40

All four v48.40 arms (A/B/C/D-main) terminate as authoritative, pipeline-valid
`RC=20` runs with `certificate_executed=true`, `gate_evaluated=true`, and the same
`development_rule_fit` dominant failure layer.  Dataset/protocol hashes, proposal
row identities/teacher labels, factor caches, model/training contracts, and stage
transfer are consistent across arms, so the ablation is usable as algorithm evidence
rather than an engineering-failure comparison.  Proposal-oracle support remains
feasible (Near top-5 contains 9 safe-positive groups; Contact contains 20), placing
the remaining bottleneck after candidate generation.

The v48.40 2x2 yields a clear asymmetric conclusion.  Dual task OCAF (B vs A) is
meaningful: on Precision certificate rows the Near harmful-vs-safe-positive
conditional harm AUC rises from roughly 0.336 to 0.482 and overall harm AUC from
roughly 0.635 to 0.677; paired group bootstrap gives a positive conditional-harm
delta around +0.143 and positive overall-harm delta around +0.042.  Contact overall
harm AUC also improves and conditional harm moves in the positive direction.
Therefore benefit/harm interaction-gradient decoupling is retained.

Frontier-normalized component regression is not retained.  C is approximately
neutral relative to A, while adding it on top of dual OCAF (D vs B) reduces Near
opportunity discrimination and rare-frontier harm discrimination; Contact
conditional harm also falls.  The `frontier_tanh` target therefore fails its causal
ablation and v48.41 returns to raw signed component-margin regression.

Near still exhibits the v48.39/v48.40 failure signature: safe-positive recovery
candidates have useful benefit ordering (certificate opportunity AUC about
0.73--0.78 in the bounded-benefit arms), but the aggregate harm path false-vetoes
most of them.  Under the A reference, rare safe-positive-vs-harmful conditional harm
AUC is below random; B materially improves it but does not make the shared rule
feasible.  Contact remains a dual bottleneck: benefit opportunity AUC is only about
0.57--0.59 and harmful false-safe selections remain frequent.  However the frozen
preference `rank_adv` has substantially stronger Contact safe-positive ordering
(AUC about 0.73), indicating useful already-learned benefit information that the
current sparse opportunity calibrator underuses.

### Algorithm change 1: component-factorized harm interaction and calibration

v48.40 decoupled benefit from harm but the harm task still forces physically
different veto factors (DRS, deployability, gap, hard-rule, harm-proxy) through one
trainable harm OCAF bridge and one shared MLP.  The supported coordinates have
substantially different prevalence and observation dependence, so dense
deployability gradients can still rotate the representation used by the rarer DRS
and gap frontiers.  FCFR extends the validated task-level decoupling *inside the
harm factorization*: every physical component receives its own
observation-conditioned action bridge and its own one-output calibrator.

All branches consume the same candidate-minus-nominal executable action and the same
nominal observation.  No Safe/Near/Contact label, router, head, threshold, or
case-specific policy is added.  The global support reliability and exact
noncompensatory max-veto are unchanged.  Reliability-zero coordinates receive no
component supervision and cannot enter the learned reserve, while independent
measured hard vetoes remain intact.
For the current global support contract `[1,1,1,0,0]`, reliability-zero harm
coordinates use parameter-free zero-context placeholders and skip their calibrator
forward.  This is an exact compute-only optimization: their effective semantic logit
remains the same fixed non-harm prior, and they remain excluded from the learned
reserve.

### Algorithm change 2: bounded monotone rank-benefit skip

The frozen preference advantage already contains useful Contact recovery ordering
that the learned benefit frontier fails to exploit.  FCFR adds one low-capacity
positive-gain skip from detached `rank_adv` into the **bounded** HAF benefit residual.
The gain is parameterized by `softplus`, so it cannot invert the frozen ranking.
The existing tanh-bounded HAF residual remains in place, deliberately avoiding the
falsified v48.39 unbounded-benefit parameterization.  No new ranking loss, regime
routing, or deployment threshold is introduced.

### New diagnostic contract

Inference/calibration now publishes the five effective component-harm probabilities
and signed predicted component margins into proposal rows.  This does not change
selection.  It allows the next run to identify whether DRS, deployability, or gap is
responsible for Near safe-positive false vetoes and Contact harmful false-safe errors
instead of inferring component causes from an aggregate max.

### Preregistered 2x2 ablation

- A: v48.40-B reference (dual task OCAF + raw component margins).
- B: A + component-factorized harm interaction/calibrators.
- C: A + bounded monotone rank-benefit skip.
- D/main: B + C.

Interpretation is fixed before the next result is read: B>A supports within-harm
component interference; C>A supports underused frozen rank evidence; D>B,C supports
complementarity.  If B fails to improve the Near rare-frontier false-veto pattern,
do not add another generic harm loss; inspect per-component observation/teacher
identifiability and data support.  If C fails despite strong frozen Contact rank AUC,
do not restack another pairwise/listwise objective.

### Explicit non-repetition and scientific protocol

Not repeated: threshold-grid densification, top-k expansion, aggressive positive
oversampling, hardest-negative population distortion as the main mechanism, generic
pairwise/listwise restacking, barrier/eligibility continuation, full joint stage-2,
learned admission residual, v48.38 one-sided tail losses, v48.39 unbounded
benefit/harm factors, v48.40 `frontier_tanh`, or regime-conditioned routing.

Because multiple development cycles have already inspected the same certificate
outputs, those certificate sets should no longer be described as pristine one-shot
independent evidence in the paper.  v48.41 mechanisms/arms are preregistered here
before execution; further coefficient/threshold tuning should use development data,
while untouched test roots and gate-authorized closed-loop evaluation remain the
final evidence for CCF-A claims.

## v48.39 — DRFR / DYNAMIC-RANGE FRONTIER RESERVE (2026-08-07)

### Evidence-based diagnosis of v48.38

The v48.38 primary D run does **not** end in a Natural-gate rejection.  Its
authoritative terminal state is `RC=30`, `pipeline_valid=false`,
`failure_stage=certificate`, `gate_evaluated=false`.  Ablation B, the other arm
using the deterministic joint reserve, has the same failure.  A and C reach valid
`RC=20` Natural-gate decisions.  The D/B RC=30 is an engineering contract failure:
RFR intentionally skips identity/final optimization and writes an honest zero-update
summary (`epochs_completed=0`, `history=[]`, `best_epoch=0`), while the inherited
metric-calibration checker still requires `best_epoch` to occur in `history` and
terminates with `best epoch 0 not found`.  v48.39 accepts such a stage only through
explicit, SHA-verified provenance back to the real factor-stage training history;
it does not fabricate a fake identity epoch.

The engineering stop hides a second, independent v48.38 algorithm-semantic defect.
Factor losses supervise candidate benefit/component logits after the nominal row has
already been pinned to zero.  v48.38 deployment reserve instead subtracts the
*pre-pin* nominal logit again.  For component harm this cancels the semantic `-2`
prior and shifts the physical safety margin by approximately `+0.05`, so training
and deployment do not use the same safety coordinate.  This explains the B/D
development all-abstain signature.  Offline replay with the aligned coordinate
recovers some candidates, but a dense 31-point shared-rule search remains infeasible;
therefore coordinate alignment is necessary but not sufficient for `RC=0`.

The v48.38 ablation also falsifies the one-sided tail-calibration loss as a primary
solution.  C (tail correction without reserve) does not improve over A on the valid
certificate: Near-contact still selects zero positives, while Contact keeps the
same three positives with more selections/harm and slightly worse point precision.
The tail losses are therefore disabled in v48.39 instead of being up-weighted.

### Representation diagnosis: signed factor dynamic-range ceiling

The remaining factor representation has useful ranking signal but cannot express the
physical target magnitude in the frontier tail.  With the v48.38 component
parameterization `tanh(raw)*6 + prior(-2)` and `tau_h=0.025`, the largest representable
positive physical violation margin is approximately `0.10`.  The uploaded v48.38
development teacher component-veto targets reach approximately `0.95` in the harmful
tail.  Correspondingly, observed predicted worst-component margins remain around
`0.06--0.09` in the high tail while severe teacher violations are an order of
magnitude larger.

The HAF benefit residual has the analogous limitation: `tanh(raw)*0.75` with
`tau_b=0.05` contributes at most `0.0375` physical headroom, whereas safe-positive
development teacher headroom has a median around `0.57` and reaches about `0.61`.
This is compatible with the empirical pattern seen since v48.37: ranking AUC can be
non-trivial while absolute frontier admission remains badly calibrated.  Another
threshold grid, top-k change, or learned admission residual cannot recover
information that the factor parameterization cannot represent.

### Algorithm change: dynamically expressive signed physical factors

DRFR keeps the validated HAF/OCAF factor semantics but removes the artificial
`tanh` range ceiling at the final physical-factor residuals.  New zero-initialized
linear residual modes are available independently for benefit and component harm:

- `direct_recovery_evidence_unbounded_benefit_factor=true`
- `direct_recovery_evidence_unbounded_harm_factors=true`

The final projections are still zero-initialized, so step-zero behavior exactly
preserves the source/prior factor coordinate.  Smooth-L1 signed physical-margin
supervision then learns the required magnitude without a hard `tanh` saturation.
The representation remains one observation-conditioned continuous factor system;
no regime label, regime branch, regime-specific loss, or regime-specific threshold
is introduced.

### Algorithm change: factor-aligned noncompensatory reserve

DRFR retains the useful idea of a deterministic noncompensatory reserve, but corrects
its coordinate semantics.  With `reserve_factor_alignment=true`, deployment uses
the exact benefit/component coordinates supervised by the factor losses, rather than
subtracting a pre-pin nominal a second time.  Supported component reliability still
defines which learned harm coordinates enter the learned reserve; unsupported
coordinates remain outside the learned max while the independent measured hard-veto
path is preserved.

For the aligned physical factors,

`r = min(benefit_headroom, -max_supported(component_violation_margin))`.

This is a single monotone continuous reserve shared across Safe, Near-contact and
Contact.  Positive benefit can never compensate a violated supported safety factor.
The shared calibration/gate protocol and top-k=5 remain unchanged.

### What is retained, removed, and not repeated

Retained because prior ablations or diagnostics support them:

- OCAF observation-conditioned physical interaction bridge;
- HAF signed benefit boundary and dense factor supervision;
- factor-preservation principle (do not rotate learned physical factors in a sparse
  identity/admission stage);
- component-veto/noncompensatory safety semantics;
- reliability-aware learned component support plus independent hard veto;
- one shared continuous rule and the pre-registered top-k/gate protocol.

Removed from the v48.39 primary path because v48.38 does not support them:

- one-sided component underestimation / safe-positive overestimation tail losses;
- joint-reserve regression and boundary loss;
- learned identity/admission residual optimization.

Not repeated because prior versions already tested them without solving the gate:
threshold-grid densification, top-k expansion, aggressive positive oversampling,
hard-negative population distortion as the main mechanism, repeated pairwise/listwise
ranking additions, barrier/eligibility continuation, full joint stage-2 refinement,
or any regime-conditioned routing.

### Pre-registered v48.39 dynamic-range ablation

All arms use the corrected factor-aligned deterministic reserve, the same data,
proposal top-k=5, calibration, certificate and one shared rule.  They differ only in
which physical factor is allowed sufficient signed dynamic range:

- **A:** bounded benefit + bounded harm (aligned-reserve control);
- **B:** bounded benefit + unbounded harm;
- **C:** unbounded benefit + bounded harm;
- **D / primary:** unbounded benefit + unbounded harm.

A/B/C can be launched concurrently; D is the primary run and is not duplicated by
default.  This cleanly tests whether the remaining bottleneck is harm range, benefit
range, or their complementarity, without reusing the disproven v48.38 tail-loss
ablation.

### Engineering hardening and runtime

- The metric-calibration contract now supports an explicitly audited zero-update
  materialized stage by resolving its metric row to the SHA-verified factor-stage
  training summary/checkpoint.  Missing or tampered provenance fails closed.
- Reserve-only materialization remains zero optimizer steps and byte-identical to
  the factor checkpoint; no fake identity epoch is created.
- All new factor semantics are included in checkpoint metadata, model/training
  contracts and factor-cache fingerprints so a bounded v48.38 checkpoint cannot be
  silently reused as a v48.39 dynamic-range factor.
- The empirically unhelpful identity stage remains skipped, retaining the v48.38
  runtime saving.  A/B/C ablations are launched concurrently on the same two-GPU
  Balanced/Precision layout with CPU/data-loader caps to reduce wall-clock time.

### Dataset/statistical interpretation

The current dataset is a material statistical constraint but is not yet sufficient
to explain away the gate failure.  Near-contact has only about 25 safe-positive
training candidates (about 1.75%) and roughly eight safe-opportunity groups in the
shared-rule development slice, close to the gate's minimum-selection boundary;
Contact is less sparse but still low-prevalence.  Nevertheless the proposal oracle
is feasible in both regimes.  Therefore v48.39 first removes the demonstrated
engineering, coordinate, and representation ceilings.  If an aligned dynamic-range
selector approaches oracle point behavior but only confidence bounds / minimum
support remain infeasible, the remaining limitation should then be reported as a
dataset statistical-power ceiling rather than hidden by further algorithm tuning.

## v48.38 — RFR / ROBUST FRONTIER RESERVE (2026-08-07)

### Evidence-based attribution from the complete v48.37 HAF ablation

All four v48.37 arms (A/B/C/D) finish with authoritative `pipeline_valid=true` and
RC=20.  Training, model and stage-transfer contracts pass, so this is a Natural-gate
rejection rather than an engineering stop.  The dominant gate layer remains
`development_rule_fit`; Near-contact and Contact retain feasible proposal oracles,
which means candidate support is not the immediate bottleneck.

The v48.37 ablation gives a useful causal result.  On the Precision selector,
factor-preserving admission (C) reduces excessive selection and raises certificate
point precision from 1/33=3.0% to 1/7=14.3% in Near-contact and from 1/75=1.3% to
2/26=7.7% in Contact.  Full HAF (D) reaches 1/6=16.7% and 2/20=10.0%.  Benefit
headroom alone (B) is weaker but has a positive Near-contact signal: positives rise
from 1 to 2 and recall from 0.111 to 0.222; for Balanced Near it also reduces
harmful selections from 10 to 5.  Thus factor preservation is the strongest
validated v48.37 mechanism, while signed benefit headroom is retained as a useful
physical anchor rather than treated as a complete solution.

However, D remains far from the confidence-bound gate.  It over-abstains and still
selects harmful tail cases.  More importantly, the remaining error is a
**frontier-tail inversion**: several truly harmful candidates with large positive
teacher component-veto margins are predicted as low harm, while genuine
safe-positive candidates with teacher safety margin -0.05 are often predicted as
high harm and low opportunity.  Broad harm ranking remains moderate, but harm AUC
inside the exact low-harm deployable tail collapses.  The same pattern exists on
development data, so it is not a certificate-only shift.

The learned admission residual itself is also not supported by v48.37.  For the
primary D arm, both Balanced and Precision identity-stage best checkpoints are
`epoch=0`; later optimization lowers its training loss while exact deployment-risk
metrics worsen and valid-safe admission remains zero.  Repeating another admission
fine-tuning stage would therefore repeat a failed mechanism rather than exploit a
new signal.

### Algorithm change: deterministic joint physical reserve

RFR removes the learned admission residual from the primary method.  It composes the
already observation-conditioned physical factors with one shared, monotone reserve.
For candidate `a` relative to nominal `a0`, let

`b = tau_b * (benefit_logit(a) - benefit_logit(a0))`

be signed benefit headroom, and let each supported component-harm margin be

`m_k = tau_h * (harm_component_logit_k(a) - harm_component_logit_k(a0))`.

The continuous safety headroom is `s = -max_k m_k`, and the joint recovery reserve is

`r = min(b, s)`.

Admission is `sigmoid(r / tau_r)`.  Therefore `r>0` if and only if the candidate is
simultaneously above the physical benefit boundary and below every supported
component-harm boundary.  This is a noncompensatory continuous AND expressed by a
single scalar reserve; it uses no Safe/Near/Contact identifier, branch, threshold,
or case-specific strategy.  The existing shared-rule fitter still applies one rule
to all regimes.

### Algorithm change: robust frontier-tail calibration

The factor stage receives three boundary-focused continuous objectives in addition
to the retained OCAF/HAF losses:

1. **Joint-reserve regression.**  The teacher reserve is
   `r* = min(teacher_advantage-positive_gain, -max_k teacher_component_margin_k)`.
   A Smooth-L1 term anchors the model reserve to the same physical frontier, with
   extra continuous weight near `r*=0`, where admission decisions are sensitive.
2. **Harm underestimation penalty.**  For component targets that truly violate the
   tolerance, penalize only `relu(target_margin - predicted_margin)`.  This directly
   targets dangerous false-safe tail errors without population rebalancing.
3. **Safe-positive harm-overestimation penalty.**  For naturally occurring
   continuous safe-positive rows, penalize only `relu(predicted_margin-target_margin)`.
   This counterbalances over-conservatism that rejects true recovery opportunities.

The two one-sided terms are deliberately asymmetric in error direction but use the
same physical margins and natural population in every regime.  They are not
regime-conditioned losses.

### Optimization change: factor stage is directly deployable

Because v48.37 shows no positive evidence for learned admission refinement, the RFR
primary path skips identity/final optimization after the factor stage.  The
identity/final checkpoint files are materialized byte-for-byte from the factor
checkpoint so legacy downstream contracts remain intact, while provenance records
`epochs_completed=0`, zero optimizer steps and an empty history.  Stage-transfer
checks require exact SHA/tensor identity.  This removes a stage that was both
computationally expensive and empirically harmful, without changing calibration,
certificate or gate semantics.

### Pre-registered v48.38 mechanism ablation

All arms keep the same data, proposal generator, top-k=5, gate and one shared rule:

- **A — HAF reference:** v48.37 benefit headroom + factor-preserving learned
  admission residual.
- **B — Reserve only:** deterministic joint reserve + joint-reserve regression,
  without the new bidirectional tail losses.
- **C — Tail only:** v48.37 factor-preserving learned admission residual + the new
  bidirectional tail losses, without deterministic joint reserve.
- **D — Full RFR:** reserve + bidirectional tail calibration.  D is the primary
  v48.38 run, so the parallel ablation launcher runs A/B/C concurrently by default
  and does not waste time duplicating D unless `RERUN_D=1` is explicitly requested.

This design tests two distinct hypotheses: whether the learned admission residual is
the wrong composition mechanism (B) and whether low-harm-tail calibration is the
remaining factor error (C).  D tests complementarity.

### Runtime and engineering hardening

- A/B/C ablations can run concurrently.  Each arm uses the same two-GPU
  Balanced/Precision layout; on a two-GPU machine this means up to three ablation
  processes per GPU.  CPU BLAS threads and data-loader workers are capped to avoid
  CPU/I/O oversubscription.
- The factor-cache fingerprint is keyed to factor-stage semantics rather than a
  cosmetic arm name, preventing stale semantic reuse while allowing safe repeated
  runs with identical factor configuration.
- Wrappers explicitly reset all v48.38/v48.37 mechanism switches, top-k, resume and
  cache controls, avoiding ambient-shell contamination across concurrent arms.
- `joint_reserve` is covered by model/training/stage-transfer contracts.  The
  skipped identity stage is only legal when the architecture declares reserve-only
  mode and factor/identity tensors are byte-identical.
- The no-training materializer writes truthful zero-epoch provenance rather than
  copying a factor-stage training record, preventing downstream analyses from
  falsely interpreting a skipped stage as trained.  It materializes only `best.pt`
  (same-filesystem hard link when possible, otherwise a byte copy) instead of
  duplicating factor-stage `latest.pt`/epoch checkpoints.
- The learned reserve excludes reliability-zero harm coordinates before the max.
  This is necessary for the observed global support `[1,1,1,0,0]`: including the
  neutral zero margins would otherwise force safety headroom `<=0` and make every
  positive reserve impossible.  The safe-positive tail loss nevertheless uses the
  complete teacher component-veto definition, so an unsupported learned coordinate
  cannot relabel a teacher-harmful example as safe-positive.
- Fresh v48.38 wrappers force `RESUME_AFTER_ADAPTATION=0`; stale shell state cannot
  accidentally trigger a no-retraining resume path.

### Dataset ceiling assessment at the time of this change

The current RC=20 cannot be attributed to the dataset alone: the proposal oracle is
feasible in both Near-contact and Contact, proving that the existing candidate/data
pipeline exposes recoverable actions.  Nevertheless, safe-positive support is
statistically sparse and concentrated.  Near-contact train support is about
25/1425 deployable candidates (1.75%, 11 groups, 7 scenes); Contact is about
106/4086 (2.59%, 41 groups, 17 scenes).  Development support is also small, with
Near-contact especially tight.  This makes the one-sided confidence-bound gate hard
and increases representation variance.  RFR therefore targets the observable tail
errors first rather than reconstructing the dataset.  If RFR repairs tail
calibration but confidence bounds remain support-limited, the dataset becomes the
next dominant ceiling and should be reported explicitly rather than hidden by
threshold relaxation.

### Previously tested directions intentionally not repeated

RFR does not repeat threshold-grid densification, top-k expansion, aggressive
positive oversampling, hardest-negative population distortion, listwise/pairwise
ranking as the primary fix, barrier continuation, full joint identity refinement,
or another learned admission residual.  Those directions were already tested in
v48.28-v48.37 and did not resolve the shared-rule/certificate failure.

### What remains unchanged

- No regime classifier or regime-conditioned policy, threshold, loss weight or
  routing logic.
- Same Safe/Near/Contact datasets and split roots.
- Same source experts, OCAF observation-conditioned bridge, component-veto physical
  semantics, proposal generator and top-k=5.
- Same positive-gain threshold, component tolerances, shared-rule fit/verify split,
  certificate confidence procedures and Natural gate.
- No test-root access is introduced during training, ablation selection or repair.


## v48.37 — HAF / HEADROOM-ALIGNED FRONTIER (2026-08-06)

### Evidence-based motivation

The repaired v48.36 OCAF pipeline is valid and terminates with the *natural* gate
exit code RC=20.  The failure is no longer an engineering stop.  The gate-failure
decomposition localizes the dominant layer to `development_rule_fit`: for both
balanced and precision, Near-contact and Contact have a feasible proposal oracle,
but no single shared development rule satisfies the preregistered admission
constraints.

The candidate set is therefore not the main bottleneck.  In the certificate
proposal top-5, Near-contact contains 9 safe-positive groups and Contact contains
20; the corresponding proposal-oracle precision lower confidence bounds are
approximately 0.846 and 0.924.  In contrast, the learned selector selects very few
safe-positive actions and many harmful actions.  A denser 21/31-point shared-rule
search remains infeasible, excluding the original 15-point threshold grid as the
primary cause.

A second diagnostic is stage drift.  The v48.36 identity stage improves proposal
identity/ranking but materially worsens the factor-stage supervised-risk metric
while jointly updating benefit, harm, admission and the OCAF interaction bridge.
This is especially problematic because safe-positive support is sparse.  The
observed failure signature is consistent with a useful ranking representation
being rotated away from the absolute physical benefit/harm boundaries needed by
the shared admission rule.

### Algorithm change: dual signed headroom alignment

HAF retains OCAF's observation-conditioned action frontier and adds a *regime-free
continuous* benefit headroom target.  For every candidate relative to nominal,

`benefit_headroom = teacher_advantage - positive_gain`.

The candidate-minus-nominal opportunity logit is mapped to a signed physical
margin with a temperature and optimized with Smooth-L1 regression.  Consequently
zero logit / opportunity probability 0.5 is explicitly anchored at the
preregistered positive-gain boundary, symmetric in spirit with the existing
signed component-harm margin heads.  The same target is used for Safe,
Near-contact and Contact; no regime identifier, bucket-conditioned threshold, or
case-specific policy is introduced.

### Algorithm change: factor-preserving admission refinement

HAF makes the two-stage optimization semantics explicit:

1. **Factor stage:** fit OCAF's observation-conditioned bridge plus compact benefit
   and component-harm factors, including signed benefit headroom.
2. **Admission stage:** freeze the benefit head, harm head and OCAF interaction
   bridge byte-for-byte, and optimize only the shared admission residual on top of
   those physical factors.  The admission prior is detached during this stage.

This preserves the factor stage as a stable continuous physical coordinate system
while still allowing the selector to learn when evidence is sufficient to leave
nominal.  It directly targets the observed admission/ranking mismatch without
changing the proposal generator, source experts, data split, certificate or gate.

### Engineering/provenance hardening

- The stage-transfer checker now accepts a frozen OCAF bridge only when the stage
  architecture explicitly registers `interaction_bridge_trainable_this_stage=false`;
  all such tensors are then required to be byte-identical across the transfer.
- The OCAF training contract understands factor-preserving admission, checks the
  signed-benefit-headroom weight/temperature, and verifies the recorded algorithm
  variant at every stage.
- Factor-cache fingerprints include the new headroom settings and algorithm
  variant, preventing stale v48.36 factor checkpoints from being silently reused.
- The v48.37 wrappers clear ambient factor-cache variables by default, force the
  preregistered algorithm switches/top-k/context rather than inheriting stale shell
  values, assign unique output directories, and default resume to off, reducing
  avoidable engineering RC=30 failures and mislabeled runs.
- The repaired v48.36 terminal-state/gate protocol is intentionally reused so that
  a v48.37 gate change is attributable to the algorithm rather than a new gate.
- The learning-gate diagnostic now resolves v48.36 terminal state with the v48.36
  resolver and is refreshed after terminal publication; this removes the stale
  `authoritative_state=false` snapshot seen in the uploaded RC=20 result.

### What is deliberately unchanged

- No Safe/Near/Contact classifier, one-hot regime input, routing branch, or
  regime-conditioned strategy/threshold.
- Same proposal generator and `top_k=5` for the primary experiment.
- Same frozen source experts and observation-consistent physical context.
- Same positive-gain definition, component-veto tolerances, shared-rule fitting,
  fit/verify scene separation, certificate statistics and Natural gate.
- Same datasets and test-root access policy.

### Pre-registered mechanism ablation

The release provides a 2x2 ablation under the identical gate/protocol:

- **A:** repaired v48.36 training semantics (no new headroom regression, joint
  identity update).
- **B:** signed benefit-headroom alignment only.
- **C:** factor-preserving admission only.
- **D:** full HAF (B+C; primary v48.37 method).

The expected falsifiable signature is not merely a lower loss.  A successful HAF
run should produce a *valid shared development rule* in both Near-contact and
Contact, increase safe-positive selected count/precision while reducing harmful
selected UCB, and preserve the noncompensatory component-harm contract.  If B helps
but C does not, calibration of benefit semantics is dominant; if C helps but B
does not, stage drift is dominant; if D is materially stronger than either alone,
the two mechanisms are complementary.

### Three-regime publication objective

The intended paper-level behavior remains one continuous policy: preserve nominal
utility/non-inferiority in Safe scenes; convert continuous recovery headroom into
larger minimum TTC / fewer unsafe interventions as situations approach contact;
and, after contact, prefer actions with positive recoverability headroom that
reduce secondary-collision exposure while retaining post-contact TTC and route
progress.  These are audit outcomes of one shared physical frontier, not three
case-specific control laws.

## v48.36.3 — RC30 TERMINAL-STATE ATTEMPT CONTRACT HOTFIX (2026-08-06)

### Scope and attribution

This is an **engineering-only** release on top of the unchanged v48.36 OCAF
algorithm. It does not alter the observation-conditioned interaction bridge,
candidate generator, model heads, losses, training data, checkpoint selection,
shared deployment rule, certificate statistics, thresholds, gate, Safe/stress
protocol, or the unified continuous treatment of Safe/Near/Contact.

The uploaded result is later than the first v48.36.1 stage-transfer failure: the
v48.36.2 no-retraining repair and resume contract both passed, calibration and the
Near/Contact certificate both executed, and balanced and precision each returned a
natural certificate-controller RC=20. The active RC=30 was produced afterwards by
the terminal-state audit.

### Root cause

`run_v48_36_ocaf_dedicated.sh` created and exported `V4836_ATTEMPT_ID`, but the
copied v48.36 calibration launcher still read `V4835_ATTEMPT_ID`. Consequently
`GATE_SPEC.json`, `dedicated_recalibration_status.json`, both candidates’
`CERTIFICATE_CALIBRATION_COMPLETE.json` and `SAFE_REGIME_STATUS.json`, and the
natural `GATE_FAILED.json` were written under `attempt_id=legacy-untracked`.

The authoritative resolver correctly refused to attach that gate marker to the
active v48.36 attempt. Its RC=4 contract rejection was normalized by the controller
to pipeline RC=30. The underlying algorithmic outcome was unchanged: no candidate
passed the Natural gate, so the authoritative result should be pipeline-valid
RC=20, not engineering RC=30.

### Engineering changes

1. Unified the v48.36 controller and calibration launcher on
   `V4836_ATTEMPT_ID`; the controller passes it explicitly and the launcher now
   fails closed if it is absent or `legacy-untracked`.
2. Added `check_v48_36_certificate_status_contract.py`. Before publishing terminal
   completion it checks that gate specification, candidate selection, candidate
   certificate completion, Safe status, gate/block markers, and NEXT_COMMANDS
   state all belong to the same non-legacy attempt and satisfy the exact RC=0/20
   contract.
3. Added an exact no-training/no-calibration repair path for the observed
   v48.36.2 terminal-state signature. It verifies adaptation/checkpoint hashes,
   both natural gate failures, certificate validity, archived gate equivalence,
   source/protocol identity, and test-root seals; backs up every touched file;
   changes provenance/status metadata only; reruns both contracts; and rolls back
   byte-for-byte if any post-repair audit fails.
4. Propagated `v48.36.3-TERMINAL-STATE-HOTFIX` through controller, calibration,
   adaptation completion and resume metadata while retaining compatibility with
   the v48.36.2 stage-transfer repair artifact.
5. Expanded post-certificate failure archiving so diagnostics, gate specification,
   candidate selection and terminal markers are preserved before an engineering
   failure publishes RC=30.
6. Added regression tests for stale-attempt rejection, exact RC=20 repair,
   rejection of an altered gate outcome, missing-attempt fail-closed behavior,
   post-repair rollback, and controller/calibration namespace wiring.

### Validation

- Version-focused compatibility matrix: 58 passed, 1 CUDA-only test skipped.
- Python `compileall`: PASS.
- 69 Shell scripts under `scripts/` and `tools/`: `bash -n` PASS.
- New tool import/`--help`: PASS.
- Algorithm implementation source is byte-identical to the uploaded v48.36.2 code.
- Full historical repository suite: 352 passed, 1 skipped, 38 inherited failures.
  Thirty-seven failures are missing archived v48.12–v48.32 scripts and one is an
  already-present v48.30 source-text assertion; all are reproducible in the
  unmodified uploaded code and are outside the active v48.36 pipeline.
- The uploaded results ZIP omits `.pt` files. Exact repair authorization therefore
  remains fail-closed locally and must run where the recorded checkpoints still
  exist.

## v48.36.2 — RC30 STAGE-TRANSFER CONTRACT HOTFIX (2026-08-06)

### Scope and attribution

This release is an **engineering-only hotfix** for v48.36 OCAF. It does not change
candidate generation, the observation-conditioned action representation, model
heads, losses, checkpoint-selection metrics, source-expert prior, shared-rule
fitter, certificate thresholds, datasets, or the gate.

The uploaded v48.36.1 run passed both A30 CUDA group-broadcast preflights and
completed factor and identity training for balanced and precision. Both variants
then stopped at `stage_transfer_integrity` with RC=31. The controller normalized
the two adaptation RCs to pipeline RC=30 before calibration or certificate.

The failure was a false integrity rejection. The v48.36 variant runner explicitly
trained `direct_evidence_interaction_bridge` during the identity stage and recorded
that prefix in `STAGE_ARCHITECTURE.json`, but it still invoked the legacy
`check_v48_32_stage_transfer.py`. That checker permits only the benefit, harm and
admission calibrators. It therefore reported all ten changed OCAF bridge tensors as
"frozen parameter drift" even though they were registered trainable parameters.
No encoder, source expert, proposal generator, or other frozen parameter was
reported changed.

### Engineering changes

1. Added `tools/check_v48_36_stage_transfer.py`, with an explicit controller-provided
   prefix contract, an approved-prefix allowlist, exact architecture/trainable-set
   agreement, OCAF context checks, full parameter-diff reporting, and fail-closed
   rejection of any true frozen drift.
2. Replaced the v48.32 checker in `adapt_ocrap_v48_36_ocaf_variant.sh`. The normal
   training path now authorizes the OCAF interaction bridge only when both the
   runner contract and stage metadata register it.
3. Added `tools/finalize_v48_36_adaptation_variant.py`. Completion metadata is now
   written atomically after checkpoint/completion SHA256 verification and records
   exact stage-2 trainable prefixes, the stage-transfer contract version, exact
   deployment eligibility and implementation version.
4. Added an exact no-retraining repair path:
   `repair_v48_36_1_stage_transfer_failure.py` and
   `repair_v48_36_1_stage_transfer_with_v48_36_2.sh`. It accepts only the observed
   signature (both variants RC=31 at stage transfer, no certificate/gate/test access,
   only interaction-bridge tensors misclassified), rechecks the existing checkpoint
   bytes, archives the old failure evidence, regenerates valid completion metadata,
   and authorizes `RESUME_AFTER_ADAPTATION=1`.
5. Extended `check_v48_36_resume_contract.py` to accept the repaired signature while
   still rejecting algorithmic RC=20, prior certificate artifacts, changed source or
   protocol identity, changed checkpoint hashes, or any unregistered RC=30.
6. Corrected a latent reference-arm contract bug: when OCAF context is used with
   `IDENTITY_TRAIN_ALL=0`, the identity-stage expected set is admission plus the
   interaction bridge, not admission alone.
7. Added a v48.36-specific failure-signature extractor. New failures no longer carry
   the stale `v48_34_failure_signature` event name.
8. Added regression tests reproducing the legacy false rejection, accepting the
   registered OCAF bridge, rejecting real encoder drift, rejecting architecture
   mismatch, repairing both variants without retraining, and validating the resume
   authorization end to end.

### Algorithm interpretation

The uploaded archive contains training summaries but no calibration, shared frozen
rule, certificate, or gate outputs. Precision identity training showed debugging
signals (at least one valid safe admission and non-all-abstain on adaptation-dev),
while balanced remained all-abstain. These are checkpoint-selection diagnostics,
not independent evidence. No algorithm or hyperparameter change is made before the
repaired run reaches a valid RC=0 or RC=20.

### Validation

- Focused v48.35/v48.35.1/v48.35.2/v48.36/v48.36.2 matrix: 48 passed and one
  CUDA-only test skipped when run by file; only non-fatal Transformer warnings.
- New v48.36.2 tests: 6 passed, including exact no-retraining repair and resume.
- Python `compileall`: PASS.
- All 68 shell scripts under `scripts/` and `tools/`: `bash -n` PASS.
- All new version-scoped tools: import/`--help` PASS.
- The delivery environment does not contain the omitted checkpoint bytes, WOMD,
  Waymax or A30 GPUs. Byte-level no-retraining authorization is therefore executed
  on the original experiment machine, not claimed from the uploaded archive alone.

## v48.36.1 — RC30 CUDA GROUP-BROADCAST HOTFIX (2026-08-06)

### Scope and attribution

This is an **engineering-only hotfix** for v48.36 OCAF.  It does not change the
candidate generator, unified continuous physical semantics, source-expert prior,
loss weights, shared deployment rule, certificate thresholds, datasets, or gate.
The uploaded v48.36 result is not an algorithm result: both balanced and precision
variants failed in factor-stage epoch 1 before producing a checkpoint, calibration,
certificate, or gate decision.

The first authoritative exception was raised in
`OCRAPModel._direct_nominal_observation_features`.  For a real A30 group with eight
rows and a 529-dimensional nominal-observation vector, CUDA advanced-index
assignment attempted to reconcile `8*529=4232` destination elements with an
incorrectly expanded `529*529=279841` value tensor and aborted in
`ATen/native/cuda/Indexing.cu`.

### Engineering changes

1. Replaced tensor-valued scalar slicing and implicit advanced-index broadcasting
   in candidate-relative and nominal-observation group construction with explicit
   `index_select` + `index_copy_` row operations.
2. Applied the same explicit row gather/scatter discipline to the recovery-set
   tournament and group-relative/set-context adapters, removing the remaining
   group-wise CUDA `tensor[idx] = value` writes from the model path.
3. Replaced the zero-unsafe `sqrt(mean(action^2))` derivative with a float32,
   clamped RMS magnitude gate.  Exact zero action still gives exact zero OCAF
   output, while nominal-row and non-detached diagnostic gradients remain finite.
4. Added `tools/check_v48_36_cuda_group_broadcast_contract.py`.  The main runner
   now executes the exact 141-D action / 529-D observation geometry, including
   backward, on **both configured training GPUs** before index construction or
   adaptation.  A failure is published as an attempt-scoped RC=30 preflight error.
5. Added real-batch-geometry CPU regression, optional CUDA regression, runtime-tool
   regression, and main-runner wiring tests.

### Validation

- Focused v48.35/v48.35.1/v48.35.2/v48.36 matrix: **42 passed, 1 CUDA-only skipped**
  in the CPU delivery environment, with only non-fatal Transformer warnings.
- Exact factor-stage geometry smoke: batch 96, group size 8, action dimension 141,
  observation dimension 529, all outputs and trainable gradients finite.
- Python `compileall`, all 67 shell scripts under `scripts/` and `tools/`, and all
  v48.36 version-scoped tool `--help` imports pass.
- Real A30 execution is still required; no RC=0 or algorithm-performance claim is
  made by this hotfix.

## v48.36 — OCAF (Observation-Conditioned Action Frontier)

- **Root cause addressed:** v48.35.2 used an action-only `physical_relative` evidence context. It could not distinguish the effect of the same braking/steering delta under different continuous clearance, contact-pose, relative-motion, route and occupancy conditions.
- **Unified representation:** added `ObservationConditionedActionFrontierBridge`, combining candidate-minus-nominal executable action with nominal-anchor observation pressure via a multiplicative interaction. No regime ID, regime-specific head, threshold or policy branch is introduced.
- **No scene shortcut:** all action-to-context outputs are bias-free; zero candidate action difference produces exactly zero interaction context.
- **Magnitude preservation:** retained a raw signed-action pathway and added an RMS-gated normalized direction pathway, avoiding LayerNorm-only loss of action magnitude.
- **Conservative transfer:** source-expert consensus prior is explicitly scaled (default `0.50`) so the learned observation-conditioned residual can correct wrong frozen-expert direction without selecting a regime expert.
- **Continuous frontier retained:** the five signed harm components and non-compensatory smooth cap remain mandatory; benefit/residual cannot compensate an unsafe component.
- **Training update:** factor and identity stages train the OCAF bridge; strengthened listwise, tail, margin-regression, frontier-pairwise and safe-utility objectives while keeping one shared rule.
- **Engineering fixes:** replaced the generic 100k calibrator cap with an exact checkpoint trainable-prefix contract; added canonical dataset-root/alias rejection; made 2x2 ablations independent; added missing version-scoped calibration tools; made context/cap contracts arm-aware; included OCAF hyperparameters in cache identity; added v48.36 authoritative RC resolver, atomic attempt-scoped terminal state, and v48.36-only Safe/stress authorization.
- **Paper alignment:** removed the regime-conditioned second admission channel and documented one continuous-headroom admission rule.
- **Validation:** targeted OCAF tests cover exact-zero action, nonzero gradient, action magnitude, nominal-scene anchoring, observation modulation, non-compensatory cap, canonical roots, authoritative RC 0/20, arm-aware 2x2 contracts, and OCAF-specific resume semantics. The revised TeX compiles after fixing the malformed hyperref declaration; the uploaded source did not include its bibliography file.

## v48.35.2 — ENGINEERING-INTEGRITY-AND-AUTHORITATIVE-STATE (2026-08-05)

### Scope

This release is engineering-only. It does **not** change the OC-RAP model, candidate generator, physical-relative representation, continuous frontier, losses, checkpoint ordering, shared-rule fitter, certificate thresholds, datasets, or Natural gate. It exists solely to make a completed run have one unambiguous terminal state and to prevent stale files or diagnostic-script drift from changing algorithm attribution.

### Root-cause attribution for the uploaded result

The uploaded result archive contains two terminal records from different executions:

- `PIPELINE_FAILED.json`: `created_unix=1785887254.9260776`, stage `training_contract`, raw RC=4, normalized RC=30, certificate/gate not executed.
- `V48_35_COMPLETE.json`: `created_unix=1785913147.147986`, raw/certificate/pipeline RC=20, `pipeline_valid=true`, certificate and gate executed.

The later completion is authoritative. `GATE_FAILED.json` and `NEXT_COMMANDS_STATUS.json(reason=natural_gate_failed)` were written immediately before it. The active source run had already cleared the old pipeline marker, but the uploaded ZIP retained the earlier entry. Therefore the apparent repeated RC=30 was a stale-package/state-resolution error, not a second pipeline failure.

A second engineering defect occurred after certificate execution: `summarize_v48_34_gate_failure.py` still required legacy `dev_frozen_rule_{near,contact}_v48.json` files, while v48.35 correctly emits one `dev_frozen_shared_rule_v48.json`. The resulting `FileNotFoundError` was hidden by `|| true`, leaving the core RC unchanged but silently dropping required diagnostics.

### Engineering changes

1. Added `tools/audit_v48_35_run_state.py`, which resolves the terminal state from `V48_35_COMPLETE.json`, attempt IDs, timestamps, and the RC/NEXT_COMMANDS contract. A stale marker can be archived, but a same-attempt contradiction is fail-closed.
2. Added attempt-scoped controller state. Every invocation writes `ATTEMPT_STARTED.json`; previous active terminal files are moved into `status_history/` rather than deleted or mixed with the new attempt.
3. All v48.35 terminal, gate, calibration, candidate-selection, completion, and learning-gate JSON writes used by this path are atomic (`fsync` + `os.replace`).
4. Failure publication now creates an attempt-scoped RC=30 completion, blocked-next-command state, and authoritative-state audit. A refused no-retraining resume also publishes a complete RC=30 terminal state instead of exiting without one.
5. Replaced the gate-decomposition reader so it supports the shared v48.35 development rule and legacy per-regime files. Artifact/protocol errors return nonzero instead of an uncaught traceback.
6. Removed silent post-certificate diagnostics. Learning-gate and decomposition failures now become `post_certificate_diagnostics` RC=30; they cannot be ignored with `|| true`.
7. Adaptation-failure signatures now record their own extraction return code and are written atomically. Signature extraction failure no longer disappears silently.
8. Downstream Safe and stress wrappers require a valid authoritative RC=0 state, rather than trusting marker-file existence alone.
9. Added `tools/package_v48_35_results.py` and `scripts/package_v48_35_results.sh`. Packaging always creates a fresh ZIP in write mode, excludes stale terminal markers/checkpoints by default, replaces generated metadata exactly once, rejects duplicate entries, round-trip verifies every entry hash, and writes a ZIP SHA256 file.
10. Generated follow-up commands quote paths and use the configured `SAFE_WOMD_SOURCE`; they no longer embed an unrelated machine-specific path.
11. Added a version-specific release test launcher. The supported v48.35.2 matrix is isolated from 17 retained historical tests whose referenced v48.12-v48.32 launchers are absent, and from unrelated v50 tests in the mixed research repository. No missing historical scripts were fabricated.
12. Removed cache artifacts and the accidental nested `mnt/data/ocrap_waymax/OC-RAP` copy from the clean release package.

### Terminal-state contract

- `RC=0`: completion is valid, gate passed, `NEXT_COMMANDS.txt` exists, and no active failure marker is allowed.
- `RC=20`: completion is valid, gate evaluated and failed naturally, `GATE_FAILED.json` and blocked-next-command status are active, and any older `PIPELINE_FAILED.json` is stale.
- `RC=30`: completion is invalid for algorithm attribution, an active same-attempt `PIPELINE_FAILED.json` and blocked-next-command status are required, and downstream evaluation is forbidden.

### Validation

- 192 tests in the supported v48.35.2 release matrix passed in isolated batches, with six non-fatal warnings.
- 28 focused v48.35/v48.35.1/v48.35.2 tests passed (11 engineering-integrity, 8 RC30-contract hotfix, 9 continuous-frontier).
- All 57 shell scripts passed `bash -n`.
- `python -m compileall -q -f src tools tests` passed.
- The uploaded result was repaired without retraining: authoritative state is valid RC=20, the stale RC=30 marker was archived, shared-rule diagnostics completed, and the clean result ZIP contains no duplicate entries or top-level stale `PIPELINE_FAILED.json`.

### Non-claims

No algorithm-quality conclusion is introduced by this release. Local validation did not rerun WOMD/Waymax training or certificate computation; it validates state, provenance, script, packaging, and diagnostic contracts only.

## v48.34.1 — RC30-MODEL-CONTRACT-AND-PROGRESS-HOTFIX (2026-08-03)

### Scope

This is an engineering-only hotfix for the uploaded v48.34 run. It does not change the BARRIER-CROSSFIT model, loss weights, proposal policy, dataset, gate thresholds, or algorithm interpretation.

### v48.34 RC=30 attribution

- Both Balanced and Precision adaptation jobs completed successfully (`adaptation_exit_codes=0/0`), produced final checkpoints, and passed stage-transfer integrity.
- The controller failed at `model_inference_contract` before certificate execution. The raw subprocess exit code was 2 and the normalized pipeline code was 30.
- The v48.34 runner passed `--expect-admission-prior-mode barrier_gated_slack` to the older `check_v48_32_model_contract.py`, whose argparse choices contained only `risk_centered`, `benefit_only`, and `safety_slack`. The checker rejected the argument before reading either checkpoint.
- Consequently `certificate_executed=false`, `gate_evaluated=false`, and no algorithm conclusion is permitted from this run.

### Pipeline fixes

1. Added version-specific `check_v48_34_model_contract.py`, including `barrier_gated_slack`, checkpoint/support SHA records, five-component reliability checks, bounded admission, slack temperature/penalty, and inference-contract verification.
2. Updated the v48.34 dedicated and ablation controllers to use the v48.34 checker. The old checker also accepts the new enum for backward-compatible diagnostics.
3. Added `repair_v48_34_rc30_model_contract_with_v48_34_1.sh`. It refuses every failure signature except the observed model-contract parser failure, verifies both final checkpoint hashes and stage-transfer metadata, reruns only model/training contracts, then resumes at certificate calibration without retraining.
4. Added a clean-run wrapper `run_v48_34_1_barrier_crossfit_dedicated.sh` for environments where the server-side v48.34 checkpoints are unavailable.
5. Repair status now distinguishes certificate-controller invocation, completed certificate execution, gate evaluation, raw certificate exit code, and normalized pipeline exit code.

### Exploratory closed-loop, baseline, and visualization fixes

1. Adaptation-dev Near and Contact now use split `evidence_adapt_dev` and the standard WOMD validation source. Contact can no longer silently use `validation_interactive` or a test split.
2. Dataset preflight validates target split, source role, official scenario IDs/source indices, raw TFRecord resolution, and any explicit `@N` scan limit before Waymax execution.
3. OC-RAP Near/Contact Balanced and Precision runs execute concurrently on two GPUs. External methods execute two at a time. Physical comparison defaults to `label_mode=fast` with zero online teacher labels; expensive selected-topk teacher audits are not used for progress-only closed loop.
4. All methods must run on the exact same target-key set. Paired reports fail closed on missing, duplicate, or mismatched scene-time targets and report absolute means, raw deltas versus scalar control, oriented deltas, and paired bootstrap intervals.
5. Safe, Near, and Contact each receive compact presentation CSV/Markdown tables plus full-metric CSV tables. The scalar control is an explicit table row rather than only an implicit delta reference.
6. Critical-scene selection requires complete regime-critical metrics, an actual intervention, positive composite physical change, and no new overlap/offroad/re-contact for positive examples. Failure examples are also exported and cannot duplicate positives.
7. Video output uses unique scene-time filenames, common paired bounds, SDC trails, selection reason/continuous metrics, MP4 `veryfast` encoding when ffmpeg is available, GIF fallback, and complete JSON/CSV provenance indices.
8. Held-out test exploration requires an explicit contamination flag and writes a permanent disclosure that outputs are exploratory only and cannot be used for future checkpoint, threshold, or algorithm selection.

### Runtime implications

- Reusing the completed v48.34 checkpoints avoids repeating approximately 29–33 minutes of parallel adaptation wall time (about 62 GPU-minutes combined in the uploaded run).
- Removing online OC-MERO teacher labeling from progress-only closed loop avoids the previously observed dominant audit-label cost while preserving Waymax physical rollout metrics.
- Running the two OC-RAP variants and pairs of external methods concurrently uses the available two-GPU server without changing scene sets or metric definitions.
- Videos rerun only the auditable selected scene-time subset with render traces rather than rendering every exploratory rollout.

### Decision rule

- Repair/clean run `RC=30`: stop; do not analyze the algorithm or run ablations/shadow/test/stress.
- `RC=20`: the pipeline and certificate are valid; return the complete result for algorithm analysis. Optional same-target closed-loop/video outputs remain progress-only.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt` for formal downstream evaluation.

## v48.34 — BARRIER-CROSSFIT (2026-08-03)

### v48.33 result attribution

- The uploaded v48.33 main pipeline is a valid algorithmic rejection: `pipeline_exit_code=20`, `certificate_exit_code=20`, `certificate_executed=true`, `gate_evaluated=true`, `gate_passed=false`, `pipeline_valid=true`, and `test_roots_read=false`. The operational rejection remains `development_rule_fit_rejection`.
- Unified top-5 proposal support is not the blocker. Adaptation-dev/certificate contain 8/9 Near and 17/20 Contact proposal-contained safe opportunities, and the proposal-constrained oracle is feasible.
- Near improved on adaptation-dev but not on the scene-disjoint certificate. Precision Near candidate safe-positive AUC rose to approximately `0.919` and the legacy evidence-only proposal correlation rose from approximately `-0.011` to `+0.249`; the closest development rule selected 13 actions, 4 safe positives and 1 harmful action with safe recall `0.50` and mean teacher advantage `+0.151`. On certificate it selected 14 actions, 0 safe positives and 7 harmful actions with mean advantage `-0.263`. Balanced Near similarly selected 2 safe positives on development but 0 on certificate.
- Contact action identity remains unresolved. Balanced/Precision certificate candidate safe-positive AUC is approximately `0.570/0.557`, proposal evidence correlation is `-0.205/-0.115`, safe-positive selections are `1/1`, harmful selections are `17/19`, and mean selected teacher advantage is `-0.204/-0.159`.
- v48.33 therefore learned a development-local Near ordering signal but not a transferable action-level physical ordering. Contact remains close to candidate-level random discrimination and still assigns excessive evidence to apparently beneficial but physically unsafe actions.
- All eight uploaded v48.33 ablations are invalid for algorithm comparison. They exited pipeline `RC=30` before identity training because Stage-2 settings and inconsistent defaults were included in the Stage-1 factor-cache identity. No C/D/A/B performance attribution is made from those runs.
- The uploaded development shadow is exploratory and contains only eight paired scenes per variant/regime. Near produces approximately `+0.011 s` TTC-p05 and `+0.015 s` terminal-TTC changes but decreases bounded NUP by approximately `0.014`; Contact produces millimetre-scale clearance/free-space changes and small TTC-recovery changes while also decreasing bounded NUP. These results do not establish submission-level closed-loop superiority.

### Root cause

1. **Soft improvement did not cross executable safety boundaries.** Precision selected epoch 12 for a small improvement in threshold-free soft risk even though valid-safe admissions remained zero and the maximum invalid-admission rate remained one. Balanced training moved from one valid-safe admission at epoch 0 to zero while soft recall increased. The scalar checkpoint objective rewarded probability mass shifts that never produced a deployable action.
2. **Unsafe recovery evidence remained compensatory.** The v48.33 admission residual could still overcome an unfavourable learned safety slack. High raw recoverability evidence therefore remained able to dominate even when one supported physical component predicted boundary violation.
3. **Development-local scene shortcuts dominated action identity.** Near candidate AUC and development correlation improved, but certificate safe hits remained zero. The selector learned opportunity/context correlations concentrated in a few scenes rather than invariant candidate-vs-nominal causal differences.
4. **Contact representation is still underidentified.** Candidate safe-positive AUC remains near random and proposal correlation remains negative. Threshold calibration cannot repair a representation that assigns the wrong sign or relative order to action-level safety.
5. **Legacy diagnostics obscured the exact failure location.** v48.33 reported an evidence-only top-1 diagnostic that ignored eligibility. The Natural gate used the correct eligible-set policy, so RC=20 is valid, but the diagnostic could not distinguish an eligibility-head failure from a ranking-after-filter failure.

### Engineering and protocol corrections

1. **Stage-1 cache boundary is exact.** Factor-cache identity contains only Stage-1 inputs and hyperparameters. Stage-2 prior, boundary and checkpoint settings no longer invalidate a reusable factor checkpoint. Reuse validates source and copied checkpoint hashes and rewrites run-local metadata.
2. **Exact and legacy policy diagnostics coexist.** Calibration emits both evidence-only top-1 and exact `rank top-k -> eligibility -> evidence rerank -> one action or nominal` metrics, plus proposal candidate rows. This separates unsafe-filter errors from eligible-set ranking errors.
3. **Hard-policy checkpoint metadata is authoritative.** Best-checkpoint selection records the complete lexicographic key, actual validation loss and scalar audit metric separately; it no longer stores a tuple element as `best_val_loss`.
4. **No all-abstain preference.** Lexicographic ordering first minimizes regimes with zero valid-safe admissions, then maximizes total valid-safe admissions and cross-scene fold-min safe top-1 recall before minimizing invalid admission and regret.
5. **Cross-scene fold validation is mandatory.** Scene-fold minimum safe top-1 recall participates in checkpoint ordering, reducing selection of epochs that concentrate all apparent success in one or two development scenes.
6. **Exploratory data scope is fail-closed.** Adaptation-dev closed-loop can no longer silently default to held-out test roots. Held-out test inspection requires explicit authorization and writes a permanent contamination declaration. Exact target roots, WOMD sources and target-contract hashes are recorded.
7. **Ablation status is unambiguous.** Every task records both pipeline and certificate exit codes, failure stage and cache identity. An engineering `RC=30` cannot be interpreted as an algorithmic `RC=20`.
8. **Critical-scene videos are auditable.** Selection scores every paired scene and exports both positive and failure examples. Videos cannot be generated from an unpaired or cherry-picked scene list.

### v48.34 unified algorithm

1. **Barrier-gated safety slack.** Let `m_max(a)` be the maximum supported learned candidate-vs-nominal safety margin. A continuous safety gate `g(a)=sigmoid(-m_max/tau)` attenuates both raw benefit and the learnable admission residual, while a softplus barrier penalizes positive safety slack. Unsafe evidence can no longer be fully compensated by a large residual. The same equation is used for Safe, Near and Contact; no regime ID or case routing is introduced.
2. **Eligibility-boundary continuation.** In addition to the eligible-set KL, safe-positive candidates are pushed beyond opportunity, harm and admission boundaries by a registered margin; harmful candidates are pushed below harm/admission boundaries; dead candidates receive a weak nominal preference. This directly optimizes executable transitions rather than only soft probabilities.
3. **Hard-first lexicographic checkpointing.** Checkpoints are selected by valid-safe admissions and cross-scene safe top-1 coverage before invalid-admission rate, safe regret and soft population risk. Soft risk is only a tie-breaker once executable behaviour is comparable.
4. **Two-stage natural-population training.** Stage 1 learns raw benefit and signed physical margins with no replacement. Stage 2 jointly trains benefit/opportunity, supported harm components and admission under barrier-gated eligible-set supervision. Adaptive teacher-gap margin and Stage 3 remain disabled.
5. **Unified top-5 and independent measured veto are retained.** Proposal generation, measured hard veto, exact eligibility, bounded one-action policy, scene-disjoint certificate and sealed test/stress roots are unchanged.

### Exploratory closed-loop and external baselines

- External baseline results uploaded with v48.33 are not directly numerically comparable to OC-RAP because they were not evaluated on the same target scene set; the observed Near/Contact scene overlap with the existing OC-RAP shadow is zero.
- v48.34 provides a same-target paired runner for OC-RAP, scalar control and external baselines. It validates identical scene IDs before paired bootstrap reporting.
- After `RC=20`, adaptation-dev exploratory closed-loop is allowed only with an explicit diagnostic flag and cannot be used to tune thresholds/checkpoints. Held-out test evaluation additionally requires an explicit contamination flag and permanently disqualifies those scenes from future model selection.
- The critical-scene pipeline records render traces and produces side-by-side Control/OC-RAP MP4 and GIF files for both positive and failure cases. These are toy examples, not substitutes for aggregate paired metrics.

### Required next decision

- Run only the v48.34 main experiment first.
- `RC=30`: stop and inspect the structured failure; do not run ablation, shadow, test, stress or exploratory comparison.
- `RC=20`: run the authorized v48.34 ablations and adaptation-dev shadow. Exploratory same-target baseline comparison/video generation is optional and must retain its disclosure. Do not use held-out results for v48.35 design.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt` for formal test/stress. Exploratory tooling may still be used for visualization, but formal and exploratory outputs must remain separate.
- No claim is made in advance that v48.34 will pass. The decisive evidence is whether valid-safe admissions become nonzero in both target regimes, Near certificate safe hits replace development-only gains, and Contact exact-eligible harmful switches/negative advantage fall without regime routing.

## v48.33 — ELIGIBLE-SET-POLICY (2026-08-02)

### v48.32.1 result attribution

- The uploaded v48.32.1 main pipeline completed adaptation and the dedicated certificate controller. Its observed controller result is `RC=20`, with `certificate_executed=true`, `gate_evaluated=true`, `gate_passed=false`, and `test_roots_read=false`. The operational rejection label is `development_rule_fit_rejection`.
- The run is not an RC=30 engineering crash. It provides valid evidence that the frozen v48.32.1 selector did not satisfy the Natural gate under the rule that was actually fitted.
- A protocol audit found that the development-rule command substituted the looser verification thresholds for the preregistered fit thresholds. Therefore the rejection remains real, but the reported fit deficits, proposal-oracle feasibility, and the unconditional `pipeline_valid=true` claim are not formal evidence under the declared preregistration.
- The intended strict fit thresholds are Near `min_selected=10`, precision LCB `>=0.50`, harmful-group UCB `<=0.12`, selected-harm UCB `<=0.22`; Contact `16`, `0.50`, `0.14`, `0.22`. v48.32.1 instead fitted with Near `8`, `0.40`, `0.14`, `0.25` and Contact `10`, `0.40`, `0.16`, `0.25`.
- Under the rule actually used, the best Precision Near adaptation-dev frontier selected 10 actions with 3 safe positives, 1 harmful action, precision LCB90 `0.154`, safe recall `0.375`, mean teacher advantage `+0.191`, and macro share `0.60`. On certificate it selected 8 actions with 0 safe positives, 4 harmful actions, mean advantage `-0.298`, and macro share `0.875`.
- Balanced Near retained strong candidate discrimination (`candidate_safe_positive_auc≈0.831`) but proposal evidence correlation was approximately `-0.014`; certificate selected no action. Precision Near had `candidate_safe_positive_auc≈0.796`, proposal evidence correlation approximately `-0.011`, and also no certificate safe-positive hit.
- Contact remains weaker at the candidate level and worse at proposal identity. Balanced/Precision certificate candidate safe-positive AUC was approximately `0.581/0.553`, proposal evidence correlation approximately `-0.136/-0.167`, selected safe positives were `0/0`, harmful selections were `5/15`, and mean selected teacher advantage was `-0.134/-0.230`.
- The uploaded v48.32 ablations all returned RC=20. Joint detached updates worsened Balanced Contact relative to admission-only; coupling reduced that degradation but suppressed Near activity. The adaptive teacher-gap margin produced no measurable difference from the fixed-margin coupled configuration. Precision A/B/C/D were effectively identical, indicating repeated epoch-0/no-op selection. Stage 3 provided no demonstrated benefit and is removed from the default main path.

### Root cause

1. **Scene opportunity is learned, action identity is not.** Near candidate AUC is useful, yet candidate-vs-nominal evidence ordering inside the frozen proposal is approximately uncorrelated with teacher utility. Contact candidate safety discrimination is close to random and proposal correlation is negative.
2. **Training and deployment selected actions in different order.** v48.32.1 checkpoint metrics selected the largest evidence score inside rank top-k and only then checked opportunity/harm. Calibration and runtime first filter proposal members by opportunity/harm and then rerank the eligible set. A deployable runner-up could therefore receive no checkpoint credit.
3. **Soft early stopping also ignored eligibility.** The threshold-free population risk assigned categorical mass from evidence alone, rewarding high-evidence candidates that the deployed policy would reject.
4. **Top-3 is structurally insufficient for the strict Near fit contract.** Adaptation-dev contains only seven top-3 proposal-contained Near safe opportunities. With strict `min_selected=10`, even an optimistic 7/10 precision has one-sided 90% Wilson LCB below 0.50. Unified top-5 contains all eight Near safe opportunities and makes the strict optimistic support bound feasible. Contact support is already saturated by top-5.
5. **Additional admission-only calibration cannot repair representation.** The former Stage 3 repeatedly selected epoch 0 and did not change certificate outcomes.

### Engineering and protocol corrections

1. **Preregistered fit thresholds are passed exactly.** Verification thresholds are no longer reused as fit thresholds.
2. **Fail closed before certificate access.** The new metric/calibration identity checker validates exact dev group counts, proposal safe-opportunity counts, strict fit thresholds, proposal top-k, evidence-rerank flag, selection order, and strict proposal-oracle feasibility.
3. **Exact hard policy order in checkpoint metrics.** Validation now executes `rank top-k -> opportunity/harm filter -> evidence rerank -> one action or nominal`.
4. **Exact soft policy order in early stopping.** Soft categorical checkpoint mass includes differentiable opportunity/harm eligibility before evidence, matching the new training objective and runtime ordering.
5. **Unified top-5 contract.** Top-k is fixed to five across factor training, identity training, checkpoint metrics, dev rule fitting, certificate verification, runtime policy metadata, cache identity, and audit tools. No regime-specific top-k is introduced.
6. **Selection semantics are explicit metadata.** `SELECTION_SEMANTICS=rank_topk_then_filter_then_evidence_rerank` is checked in `POLICY_CONTRACT.env` and `GATE_SPEC.json`.
7. **Ablations are authorization-gated.** The v48.33 eight-task suite runs only after a valid main pipeline with certificate RC=20. It reuses the exact top-5 Stage-1 factors and never reads test/stress roots.
8. **Ineffective Stage 3 is disabled by default.** The identity checkpoint is copied atomically into the final run with stage-transfer integrity checks.
9. **Known safety contracts are retained.** Natural no-replacement training, exact physical eligibility, support reliability, independent measured hard veto, bounded admission, and test-root sealing remain unchanged.

### v48.33 unified algorithm

1. **Eligible-set policy objective.** Inside the frozen unified top-5 proposal, the student categorical score is admission evidence plus continuous log soft-eligibility from opportunity and harm heads; nominal is an explicit abstention class. The teacher distribution is continuous safe utility. This gives deployable runner-up actions gradient instead of rewarding an ineligible evidence top-1.
2. **Multi-head identity coupling.** Stage 2 jointly updates compact benefit, supported physical-margin harm, and admission calibrators. The eligible-set KL propagates finite gradients through all three heads. No Safe/Near/Contact ID or case-specific strategy is used.
3. **Fixed hardest-negative margin.** The adaptive teacher-gap scale is disabled because C/D ablations were indistinguishable. Hardest-negative supervision remains as a simpler proposal-local separation term.
4. **Two-stage default.** Stage 1 learns raw benefit and signed continuous physical margins on the natural population. Stage 2 learns the exact eligible-set one-action policy. Admission-only Stage 3 is disabled unless a future ablation provides evidence for it.
5. **Strict checkpoint contract.** Loss, soft early stopping, hard validation counters, dev threshold fitting, certificate verification, and runtime now share the same proposal/filter/rerank semantics.

### Required next decision

- Run only the v48.33 main experiment first.
- `RC=30`: stop; no algorithm conclusion, ablation, shadow, test or stress.
- `RC=20`: the corrected strict Natural gate was evaluated; run the authorized v48.33 ablations and adaptation-dev physical shadow only.
- `RC=0`: execute only the generated `NEXT_COMMANDS.txt`.
- No claim is made in advance that v48.33 will pass. The decisive evidence is whether top-5 plus eligible-set training converts Near candidate AUC into certificate safe top-1 hits and whether Contact harmful selections/negative advantage fall materially without regime routing.

## v48.32.1 — RC30-INTEGRITY-HOTFIX (2026-08-02)

### v48.32 result attribution

- The uploaded v48.32 main controller returned a genuine pipeline `RC=30`: Balanced and Precision adaptation both exited 1, `failure_stage=adaptation`, `certificate_exit_code=null`, `gate_evaluated=false`, `pipeline_valid=false`, and `test_roots_read=false`.
- This run is not `development_rule_fit_rejection` and provides no valid evidence about the v48.32 algorithm. No ablation, physical shadow, test or stress conclusion is authorized.
- Both variants completed Stage-1 factor training, then failed during Stage-2 epoch-0 validation with the same `IndexError: too many indices for tensor of dimension 0` at the factorized component-veto call.
- The deterministic root cause is Python variable shadowing. The candidate-level vector `teacher_gap` was overwritten inside the group loop by the scalar adaptive teacher-utility gap. After the first safe-positive group, a later group attempted `teacher_gap[recs]` on a zero-dimensional scalar.

### Engineering hotfixes

1. **Separate tensor/scalar identities.** The population vector is `teacher_gap_vector`; the per-group scalar is `adaptive_teacher_gap`. The algorithm and margin formula are unchanged.
2. **Exact multi-group preflight.** Before index construction or GPU training, a two-group synthetic contract exercises factorized harm, adaptive hardest-negative, forward, backward and finite-gradient checks.
3. **Static group-loop shadowing guard.** The preflight inspects the exact loss function AST and rejects assignments that overwrite outer tensors inside the scene-time group loop.
4. **Strict shape contract.** Main training no longer uses `n=min(sizes)` to silently truncate mismatched model, teacher or metadata tensors. It fails closed and requires exactly one nominal per group.
5. **Deterministic CUDA contract.** All v48.32.1 entry points set `CUBLAS_WORKSPACE_CONFIG=:4096:8`; deterministic CUDA LCVaR avoids the nondeterministic `cumsum` path by using an exact lower-triangular prefix operator.
6. **Stage-aware failures.** Variant failures record the active stage, shell command and return code. The controller adds a parsed exception type, message, Python frame and bounded log tail.
7. **Exact Stage-1 materialization.** Reuse verifies source checkpoint SHA against both completion files, atomically copies the stage, verifies the copied SHA, and rewrites completion metadata to the new checkpoint path.
8. **Variant-specific reuse.** Balanced and Precision may independently reuse their successfully completed v48.32 factor stages after source/index/support/hyperparameter contract verification.
9. **Correct certificate semantics.** `certificate_executed`, `gate_evaluated`, nullable `certificate_exit_code`, and `pipeline_exit_code` are recorded separately. Certificate artifact/protocol RC=30 no longer claims that the Natural gate was evaluated.
10. **Explicit authorization state.** RC=0 requires `NEXT_COMMANDS.txt` plus generated status; RC=20/30 requires blocked status. Manual authorization remains prohibited.
11. **Complete certificate dependency closure.** The v48.32.1 validation/calibration population-identity checker is packaged under the exact name used by the certificate controller and covered by a release dependency audit.

### Runtime effect

- The failed v48.32 run already spent 1175.16 seconds on Balanced Stage-1 and 1147.04 seconds on Precision Stage-1.
- If the original server run still contains both factor checkpoints, v48.32.1 can resume from Stage-2 after exact cache verification, avoiding approximately 38.7 GPU-minutes of repeated factor training and roughly 19.6 minutes of concurrent wall time.
- Teacher indexes may be copied to the new output directory and are reused only after exact dataset/label contract validation.

### Decision rules

- `RC=30`: no algorithm conclusion and no ablation/shadow/test/stress. Inspect the multigroup preflight, pipeline failure, variant stage marker and exception signature.
- `RC=20`: pipeline and certificate are valid; only then is a Natural-gate rejection established.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt`.

### Local validation boundary

- Exact two-group loss preflight: passed with finite loss and non-zero admission gradient.
- `PYTHONPATH="$PWD/src" pytest -q`: 294 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- Shell syntax and new-script dependency closure: passed.
- Real WOMD/Waymax, the server-side factor `.pt` files, and two A30s are unavailable locally; no gate result is claimed in advance.

## v48.32 — OC-TRAC-IDENTITY-UTILITY-BRIDGE (2026-08-02)

### v48.31 result attribution

- The uploaded v48.31 main controller did **not** return a valid Natural-gate result. Balanced adaptation exited 0, Precision exited 31, the controller normalized this to `RC=30`, `failure_stage=adaptation`, `raw_certificate_exit_code=null`, `gate_evaluated=false`, and `pipeline_valid=false`. The main run is therefore an engineering failure before certificate access, not `development_rule_fit_rejection` and not RC=0.
- `NEXT_COMMANDS.txt` was absent because v48.31 emitted it only after a valid certificate RC=0. The controller exited before certificate evaluation. The underlying Precision failure was a false stage-transfer rejection: Stage 3 legally selected its epoch-0 input checkpoint, changed zero allowed and zero disallowed parameters, but the checker treated the no-op fail-safe as corruption.
- The eight-task ablation suite was incomplete. A/B Balanced and Precision copied a no-joint checkpoint without `TRAINING_COMPLETE.json` and `EVIDENCE_CORRECTION_COMPLETE.json`; C/D Precision hit the same epoch-0 false rejection. Only C/D Balanced reached certificate, and both returned a valid `RC=20` with `development_rule_fit_rejection`.
- The v48.31 adaptation-dev physical shadow did not produce physical evidence. Both variants exited 2 because the invalid main pipeline never produced calibration gamma. No v48.31 TTC, clearance, free-space, re-contact, stable-stop or intervention conclusion is claimed.
- Valid C/D Balanced evidence confirms that proposal support is not the blocker: the certificate contains 9 Near and 20 Contact proposal-contained safe-positive groups. The learned selector still chooses no safe positives.
- Near retains strong candidate diagnostics but weak deployable identity. D Balanced development safe-positive AUC is approximately 0.902 and proposal safe-top-1 AUC approximately 0.917, yet the closest rule selects only 1 safe positive in 4 actions, precision LCB90 approximately 0.078 and recall 0.125. On certificate it abstains entirely; C selects 3 actions, 0 safe positives and 2 harmful actions.
- Contact remains substantially below submission maturity. D Balanced certificate safe-positive AUC is approximately 0.580, group top-1 correlation approximately -0.170, safe-top-1 AUC approximately 0.450, and 17 selections contain 0 safe positives and 5 harmful actions with mean teacher advantage approximately -0.141.
- v48.31 support reliability is not a side-effect-free performance gain. It suppresses Near harmful selection by abstaining, but Contact harmful selections increase from 3 to 5 and mean selected advantage becomes more negative. It remains a support-safety contract, not a proven ranking module.

### Confirmed engineering corrections

1. **Correct no-op checkpoint semantics.** Stage-2 or Stage-3 selection of the evaluated initial checkpoint is valid when no frozen parameter changed and no key disappeared. It no longer produces a false RC=30.
2. **Complete no-final metadata.** Disabled-final ablations copy checkpoint, architecture, policy, training-complete and evidence-complete artifacts.
3. **Explicit NEXT state.** RC=0 writes `NEXT_COMMANDS.txt` plus generated status; RC=20/30 writes `NEXT_COMMANDS_BLOCKED.json` plus status. The controller verifies RC/file consistency.
4. **Unambiguous controller status.** `pipeline_exit_code` is separate from nullable `certificate_exit_code`; a pre-certificate failure never masquerades as a certificate return.
5. **Run-local status hygiene.** Every controller start clears stale adaptation, gate, calibration, NEXT and completion markers. Successful variant reruns delete stale failure markers.
6. **Complete task failure materialization.** Ablation root failures and every task-stage failure write structured JSON and log tails instead of disappearing under `set -e`.
7. **Fail-closed shadow preflight.** Waymax is not launched unless pipeline, certificate status, checkpoint, gamma, target, provenance, runtime and paired-scene contracts are valid.
8. **Exact factor-cache identity.** Cache reuse requires matching source-checkpoint SHA, train/dev index SHAs, support-contract SHA, variant and all relevant factor hyperparameters. Paths may differ only when file contents are identical.
9. **Audited teacher-index reuse.** Train and adaptation-dev indexes are reused only after exact dataset/label contract checks; otherwise they are rebuilt automatically.
10. **Correct script dependency closure.** Every referenced v48.32 Python and shell tool exists and is covered by shell-to-tool dependency audit.

### Algorithm diagnosis

- v48.31 Stage 2 trained only the admission calibrator while freezing compact benefit and component-harm calibrators. The deployed safety-slack prior also detached benefit and component logits. Safe-utility, listwise and hardest-negative gradients therefore could not correct the action-identity errors that caused a harmful action to outrank a safe action within the same proposal.
- This explains the observed candidate-AUC/top-1 gap: candidate classification can improve while exact proposal top-1 correlation remains near zero or negative and valid-safe admission collapses to zero.
- Contact additionally has weak candidate representation, so threshold fitting or additional binary admission loss cannot repair it. The next change must couple group-local safe utility to benefit and supported physical margins without regime routing.

### v48.32 unified algorithm

1. **One continuous selector across all regimes.** No Safe/Near/Contact ID or regime-specific branch is added. Reporting strata remain external to the model.
2. **Support-weighted safety slack.** Keep the global nominal-relative coordinates and independent measured hard veto. The deployed utility remains `B(a) - lambda * relu(max_k r_k m_k(a))`.
3. **Proposal-local identity stage.** Stage 2 jointly trains the compact benefit, component-harm and admission calibrators by default.
4. **Deployment-utility gradient bridge.** Stage-2 admission prior can be non-detached, allowing safe-utility/listwise/hard-negative loss to update benefit and supported component margins. This changes gradient flow only, not inference semantics.
5. **Adaptive teacher-gap margin.** Hard-negative separation is `base_margin + scale * clamp(teacher_safe_utility_gap, 0, 0.25)`; no-safe groups receive a continuous no-op-depth margin. The same formula is used across all regimes.
6. **Admission-only final calibration.** Stage 3 adjusts only the bounded admission residual with low learning rate; epoch zero remains a legal fail-safe.
7. **Retain natural population and exact checkpoint contract.** All stages use natural groups without replacement, exact executable eligibility and safe-top-1 checkpoint barriers.

### v48.32 ablations

1. `A_admission_only_detached_fixed_margin`
2. `B_joint_identity_detached_fixed_margin`
3. `C_joint_identity_coupled_fixed_margin`
4. `D_full_identity_utility_bridge`

B>A tests compact joint identity learning; C>B tests deployment-utility gradient coupling; D>C tests the adaptive continuous teacher-gap margin. Support reliability is retained in all groups because it is an already-required unsupported-coordinate safety contract.

### Runtime changes

- v48.31 repeated Stage-1 factor training eight times in ablations, consuming approximately 11,302 seconds (3.14 hours) and about 65.3% of effective ablation training time.
- Standalone v48.32 ablations train only one factor stage per variant, saving approximately 2.38 hours.
- The recommended post-main run reuses the two exact-contract main factor stages, eliminating all additional factor training and saving approximately 3.14 hours relative to v48.31.
- Certificate populations and Waymax rollout horizons are not reduced. Speed is obtained only by exact-contract reuse and index caching.

### Decision rules

- `RC=0`: `NEXT_COMMANDS.txt` and generated status must both exist; execute only those commands.
- `RC=20`: pipeline and certificate are valid but the Natural gate rejected the selector. Do not read test/stress; run v48.32 ablations and adaptation-dev physical shadow.
- `RC=30`: make no algorithm conclusion. Inspect `PIPELINE_FAILED.json`, `NEXT_COMMANDS_BLOCKED.json` and the named contract failure.
- Do not lower the registered gate, expand to top-8, split regimes into separate policies or manually create authorization.

### Local validation boundary

- `PYTHONPATH="$PWD/src" pytest -q`: 285 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- v48.32 shell-to-tool dependency audit: passed with no missing references.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.32 gate or physical result is claimed in advance.

## v48.31 — OC-TRAC-CONTRACT-SLACK-RANK (2026-08-02)

### v48.30 result attribution

- The uploaded v48.30 controller produced a valid `RC=20`: the pipeline completed, the Natural gate was evaluated, test/stress roots were not read, and all four branches were rejected during adaptation-development rule fitting.
- Proposal support is not the blocker. The frozen top-3 proposal contains 9 Near and 20 Contact safe-positive certificate groups, with optimistic oracle precision LCBs of approximately 0.846 and 0.924.
- Learned action identity is the blocker. Near proposal safe-positive AUC is approximately 0.735/0.784 for Balanced/Precision, yet proposal evidence top-1 correlation is approximately +0.021/-0.023 and both certificate branches select zero safe positives. Contact correlation is approximately -0.118/-0.116 and both certificate branches also select zero safe positives.
- Precision Near retains the strongest local signal: the closest development rule selects 3 safe positives in 7 interventions, safe recall 0.375 and mean teacher advantage +0.273. It does not generalize to the certificate, where 3 of 6 selections are harmful and mean advantage is -0.294.
- Contact has not established safe admission. Balanced/Precision select 17/28 certificate actions, zero safe positives, 6/16 harmful actions and negative mean teacher advantage.
- The Safe recovery threshold calibration is valid, but no scene-disjoint Safe policy certificate is registered. Safe therefore still requires non-empty paired non-inferiority evidence.
- The uploaded v48.30 development-shadow controller did not run simulation. It stopped before Waymax because `audit_v48_30_shadow_provenance.py` was missing; two additional referenced v48.30 checker files were also absent. No v48.30 closed-loop physical conclusion is claimed from this package.

### Confirmed engineering corrections

1. **Exact executable validation population.** Training validation now applies the same eligibility contract as calibration: supported macro, feasible candidate, measured hard rule no greater than the registered maximum, and candidate-vs-nominal prefix deviation above the registered minimum.
2. **Correct all-abstain semantics.** When the safe contract is available, abstention is defined by zero valid safe admissions. Harmful or invalid switches can no longer make a checkpoint appear executable.
3. **Safe top-1 checkpoint barrier.** `direct_contract_safe_rank_risk` lexicographically penalizes any reporting stratum with zero proposal-contained safe top-1 hits, safe top-1 recall shortfall, invalid admission, valid-safe abstention and safe top-1 regret.
4. **Natural population in every optimization stage.** Factor training, admission training and joint refinement all use natural, without-replacement scene-time groups. v48.30 applied this repair only to the admission stage while freezing the factor heads learned under replacement sampling.
5. **Metric/calibration population audit.** Before certificate verification, the selected checkpoint's exact-eligible group counts and proposal-contained safe-opportunity counts must exactly match adaptation-dev calibration.
6. **Fail-closed model/inference audit.** Component count, scale, frontier prior, bounded admission, slack temperature/penalty and support reliability must be identical in checkpoint training and inference.
7. **Stage-transfer audit.** Stage 2 must not alter the frozen benefit/harm factors. Stage 3 may change only the three registered evidence calibrators.
8. **Repaired development shadow.** Add v48.31 provenance, runtime, physical-support and regime-target checkers; require non-empty paired scenes and valid metric semantics; all referenced tools are covered by a focused regression test.

### Algorithm corrections

1. **Global support-conditioned continuous slack.** Keep one nominal-relative continuous physical representation across Safe, Near and Contact, with no regime ID or regime-specific policy. Each learned component is shrunk toward its semantic non-harm prior according to global data support.
2. **Do not learn unsupported veto coordinates.** The current training index supports DRS, deployability and gap margins, but `harm_proxy` is constant and the exact-eligible learned hard-rule coordinate has no positive examples. The default support contract is therefore `1,1,1,0,0`. The independent measured hard veto remains active and uncompensated.
3. **Support-weighted factor supervision.** Component BCE and signed-margin regression are weighted by the same global reliability used at inference, eliminating train/runtime disagreement.
4. **Three-stage optimization.** Stage 1 learns raw benefit and supported physical factors; Stage 2 learns bounded admission with proposal-level safe utility, listwise ranking and hardest negatives; Stage 3 performs low-learning-rate joint refinement of benefit, harm and admission calibrators only.
5. **Action identity over scene classification.** Candidate AUC is retained only as a diagnostic. Checkpoint selection is driven by proposal-contained safe top-1 support, safe admission precision/recall, harmful mass and regret.
6. **Preserve the frozen top-3 proposal.** Oracle support remains feasible. Expanding to top-8 would increase calibration burden and harmful exposure without addressing the observed identity failure.
7. **Preserve the unified paper semantics.** Near/Contact are reporting strata only. The appendix's legacy regime-conditioned protective-certificate text should be replaced before submission with the support-conditioned unified margin contract.

### v48.31 ablations

Four waves, at most two concurrent tasks, one Balanced job on GPU0 and one Precision job on GPU1:

1. `A_contract_natural_no_reliability_no_joint`
2. `B_add_support_reliability_no_joint`
3. `C_add_joint_refinement_no_reliability`
4. `D_full_contract_slack_rank`

B>A isolates global support reliability. C>A isolates low-rate joint refinement. D>max(B,C) demonstrates complementarity. Development improvement without certificate improvement indicates nuisance/macro generalization failure; offline improvement without paired physical improvement indicates teacher/closed-loop mismatch.

### Decision rules

- `RC=0`: execute only the generated authorization commands; then run held-out stress and publication checks.
- `RC=20`: do not read test/stress. Run the repaired adaptation-dev shadow and four-wave ablations. First inspect exact-contract audits, safe top-1 hits, harmful selection and macro concentration.
- `RC=30`: make no algorithm conclusion. Inspect protocol, support, model/inference, stage-transfer, metric/calibration and artifact failures.
- Do not lower the registered Natural gate after observing certificate results.

### Local validation boundary

- Full pytest suite: passed after adding the v48.31 focused tests.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- v48.31 shell-to-tool reference audit: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.31 gate or closed-loop result is claimed in advance.

## v48.27 — OC-TRAC-FACTOR-PHYSICS-BRIDGE (2026-08-01)

### v48.26 result attribution

- The v48.26 controller returned a valid `RC=20`: `pipeline_valid=true`, `gate_evaluated=true`, `gate_passed=false`, and `test_roots_read=false`. This is not a pipeline failure.
- Proposal-constrained top-3 oracle support is feasible on both adaptation-dev and the complete certificate population. The certificate contains 9 Near and 20 Contact proposal-contained safe-positive groups; oracle precision LCB is approximately 0.846 and 0.924. The current failure is therefore not candidate support or a mathematically impossible certificate contract.
- All four adaptation-dev threshold-fit jobs failed to produce a rule satisfying the joint selected-count, safe-positive precision/recall and harmful-selection constraints. v48.26 subsequently verified a `diagnostic_fit_rule` with `source_rule_satisfied_dev_constraints=false`. The correct rejection class is `development_rule_fit_rejection`, not a generic learned-gate failure.
- The v48.26 development shadow did not produce physical evidence. Each Near/Contact audit loaded 16 offline targets but scanned only the first 900 `validation_interactive` scenarios, matched zero targets and emitted zero-scene JSON. Paired comparison then failed. Existing shadow zeros/nulls cannot be interpreted as no collision, no re-contact or good intervention behavior.
- Near Precision retains useful local evidence (safe-positive AUC about 0.828, high-opportunity conditional-harm AUC about 0.842, false-switch about 0.085), but selects zero safe-positive certificate actions. Near Balanced selects 12 actions with zero safe positives and nine harmful selections.
- Contact does not establish safe admission. Balanced/Precision select 50/46 actions, only one safe positive each, approximately 44%/46% harmful, recall 0.05 and negative mean selected teacher advantage.
- The A/B ablations contain a small but important viable region: Balanced Near selects 3 safe positives among 5 actions with empirical precision 0.60 and mean teacher advantage +0.251. C/D safe-utility/full objectives destroy that region, supporting an objective-scale and gradient-interference diagnosis rather than a no-signal diagnosis.

### Engineering corrections

1. Remove the fixed development-shadow scan cap. `DEV_SHADOW_RAW_MAX_SCENARIOS=0` now scans the complete raw source for sparse target IDs.
2. Add `DEV_SHADOW_WOMD_SOURCE` so the exact raw WOMD source used by the offline bucket can be supplied explicitly.
3. Canonicalize target and raw scenario IDs by preferring `original_scenario_id` and removing only the operational `__wx########` loader suffix.
4. Set `closed_loop.require_bucket_targets=true`; fail immediately when the target manifest is empty or no target matches after the scan.
5. Mark empty closed-loop aggregates with `metrics_valid=false` and `empty_reason=no_closed_loop_scenes`; unavailable physical metrics are `null`, never evidence-valued zero.
6. Require non-empty paired scenes in the shadow controller and comparator. Add a repair-only script that re-runs existing v48.26 checkpoints with the corrected runner without retraining.
7. Distinguish `development_rule_fit_rejection`, `certificate_verification_rejection`, `structural_support_infeasible`, and engineering/artifact failure.
8. Rename sampler diagnostics semantically: retain legacy `num_safe_positive` for compatibility, add `legacy_root_safe_sample_count` and the gate-relevant `safe_positive_group_count`.
9. Persist and fail-closed check the number of component-harm heads in training and inference checkpoints.

### Algorithm corrections

1. **Five-factor non-compensatory harm representation.** Match the Natural-gate harmful label with five explicit heads: DRS, deployability, oracle-to-deployable gap, hard-rule violation and harm proxy. v48.26 predicted only the first three, so two gate vetoes were unrepresentable.
2. **Separate raw benefit from safe admission.** The opportunity head learns continuous raw benefit. Five harm heads learn veto factors. The admission head alone learns safe utility. Gate positives remain proposal-contained safe-beneficial actions.
3. **Execution-exact objective scale.** Regression, listwise ranking and frontier contrast all use the deployed score `sigmoid(admission_logit)-0.5`; no auxiliary objective compares unbounded logits with a bounded teacher.
4. **Two-stage factor/admission training.** Stage 1 trains only raw-benefit ranking and five harm factors. It disables admission, setwise admission and selective-risk/coverage gradients. Stage 2 freezes all factors and trains only a bounded admission residual with deployment-exact safe utility and categorical nominal-plus-top-k supervision.
5. **Bounded admission after factor repair.** v48.26 development checkpoints had invalid-admission rates of approximately 0.83–0.94. v48.27 returns to an identity-preserving bounded residual after the factor heads have been trained separately.
6. Keep the frozen top-3 proposal and disabled legacy Noisy-OR. Oracle support does not justify expanding proposal width.

### Physical-diagnostic policy

- The Near metric set covers clearance/TTC minima and terminal values, exposure duration/episodes/longest runs, deficit AUC, recovery gain, time to the dangerous point, collision/offroad, intervention bursts, route progression, acceleration, jerk and yaw rate.
- The Contact set covers secondary/re-contact events and episodes, overlap duration/longest run, free-space and clearance-deficit AUC, terminal/peak clearance, sustained escape and time-to-escape, stable-stop quality and time-to-quality-stop, plus offroad/dynamics/intervention burden.
- v48.26 did not execute these metrics on any scene, so their numerical correctness is not inferred from that run. v48.27 adds non-empty/fail-closed checks and focused unit tests. Empirical validation requires the repaired development shadow.
- Existing teacher indices do not contain candidate-level temporal physical labels. Do not fabricate them. If offline safe ranking improves while the repaired shadow does not, the next preregistered version should build candidate-level physical teacher rollouts and a separate temporal-recovery auxiliary head.

### Required v48.27 experiment and ablations

Main experiment: `scripts/run_v48_27_factor_physics_dedicated.sh`, Balanced on GPU0 and Precision on GPU1.

Before retraining, `scripts/repair_v48_26_dev_shadow_with_v48_27.sh` may re-run the existing v48.26 development shadow with complete scanning and canonical IDs.

Four ablation waves, one task per A30:

1. `A_three_factor_joint` — legacy three-factor joint training.
2. `B_five_factor_joint` — add hard-rule and harm-proxy factors.
3. `C_five_factor_two_stage_regression` — five factors, two stages and exact safe-utility regression.
4. `D_full_factor_physics_bridge` — C plus exact listwise/frontier objectives and corrected physical diagnostics.

### Decision rules

- `RC=0`: only the automatically generated `NEXT_COMMANDS.txt` authorizes held-out stress.
- `RC=20`: do not read test/stress. Run non-empty adaptation-dev shadow and the four-wave ablations. First identify development-rule fit versus certificate generalization failure.
- `RC=30`: no algorithm conclusion. Inspect pipeline, model-contract, index, checkpoint and artifact failures.
- B>A identifies missing harm representation; C>B identifies joint-gradient interference; D>C identifies value from exact listwise/frontier supervision.
- Offline improvement without physical-shadow improvement identifies teacher/closed-loop mismatch and motivates preregistered temporal physical supervision.
- Do not lower the registered Natural gate, manually create authorization, or call the repeatedly inspected certificate a final untouched paper certificate.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 242 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.27 gate or closed-loop result is claimed in advance.

## v48.26 — OC-TRAC-EXECUTION-PHYSICS-BRIDGE (2026-07-31)

### v48.25 result attribution

- The uploaded v48.25 controller returned `RC=30`, but both Balanced and Precision adaptation jobs completed successfully and produced checkpoints. All four Near/Contact certificate workers also loaded and scored their full populations (Near: 2,412 samples / approximately 305 groups; Contact: 6,929 samples / approximately 780 groups). The first actual failure occurred only when `tools/calibrate_policy_risk_v48.py` serialized `vars(args)`: `args.frozen_rule_json` is a `pathlib.PosixPath`, causing `TypeError: Object of type PosixPath is not JSON serializable`. Consequently `gate_evaluated=false`; v48.25 has no Natural-gate result and cannot be called an algorithmic rejection or regression.
- A second silent engineering defect would have invalidated the certificate even after fixing JSON. Training forwarded `direct_recovery_evidence_frontier`, `direct_recovery_evidence_component_prior_logit`, and `direct_recovery_evidence_admission_bounded`, but checkpoint inference omitted all three. The same checkpoint was therefore instantiated as different algorithms in training validation and certificate inference.
- The v48.25 checkpoint metric did not implement the Natural-gate opportunity contract. Its denominator used raw-positive groups in the full candidate set rather than proposal-contained safe-positive groups, and an admitted harmful action could still count as a positive hit. This could select an epoch that improves raw-benefit admission while failing safe admission.
- The safe-utility auxiliary target was also scale-inconsistent: training used `tanh(admission_delta/2)`, whereas runtime executes `sigmoid(admission_delta)-0.5`; the former is exactly twice the latter.
- v48.25 adaptation-dev trajectories do show a limited signal rather than universal abstention. Balanced and Precision reach Contact raw admission rates of approximately 0.034 and 0.048, and Balanced Contact evidence positive top-1 accuracy temporarily rises from approximately 0.321 to 0.500 while regret falls from approximately 0.265 to 0.204. However the flawed checkpoint metric selects epoch 0 for Balanced, Near positive admission remains zero, and the old metric does not require non-harmful selection. These observations are diagnostic only and do not establish safe admission.

### Certificate decision

- The certificate layer is retained. Its purpose is to verify a deterministic, adaptation-dev-frozen selector on an independent scene population before any held-out test/stress access; deleting it would hide unsupported behavior rather than improve the planner.
- v48.26 keeps the external-rule full-verification protocol: thresholds are fit only on `evidence_adapt_dev`, the complete rule and SHA256 are frozen, and the full `certificate_pool` is verification-only. Certificate labels never modify the rule.
- Valid structural-support or learned-selector rejection maps to controller `RC=20`. Empty data, corrupt checkpoint/artifact, split/index/protocol mismatch, inference-contract mismatch, or runtime exception maps to `RC=30`.
- `calibration.exact_split_ids=true` prevents the historical `calibration -> {calibration, certificate_pool}` alias from allowing accidental cross-role reads. Safe calibration, adaptation-dev fitting and certificate verification each consume exactly one registered split ID.
- A diagnostic dev rule may still be emitted when no dev rule satisfies constraints, but provenance records `source_rule_satisfied_dev_constraints=false`; it cannot be represented as a successfully constrained development selector.
- Safe currently has no independent scene-disjoint policy Natural-gate population. v48.26 writes `SAFE_REGIME_STATUS.json` and treats Safe through standard recovery-threshold calibration, nominal-first execution and held-out paired non-inferiority. It does not falsely describe standard calibration as a Safe policy gate.
- Because the current certificate population has been repeatedly inspected, it is development evidence. A final paper claim requires a newly sealed or preregistered certificate population.

### Engineering corrections

1. Recursively JSON-normalize `Path`, NumPy scalars/arrays, tuples, sets and nested structures in the policy-certificate output.
2. Forward and checkpoint the frontier, component-prior and bounded/unbounded-admission fields in both training and inference. Add a fail-closed `MODEL_INFERENCE_CONTRACT.json` preflight before certificate evaluation.
3. Define checkpoint opportunities exactly as proposal-contained actions with `teacher_advantage >= positive_gain` and `component_harmful=false`. Harmful actions cannot count as positive admission.
4. Add safe-positive admission recall, safe admission precision, invalid admission rate, evidence safe top-1 accuracy and evidence safe top-1 regret to checkpoint selection.
5. Train safe utility on the exact runtime score `sigmoid(admission_delta)-0.5`, with teacher targets clamped to the same `[-0.5,0.5]` execution range.
6. Enforce exact split IDs for standard calibration, adaptation-dev rule fitting and certificate verification.
7. Preserve three-way return-code semantics: 0 pass, 20 valid gate rejection, 30 engineering/protocol/artifact failure.
8. Add a repair-only script that re-evaluates existing server-side v48.25 checkpoints using corrected inference and JSON serialization without retraining. Its result is diagnostic and does not validate v48.26 training changes.
9. Fix post-contact bucket classification so `near_contact` cannot be classified as Contact merely because its name contains the substring `contact`.

### Physical execution diagnostics

The existing offline PCD teacher (`DRS * sigmoid(R_dep) * exp(-gap)`) describes deployable recovery headroom but does not directly encode temporal recovery processes. Existing NPZ/teacher indices do not contain candidate-level physical labels, so v48.26 does not fabricate an auxiliary training target. Instead it first makes the development shadow and authorized held-out evaluation sufficiently expressive to diagnose offline/physical transfer.

**Near-contact additions**

- near/critical exposure episode count and longest continuous exposure run;
- time to minimum clearance and minimum TTC;
- terminal clearance/TTC and recovery gain from the most dangerous point;
- clearance/TTC deficit AUC;
- absolute acceleration p95/max, maximum deceleration, jerk max and yaw-rate max;
- paired checks for collision/overlap non-inferiority, intervention bursts, route progression and dynamics.

**Post-contact additions/fixes**

- use an explicit causal post-contact anchor at step 0 for a post-contact target instead of discovering the first overlap inside the rollout;
- secondary/re-contact event, episode and scene rates;
- overlap episode count, duration and longest run;
- normalized free-space AUC and clearance-deficit AUC;
- terminal clearance, clearance gain and time to peak clearance;
- sustained escape rate and time to sustained escape;
- stable-stop quality requiring low speed, no overlap, on-road state, bounded yaw rate and sustained duration, plus time to quality stable stop;
- paired checks for offroad, route progression, jerk, yaw rate and intervention burden.

If safe offline ranking improves but these development-shadow metrics do not, the next version should preregister candidate-level physical teacher rollouts and a separate temporal recovery auxiliary head. Certificate labels must not be changed retrospectively to fit observed shadow outcomes.

### v48.26 algorithm: EXECUTION-PHYSICS-BRIDGE

**EXECUTION-PHYSICS = Exact eXecutable Evidence Contract, Unified safe utility, Checkpoint-safe opportunity, Independent Thresholding, Observation-consistent Nominal policy, and Physical recovery diagnostics.** The deployed model remains unified and receives no Safe/Near/Contact regime ID.

1. Preserve frozen top-3 proposal, semantic low-risk prior, centred frontier admission, categorical nominal-plus-top-k policy, safe-utility supervision and disabled legacy Noisy-OR.
2. Replace raw-positive checkpoint accounting with gate-exact safe-positive accounting.
3. Make safe-utility training mathematically identical to the executed score.
4. Select checkpoints using executable safe recall/precision, invalid admission and safe ranking, rather than soft mass or harmful raw-positive hits.
5. Keep adaptation-dev threshold fitting and independent full-certificate verification separate and hash-addressed.
6. Add rich temporal closed-loop diagnostics before introducing any new physical auxiliary teacher.

### Required v48.26 ablations and two-A30 schedule

1. `A_engineering_contract_only`: JSON, inference parity, exact split and return-code repair only.
2. `B_add_safe_checkpoint_contract`: A plus gate-exact checkpoint opportunity/precision/invalid-admission metrics.
3. `C_add_execution_exact_safe_utility`: B plus exact runtime safe-utility scale.
4. `D_full_execution_physics_bridge`: complete v48.26 algorithm and physical diagnostics.

The launcher runs four waves. In every wave Balanced occupies GPU0 and Precision occupies GPU1; each A30 runs one task at a time and receives four tasks total. Maximum concurrency is two.

### Decision rules

- Run the repair-only certificate first to determine whether the existing v48.25 checkpoint can now be evaluated. It is not a replacement for the full v48.26 experiment.
- `RC=0`: only the automatically generated `NEXT_COMMANDS.txt` may authorize Safe paired and held-out stress/closed-loop.
- `RC=20`: do not read test/stress. Run adaptation-dev shadow and the four-wave ablation suite. Distinguish low safe recall, low precision/high invalid admission, dev-to-certificate generalization failure, and offline-to-physical mismatch.
- `RC=30`: no algorithm conclusion is permitted. Inspect `PIPELINE_FAILED.json`, model-contract files, checkpoint summaries and logs.
- Do not manually create `NEXT_COMMANDS.txt`, lower the registered gate after observing results, call the repeatedly inspected certificate a final untouched paper certificate, or invent physical training targets absent from the current teacher data.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 233 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- JSON serialization, training/inference parity, safe-positive checkpoint semantics, exact execution score, exact split IDs, Near/Contact bucket separation and new physical metrics have focused tests.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.26 gate or closed-loop result is claimed in advance.

## v48.25 — OC-TRAC-INTEGRITY-BRIDGE (2026-07-31)

### v48.24 result attribution

- The uploaded v48.24 run returned `RC=30`, but this does **not** establish algorithmic regression. Balanced and Precision both completed adaptation and produced checkpoints. Four non-empty Near/Contact certificate artifacts were evaluated; the controller converted each worker's structural-support rejection (`rc=4` in v48.24) into pipeline failure 30. Structural infeasibility is a valid Natural-gate rejection and must map to worker 3 / controller `RC=20`; only missing, empty, corrupt or protocol-inconsistent artifacts map to `RC=30`.
- The v48.24 model was not the intended SUPPORT-BRIDGE implementation. The run scripts set `model.direct_recovery_evidence_frontier=true` and `model.direct_recovery_evidence_component_prior_logit=-2.0`, but `ocrap.cli.train` omitted both keyword arguments when constructing `OCRAPModel`. Runtime therefore used the legacy default `frontier=false`; zero component logits again represented approximately 0.5 harmful probability and admission again used the non-centred `benefit-softplus(harm)` prior. This silent configuration drop is sufficient to produce universal abstention and invalidates v48.24 as a test of the intended semantic prior/frontier design.
- Validation stratification also reused the adaptation-train teacher index for adaptation-dev paths. The log consequently reported all 409 dev groups as `dead_or_mixed`, despite separately computed validation statistics containing positive groups. Checkpoint selection was therefore based on a mislabeled validation sampler.
- Both variants selected checkpoints with `direct_raw_admission_rate_near=0`, `direct_raw_admission_rate_contact=0`, and zero positive admission recall. The v48.24 `direct_frontier_selection_risk` could still improve through soft mass while the executable policy remained all-abstain.
- Expanding the frozen proposal from top-3 to top-8 did not recover additional fit-side safe-positive support. Near fit remains 3 safe-positive groups at k=1/3/5/8. Contact fit increases from 7 at k=1 to 10 at k=3 and remains 10 at k=5/8. Proposal width is therefore not the current bottleneck; top-8 adds computation and ranking ambiguity without adding certificate opportunities.
- Near Balanced preserves raw-benefit AUC (0.855) but learned safe-benefit AUC, conditional harm AUC and regret worsen. Near Precision shows one partial positive signal—conditional harm AUC improves from 0.527 to 0.611 and false-switch falls from 0.221 to 0.153—but harmful-switch rises from 0.492 to 0.622, top-1 regret rises from 0.091 to 0.202, and coverage stays zero. This is not safe admission.
- Contact remains unresolved. Precision Contact learned benefit/safe-benefit/harm AUC all decline, conditional harm AUC falls to 0.443, correlation remains negative, and top-1 regret rises to 0.264. Balanced Contact remains below-random for safe/harm ordering and has negative correlation. No v48.24 closed-loop shadow result exists, so none of the physical Near/Contact publication targets is supported.

### Certificate decision

- The certificate concept is retained. It is the statistical authorization layer that correctly prevents an unsafe or unsupported learned selector from being evaluated on held-out test/stress. Removing it would hide failure rather than solve it.
- The old internal 50/50 certificate fit/verify split is no longer suitable for this sparse safe-positive population. Across the complete Near certificate there are 9 proposal-contained safe-positive groups, but the old split separately required approximately 8 fit positives and 5 verify positives; that contract can be impossible even when the full independent population has enough support. Contact has 20 total safe-positive groups, but its old fit half still misses the precision-LCB requirement.
- v48.25 fits all opportunity/harm/score/rank thresholds on `evidence_adapt_dev`, freezes the rule and SHA256 provenance, then evaluates the **entire** scene-disjoint certificate population in verification-only mode. Certificate labels never alter thresholds. The numerical verification requirements are not relaxed.
- This is a new protocol and must not be used to reinterpret v48.24 retrospectively. Because the current certificate population has already been inspected during algorithm development, results on it are development evidence. A final CCF-A paper claim requires a newly sealed/preregistered certificate population, even if the fixed dataset is retained for the next diagnostic round.

### Engineering corrections

1. Forward `direct_recovery_evidence_frontier`, `direct_recovery_evidence_component_prior_logit`, and `direct_recovery_evidence_admission_bounded` from the CLI config into `OCRAPModel`.
2. Build a separate exact teacher-PCD index for adaptation-dev and pass it through `training.validation_group_index_path`; otherwise validation stratification is disabled rather than silently using a train-only index.
3. Rename the file-name heuristic `safe_positive_fraction` to `legacy_safe_root_positive_fraction`; exact safe-positive prevalence is reported only by the teacher index.
4. Map valid structural/learned certificate rejection to `RC=20`. Reserve `RC=30` for empty data, corrupt artifacts, protocol/index mismatch, training/checkpoint failure or runtime exceptions.
5. Keep strict deployment authorization, while allowing an adaptation-dev-only shadow diagnostic to load the dev-frozen selector after independent certificate rejection.
6. Add `check_v48_25_regime_targets.py` and restore complete Near/Contact physical target checking in the shadow workflow.

### v48.25 algorithm: INTEGRITY-BRIDGE

**INTEGRITY = Identity-preserving Non-regime Evidence with Gate-True Risk, Independent dev labels, Threshold freezing and Yielding admission.** It remains one unified model and does not expose Near/Contact regime IDs at inference.

1. **Correct semantic frontier execution.** The low-risk component prior and centred identity-preserving admission path are now actually instantiated, not merely written to the shell command.
2. **Executable-admission checkpoint barrier.** `direct_integrity_selection_risk` adds hard Near/Contact positive-recall shortfall and an explicit all-abstain penalty to the existing frontier risk. A checkpoint cannot win only by improving soft mass while selecting no action.
3. **Unbounded, zero-initialised admission residual.** The primary model removes the `tanh` ceiling from the admission residual. Zero initialisation still preserves the transferred prior exactly, but the residual can cross the nominal-vs-recovery boundary when the source prior is conservatively negative. Global gradient clipping remains active.
4. **Top-3 restored.** Since k=8 adds no safe-positive support, the default returns to frozen top-3. This avoids extra harmful/ambiguous candidates and makes the ablation isolate algorithm integrity rather than proposal width.
5. **Safe-utility remains the deployed target.** Continuous safe-utility regression/listwise supervision and categorical nominal-plus-top-k learning remain active; legacy Noisy-OR remains disabled.
6. **No retrospective label relaxation.** The current component-veto teacher is kept for the clean v48.25 main run. A future protective-Pareto label ablation may separate hard collision/offroad vetoes from soft DRS/deployability/gap trade-offs, but it must be versioned and evaluated on fresh sealed evidence.

### Non-repeated ablations and two-A30 schedule

- `A_wiring_fix_bounded`: only repair the missing semantic frontier/prior wiring; retain bounded admission and the previous checkpoint metric.
- `B_add_integrity_checkpoint`: A plus executable-admission checkpoint barrier.
- `C_add_unbounded_admission`: B plus the unbounded zero-initialised admission residual.
- `D_full_integrity_bridge`: C plus continuous benefit listwise and stronger safe-vs-harmful frontier contrast.

The launcher runs four waves. In every wave Balanced occupies GPU0 and Precision occupies GPU1; each A30 runs one task at a time and receives four tasks total. Maximum concurrency is two.

### Return-code interpretation

- `RC=0`: both Near and Contact pass the dev-frozen, full-certificate verification gate; only the authorization-checked stress script may read held-out test/stress.
- `RC=20`: pipeline and certificate data are valid, but structural support or the learned selector fails. Do not read held-out test/stress. Run adaptation-dev shadow and the A/B/C/D ablations.
- `RC=30`: engineering/protocol/index/training/checkpoint/empty-artifact failure. Inspect `PIPELINE_FAILED.json` and logs before drawing any algorithm conclusion.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 225 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- `check_v48_25_regime_targets.py --help`: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.25 gate or closed-loop result is claimed in advance.

## v48.24 — OC-TRAC-SUPPORT-BRIDGE (2026-07-31)

### v48.23 result attribution

- The uploaded v48.23 controller is a valid Natural-gate evaluation: both variants trained, certificate fit/verify folds are non-empty and scene-disjoint, held-out test/stress roots were not read, and the controller returned `RC=20`.
- The decisive new finding is that the v48.23 proposal-constrained oracle does **not** pass the fit fold. Under frozen top-3 and the current component-veto label, Near fit contains only 3 safe-positive groups; its optimistic oracle can select 10 but obtains only 3 positives and precision LCB 0.1538. Contact fit contains 10 safe-positive groups; selecting 16 yields precision LCB 0.4652, below the 0.5 fit requirement. Verify is feasible in both regimes. Therefore this round is not only a calibrator/representation failure: the fit-side proposal/label/gate support contract is itself insufficient.
- The earlier statement that proposal recall is approximately 0.97--1.00 remains true only for **raw-benefit** opportunity. It must not be interpreted as safe-positive proposal sufficiency. The distinction is now reported explicitly.
- Near Balanced retains high raw-benefit AUC but has below-random harm ordering and negative learned top-k correlation. Near Precision retains part of broad risk recognition, but harmful-switch remains about 0.49 and verify coverage remains zero. The benefit signal has not become safe admission.
- Contact Precision retains broad harm AUC near 0.65, but conditional harm AUC falls to about 0.50, learned benefit AUC remains about 0.42, correlation is negative, and regret is unchanged. Balanced Contact is materially worse than v48.22. The claimed Contact improvement therefore did not materialize.
- All eight v48.23 ablations complete and reject. A recovers part of broad harm AUC; B does not establish continuous ranking; C gives only small frontier changes; D does not dominate B or C. Additional epochs on the same objective are not a justified next step.

### Engineering defects fixed

1. **RC=20 dev-shadow was impossible to launch.** `run_v48_23_dev_shadow_closed_loop.sh` called the strict deployment entry, while `run_ocrap_v48_trac_sr.sh` rejected every certificate with `valid_for_deployment=false`. This is an engineering defect, not a user command error. A dedicated `DEV_SHADOW_DIAGNOSTIC=1` path now consumes only a fit-derived diagnostic selector and remains forbidden from test/stress.
2. **Runtime did not execute the certified policy.** Training and calibration used frozen top-k plus Evidence reranking, but the closed-loop loader read only score/opportunity/harm thresholds. Runtime silently fell back to `proposal_top_k=1` and `evidence_rerank_top_k=false`. It now loads the complete selector contract, including rank margin, top-k, rerank and conditional-ranking flags.
3. **Categorical and Noisy-OR objectives were both active.** v48.23 introduced a one-action categorical objective but left the legacy group-opportunity/Noisy-OR term at weight 1.25. The two targets are incompatible when only one action can execute. SUPPORT-BRIDGE disables Noisy-OR by default.

### v48.24 algorithm: SUPPORT-BRIDGE

**SUPPORT = Safe-Utility Proposal-Policy Ordering with Runtime-True Transfer.** It preserves one unified model and does not expose Near/Contact IDs to inference.

1. **Safe-positive support width.** The frozen proposal is widened from top-3 to top-8 for the new preregistered version. This is not unrestricted proposal retraining; it tests whether safe recovery variants exist below the raw-benefit top-3.
2. **Support curve audit.** Every certificate reports optimistic proposal-constrained oracle feasibility for k=1,3,5,8 and the active k, separately for fit and verify. Structural support failure is no longer confused with learned-gate failure.
3. **Safe-benefit opportunity semantics.** Gate training and calibration use continuous positive PCD only when component harm is false. Raw-beneficial but harmful actions are not counted as admission opportunities.
4. **Direct deployed safe-utility target.** The exact final admission logit receives continuous regression and listwise supervision. A safe action target is its signed PCD advantage; a component-harmful action receives a strictly negative target `-max(|delta|, positive_gain)`. This removes the requirement that an indirect benefit head and sparse risk head happen to cancel correctly.
5. **One-action-only group learning.** The categorical nominal-plus-top-k policy remains active, while legacy Noisy-OR group opportunity is disabled.
6. **Safe-positive group sampling.** Group batching explicitly preserves safe-positive groups and balances hard-negative/harmful groups without regenerating the dataset.
7. **Light frontier contrast.** Pairwise safe-versus-harmful contrast is retained only as a small auxiliary term; it no longer carries the main safety-transfer burden.
8. **Runtime-true certificate contract.** Deployment and diagnostic execution consume the same top-k/rerank/rank-margin contract written by the adaptation and certificate stages.

### New non-repeated ablations and two-A30 schedule

- `A_top3_safe_label_baseline`: safe-benefit labels with top-3; isolates label semantics from support width.
- `B_top8_support_only`: A plus top-8; isolates proposal support width.
- `C_top8_safe_utility`: B plus direct continuous safe-utility regression/listwise learning.
- `D_full_support_bridge`: C plus light high-benefit frontier contrast.

The launcher runs four waves. Each wave starts Balanced on GPU0 and Precision on GPU1, so only one task occupies each A30 at a time. Each GPU receives exactly four tasks; maximum concurrency is two rather than four processes per card.

### Decision rules

- `RC=0`: the preregistered certificate passed; run only the authorization-checked held-out stress/closed-loop script.
- `RC=20`: top-8 proposal support is feasible but the learned policy still fails. Do not read test/stress. Run fixed adaptation-dev shadow and A/B/C/D to distinguish safe-utility learning from physical transfer.
- `RC=30` with certificate support diagnostics: the new top-8 safe-positive oracle still cannot satisfy the gate or an engineering/protocol stage failed. No amount of calibrator retraining can make the current contract pass; inspect `PIPELINE_FAILED.json`, `proposal_support_curve`, and logs.
- Do not claim RC=0 in advance. It is theoretically plausible only if top-8 recovers enough fit-side safe-positive support and the learned safe-utility ordering reaches the finite-sample gate.

### Local validation

- `PYTHONPATH="$PWD/src" pytest -q`: 220 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment, so no v48.24 gate or physical closed-loop result is claimed.

## v48.23 — OC-TRAC-FRONTIER-BRIDGE (2026-07-31)

### v48.22 result attribution

- The uploaded v48.22 dedicated controller completed both Balanced and Precision
  adaptations, non-empty scene-disjoint certificate fitting/verification, protocol
  and teacher-index contract checks, and the test-root seal. It records
  `pipeline_valid=true`, `gate_evaluated=true`, `test_roots_read=false`, and
  `RC=20`. This is a genuine Natural-gate rejection rather than a controller,
  protocol, empty-pool, parameter-count, unsupported-support, or partial-variant
  failure.
- The preregistered gate remains mathematically feasible, but it is intentionally
  close to an oracle-quality selective policy at the current support. Near-fit must
  select at least 10 groups with at least 8 positives and zero harmful selections;
  Near-verify needs at least 5/8 positives and zero harmful selections. Contact-fit
  needs at least 11/16 positives and at most one harmful selection; Contact-verify
  needs at least 6/10 positives and zero harmful selections. These conditions must
  not be relaxed retrospectively in the v48.22 protocol.
- v48.22 does not fail only because the gate is strict. Precision learns broad
  component risk (candidate harm AUC about 0.64--0.66), but Near/Contact safe-action
  precision remains about 0.05--0.10 at the closest fit rules. Balanced selects the
  epoch-zero identity, leaves harm at 0.5, and effectively abstains. Every main and
  ablation verify certificate has zero coverage.
- The frozen proposal remains high recall: positive-group oracle-best hit is about
  0.97--1.00. Candidate generation is not the primary bottleneck. The unresolved
  problem is converting a proposal-contained opportunity into a calibrated,
  high-benefit, non-harmful action and deciding when to leave nominal.
- Training support is extremely imbalanced. Near has 25 safe-beneficial candidates
  among 1425 deployable candidates (1.75%), across 11 groups and 7 scenes; Contact
  has 106/4086 (2.59%), across 41 groups and 17 scenes. Global harm prevalence is
  approximately 54% in Near and 45% in Contact. Broad candidate AUC is therefore
  insufficient evidence of safety-frontier transfer.

### Root engineering and objective defects found in v48.22

1. **Neutral risk was encoded as 0.5 harmful probability.** Zero-initialized
   component logits produced `P(harm)=0.5`, even though the target contract defines
   a tolerance/deadband within which a candidate is non-harmful. Balanced early
   stopping therefore preferred an artificial all-abstain identity rather than a
   semantically neutral source policy.
2. **The detached admission prior had a constant negative offset.** v48.22 used
   `benefit_logit - softplus(harm_logit)`. At the zero residual identity this
   subtracts `log(2)`, so the new admission head is not identity-preserving and is
   pushed toward abstention before any target-domain evidence is learned.
3. **The two-head fallback was also structurally over-conservative.** Its score
   `P(benefit)*(1-P(harm))-0.5` is non-negative at neutral harm only when benefit is
   nearly certain. The A ablation therefore cannot isolate the third head fairly.
4. **Noisy-OR does not match the one-action deployment event.** The group objective
   treated top-k candidates as independent Bernoulli opportunities, while runtime
   chooses exactly one candidate or nominal. Noisy-OR inflates opportunity as top-k
   grows and can be satisfied by diffuse weak scores rather than one executable
   safe action.
5. **Benefit supervision was mostly a binary sign test.** Continuous PCD magnitude
   ordering was disabled (`CENTERED/DELTA_NLL/pairwise benefit` effectively zero or
   weak), explaining high Near candidate AUC but weak or negative score correlation,
   unstable Contact ranking, and positive-group top-1 regret.
6. **Checkpoint selection emphasized broad risk instead of the high-benefit safety
   frontier.** Global harm AUC/mass is dominated by abundant dead or obviously
   harmful candidates. The deployment-critical distinction is safe high-benefit
   versus harmful high-benefit recovery.
7. **Primary-gate-only debugging is information-poor.** A zero-coverage certificate
   cannot identify whether failure comes from proposal support, labels/features,
   ranking, admission, or overly conservative finite-sample certification. A
   proposal-constrained oracle audit and adaptation-dev-only shadow closed loop are
   required, without reading held-out test/stress roots.
8. **Contact event aggregation was incorrect.** `secondary_overlap_event` was
   aggregated by maximum across scenes, making any single event report as 1.0.
   v48.23 reports scene rates for secondary contact, stable stop and sustained
   escape. A duplicate scene-quantile computation was also removed.

### v48.23 algorithm: FRONTIER-BRIDGE

**FRONTIER = Factorized Recovery Opportunity with Non-compensatory Threat Evidence,
Rank-consistent Transfer, and Intervention Evaluation.** It remains one unified
model over Safe, Near and Contact. No regime ID, bucket router, bucket-selected
calibrator, or regime-specific residual is available at inference.

1. **Semantic non-harm prior.** Component risk logits start from a configurable
   low-risk prior (`-2.0` by default) and learn bounded residuals around that prior.
   The prior represents the component-veto deadband rather than an arbitrary 0.5
   harmful probability. Exact non-compensatory `max` aggregation is retained.
2. **Centered identity-preserving admission.** The detached prior is
   `benefit - [softplus(harm)-softplus(harm_prior)]`. At zero residual the admission
   logit exactly equals the transferred benefit logit. Risk can subsequently veto
   an action without imposing a fixed pre-training abstention penalty.
3. **Categorical one-action group policy.** The primary group objective is a softmax
   over nominal plus frozen proposal top-k, matching the actual decision that one
   action (or nominal) is executed. It replaces noisy-OR in FRONTIER runs while the
   legacy path remains checkpoint-compatible.
4. **Continuous top-k benefit ranking.** A vectorized listwise/KL objective uses the
   continuous raw PCD advantages inside the exact deployment top-k. It teaches which
   beneficial action is better, not only whether its signed delta exceeds a
   threshold. Candidates outside frozen top-k receive no listwise gradient.
5. **High-benefit safety-frontier contrast.** Admission logits for safe beneficial
   candidates are trained to outrank raw-beneficial but component-harmful candidates
   in the same group. This directly targets the false-safe frontier that broad harm
   AUC misses.
6. **FRONTIER checkpoint risk.** Early stopping adds high-opportunity harmful policy
   mass and false-admission mass to the threshold-free cross-regime risk, with a
   small global-harm tie-break only. Near/Contact remain validation strata and are
   never model inputs.
7. **Proposal-constrained oracle gate audit.** Certificate artifacts now report the
   most optimistic fit/verify feasibility achievable using non-harmful opportunities
   already contained in the frozen proposal. This audit ignores macro-concentration
   constraints and is therefore necessary but not sufficient. Oracle failure means
   more training cannot pass the current proposal/label/gate contract; oracle pass
   with model failure localizes the bottleneck to representation/ranking/admission.
8. **Adaptation-dev shadow closed loop.** After `RC=20`, a separate script runs only
   on adaptation-dev roots, never certificate/test/stress roots. It is explicitly
   diagnostic and non-paper. Held-out stress remains authorization-gated by
   `NEXT_COMMANDS.txt`.
9. **Physical regime diagnostics.** Near adds minimum clearance/TTC, near-contact and
   critical-TTC exposure durations, and clearance/TTC deficit integrals. Contact
   adds overlap duration and longest run, secondary contact rate, post-contact
   clearance/free-space integral, sustained escape rate/time, and stable-stop rate.
   The paired comparator reports metric-aware improvement direction.

### Non-repeated v48.23 ablations

1. `A_semantic_prior_categorical`: semantic risk prior, centered admission identity
   and categorical one-action group objective; no continuous ranking or frontier
   contrast. This isolates the v48.22 engineering corrections.
2. `B_add_benefit_listwise`: A plus continuous top-k PCD listwise supervision. This
   isolates benefit magnitude/ranking transfer.
3. `C_add_frontier_contrast`: A plus high-benefit safe-versus-harmful admission
   contrast. This isolates safety-frontier discrimination.
4. `D_full_frontier`: semantic prior, categorical group policy, continuous benefit
   ranking, component veto and frontier contrast.

All eight tasks are launched simultaneously. Round-robin assignment places four
jobs on GPU0 and four jobs on GPU1. Each job uses one DataLoader worker, batch size
56 and bounded host math threads. Main Balanced/Precision use separate A30s, batch
size 96, three workers, pinned persistent workers, prefetching and bfloat16 AMP.
The new losses are vectorized inside the existing model forward pass and do not add
an additional encoder or duplicate proposal computation.

### Near/Contact development targets and diagnostic policy

- **Near-contact:** preserve zero collision/non-inferior nominal safety; improve
  minimum clearance by at least 0.10 m and minimum TTC by at least 0.20 s in paired
  development analysis; reduce near-contact/critical-TTC exposure and safety-margin
  deficit integrals; target PCD >= 0.54, FRA <= 0.12, DRS >= 0.88, NUP >= 0.995,
  intervention <= 0.02, intervention-episode rate <= 0.012, maximum intervention run
  <= 1, and selector miss <= 0.034 for development (<= 0.025 publication target).
- **Contact:** target PCD >= 0.52, FRA <= 0.16, DRS >= 0.84, NUP >= 0.985,
  intervention <= 0.04, intervention-episode rate <= 0.025 and maximum run <= 2;
  reduce paired secondary-overlap scene rate by at least 0.02, overlap duration/run
  and residual impact; increase post-contact free-space, sustained escape and stable
  stop by at least 0.02 while preserving route/offroad/comfort constraints.
- These are development targets, not claimed v48.23 results. Publication claims
  require the preregistered gate, held-out authorized closed loop, paired confidence
  intervals and multi-seed confirmation.

### Decision and non-repetition rules

- `RC=0`: run only the authorization-checked held-out stress/closed-loop command,
  then multi-seed confirmation and final Safe paired non-interference evaluation
  using the same selected checkpoint.
- `RC=20`: do not read test/stress. First inspect the proposal-constrained oracle
  audit. If it fails, repair proposal/label/gate support under a newly preregistered
  protocol rather than training another calibrator. If it passes, run the
  adaptation-dev shadow closed loop and A/B/C/D ablations to localize
  ranking-versus-frontier-versus-admission failure.
- A dev shadow physical improvement with primary-gate failure means the certificate
  may be too conservative for the available sample size; document this and create a
  new preregistered protocol in a later version. Never relax v48.22/v48.23 gates
  retrospectively or use dev shadow results as paper results.
- `RC=30`: no algorithm conclusion is allowed. Repair the named protocol, index,
  training, checkpoint, calibration or artifact stage first.
- Do not repeat opportunity-only noisy-OR, neutral harm at probability 0.5,
  uncentered softplus admission, binary-only benefit supervision, global-harm-only
  checkpointing, regime-specific calibrators/residuals, raw high-dimensional
  context, proposal retraining, threshold-grid-only tuning, or dataset regeneration
  in this round.

### Local validation

- `pytest`: 216 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- Missing-protocol fault injection normalizes the controller and both failure
  artifacts to `RC=30`, `pipeline_valid=false`, and `test_roots_read=false`.
- New tests cover semantic risk initialization, identity-preserving admission,
  continuous top-k ranking gradients, frontier contrast gradients, categorical
  one-action probability, FRONTIER checkpoint sensitivity, proposal-oracle/dev-
  shadow plumbing and eight-task/two-GPU assignment.
- The delivery environment has no real WOMD/Waymax data or A30 GPUs. No v48.23
  Natural-gate or closed-loop result is claimed.


## v48.22 — OC-TRAC-COVENANT-BRIDGE (2026-07-30)

### v48.21 result attribution

- The v48.21 dedicated controller is valid and records `pipeline_valid=true`,
  `test_roots_read=false`, and `RC=20`; Near/Contact fit and verify support are
  non-empty and feasible. This is a genuine Natural-gate rejection, not a data,
  parameter-count, unsupported-gate, or controller failure.
- v48.21 converted part of the Near signal into learned selector evidence, but only
  in one objective branch. Balanced Near reaches candidate benefit AUC 0.841 and
  learned frozen-top-k benefit AUC 0.805, with positive top-1 regret 0.005. Precision
  Near collapses to 0.201/0.339. The Near signal is therefore real but not stable
  under the current training objective.
- Contact shows the first partial evidence of transferable safe-benefit selection in
  one branch: Precision candidate/learned benefit AUC is 0.586/0.613 and learned
  top-1 correlation is 0.166. Balanced Contact remains 0.540/0.432 with correlation
  -0.165. Contact capability is not jointly present with the strongest Near model.
- Component-risk learning is retained: learned harm AUC is approximately 0.60--0.65
  in Near and 0.64--0.66 in Contact. However, harmful top-1 switch rates remain
  0.45--0.70. Global harm AUC mainly separates harmful candidates from abundant dead
  candidates and does not adequately separate harmful candidates from high-benefit
  safe candidates at the deployment frontier.
- Every main and ablation certificate still has zero verify coverage. The nearest
  Balanced-Near fit rule selects 12 groups with only 2 positives, precision LCB
  0.071 and harmful UCB 0.673. The nearest Precision-Contact rule selects 19 groups
  with 2 positives, precision LCB 0.045 and harmful UCB 0.292. This is not a small
  threshold miss.
- The frozen proposal remains non-limiting: positive top-k oracle hit is about
  0.97--1.00. Proposal retraining remains prohibited in this round.

### Root defects found in v48.21

1. **Three deployment semantics were compressed into two heads.** Raw PCD benefit,
   componentwise harm, and final safe admission are different hypotheses. v48.21
   trained its benefit head on safe benefit while the preregistered gate continued
   to measure raw benefit plus an independent harm veto. This made one logit serve
   incompatible training, reporting, and certificate meanings.
2. **The safe-opportunity MIL probability was not safe.** The noisy-OR group loss used
   only `sigmoid(opportunity)` and ignored harm, even though its target was “at least
   one raw-beneficial and non-harmful candidate”. It therefore rewarded high
   opportunity on positive-but-harmful candidates—the exact false-safe failure mode
   exposed by Contact.
3. **Benefit and harm were combined differently in loss, checkpoint selection,
   calibration, and runtime.** Group MIL, safe-set loss, soft dev metrics and
   deployment did not all consume one explicit admission score. Candidate AUC could
   improve without improving the action actually certified or executed.
4. **Sparse admission gradients still competed semantically with raw benefit.** The
   same benefit logit was expected to preserve raw opportunities and also encode the
   conjunction `benefit AND non-harm`. Balanced learned Near while Precision learned
   part of Contact, producing complementary specialization rather than one model
   that works in both regimes.
5. **The zero-initialized source identity was never evaluated as a checkpoint.**
   Training started checkpoint comparison after epoch 1. A useful source/consensus
   initialization could not win if the first update damaged one regime.
6. **Sampler semantics needed an explicit task choice.** Raw benefit and safe
   admission now have separate heads. The positive group sampler must deliberately
   oversample safe-admission groups without changing the raw-benefit target.
7. **Certificate diagnostics were incomplete at the safety frontier.** Global
   benefit/harm AUC concealed failure among high-opportunity candidates. Safe-positive
   AUC and conditional high-opportunity harm AUC are required diagnostics.
8. **New admission branch had an uncovered runtime error during development.** The
   loss accepted `pred_admission_logit` but initially failed to initialize the local
   `admission_logits` tensor. A gradient-level unit test exposed and fixed this before
   delivery; static compilation alone would not have caught it.

### v48.22 algorithm: COVENANT-BRIDGE

**COVENANT = Cross-regime Opportunity, Veto Evidence, and Non-regime-specific
Admission with Nominal-preserving Transfer.** It remains one unified model over Safe,
Near and Contact. No regime ID, bucket-selected calibrator, or regime-specific
residual is available at inference.

1. **Three factorized hypotheses.** The unified model now predicts:
   - raw recovery benefit, trained on total PCD improvement and used by the primary
     opportunity gate;
   - DRS/deployability/gap component harm, aggregated by exact non-compensatory
     maximum and used by the independent harm veto;
   - final safe admission, trained on `raw benefit AND no component veto` and used
     for top-k reranking and group admission.
2. **Detached conservative admission prior.** The admission logit starts from
   `detach(raw_benefit_logit) - softplus(detach(harm_logit))` plus a zero-initialized,
   bounded admission residual. Sparse admission gradients cannot overwrite benefit
   or risk heads, while the prior remains conservative and context-correctable.
3. **Correct safe-opportunity MIL.** Frozen-top-k noisy-OR now uses the explicit
   admission probability. In the two-head ablation it uses
   `P(raw benefit) * (1-P(harm))`; it never uses opportunity alone. Candidates
   outside frozen top-k receive no group-opportunity gradient.
4. **One deployment score everywhere.** Training safe-set loss, soft checkpoint
   metrics, calibration, evaluator, selector and closed-loop runtime all use the
   explicit candidate-vs-nominal admission score. Legacy checkpoints fall back to
   their historical opportunity-minus-harm score.
5. **Raw-benefit/safe-admission sampler decoupling.** Raw benefit remains the benefit
   head target, while `group_batch_safe_positive_target=true` stratifies minibatches
   using safe-positive groups. Harmful-benefit overlap groups still supervise raw
   benefit and harm tails, but are not presented as positive admission groups.
6. **Epoch-zero checkpoint evaluation.** The source/consensus identity is validated
   and saved before any optimizer step and may win early stopping.
7. **COVENANT checkpoint risk.** The threshold-free CONCORD risk is retained and adds
   worst-regime harmful policy mass and false-admission penalties. This is dev-only,
   uses no test root, and does not relax the Natural gate.
8. **Frontier diagnostics.** Certificate reports now include candidate safe-positive
   AUC, learned top-k safe-positive AUC, and harm AUC conditioned on high opportunity,
   separating broad risk discrimination from the safety-critical decision frontier.

### Non-repeated v48.22 ablations

1. `A_two_head_safe_probability`: raw benefit + component harm, corrected
   `P(benefit)*(1-P(harm))` group probability, no third admission head. This isolates
   the v48.21 MIL/score engineering correction.
2. `B_triad_candidate_only`: three heads and candidate admission BCE, but no group MIL
   or setwise objective. This isolates the third hypothesis.
3. `C_triad_group_mil_aggregate`: admission head plus group objectives with one
   aggregate harm tail. This isolates component risk heads.
4. `D_full_covenant`: raw benefit, three component veto heads, admission head,
   deployment-exact safe-set loss, safe-opportunity MIL and COVENANT checkpoint risk.

All eight tasks (four groups times Balanced/Precision) are launched together. Round-
robin assignment places four tasks on GPU0 and four tasks on GPU1. Each task defaults
to batch size 48, one DataLoader worker, and bounded host math threads to reduce A30,
CPU and disk contention.

### Decision and non-repetition rules

- `RC=0`: run only the authorization-checked stress command generated by the
  independent certificate, then multi-seed confirmation and final Safe paired
  non-interference evaluation using the same selected checkpoint.
- `RC=20`: do not read test. Use A/B/C/D to identify whether the missing factor is the
  admission hypothesis, group supervision, or component veto. Do not tune the gate
  or proposal.
- `RC=30`: no algorithm conclusion is allowed. Repair the named protocol, index,
  training, checkpoint or certificate stage first.
- Do not repeat safe-benefit-overloaded opportunity heads, opportunity-only MIL,
  shared two-head admission semantics, fixed-threshold early stopping, exact-min
  benefit transfer, regime-specific calibrators, raw high-dimensional context,
  proposal retraining, threshold-grid-only tuning, or dataset regeneration in this
  round.

### Local validation

- `pytest`: 209 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- New tests cover three-head bucket invariance/capacity, detached-gradient isolation,
  harmful-benefit MIL rejection, independent safe-positive sampling, explicit
  admission-score plumbing, epoch-zero selection, and eight-task/two-GPU assignment.
- The delivery environment has no real WOMD/Waymax data or A30 GPUs. No v48.22 gate or
  closed-loop result is claimed.


## v48.21 — OC-TRAC-CONCORD-BRIDGE (2026-07-30)

### v48.20 result attribution

- The uploaded v48.20 dedicated controller completed both Balanced and Precision adaptations, a non-empty scene-disjoint certificate, manifest/protocol checks, and the test-root seal. It records `pipeline_valid=true`, `test_roots_read=false`, and `RC=20`. Fit/verify support feasibility is true in both Near and Contact, so this is a real Natural-gate rejection rather than the historical unsupported-gate or parameter-guard failure.
- Component-risk learning improved materially. Main candidate harm AUC is approximately 0.669--0.676 in Near and 0.661 in Contact, compared with approximately random harm evidence in v48.19. The component semantic reset and exact non-compensatory maximum are therefore retained.
- Natural-gate usability did not improve: every main and ablation certificate still selects zero verify groups. Near benefit is unstable across objectives/variants (Balanced main 0.445 versus Precision main 0.790), learned proposal-evidence correlation is only -0.074--0.125, and no gate-authorized Near closed-loop result exists.
- Contact remains the limiting regime. Main candidate benefit AUC is 0.487--0.495, proposal-evidence benefit AUC is 0.430--0.439, learned correlation is -0.074--0.105, and all verify coverage is zero. The strongest Contact benefit result is only the candidate-tail Precision ablation (0.603), whose harm AUC falls to 0.419 and still yields zero certificate coverage.
- The frozen proposal remains high recall (positive-group top-k oracle hit approximately 0.982--0.991). Proposal generation is not retrained in this version.

### Root defects found in v48.20

1. **Benefit/risk negative transfer in one shared adapter.** Safe-benefit positives are only about 3% of deployable candidates, whereas component-harm positives are about 45--54%. A single shared hidden representation allowed dense risk gradients to overwrite sparse benefit structure. The ablations expose this directly: candidate-only models retain the strongest Near/Contact benefit, while component-head models improve harm AUC but damage benefit AUC.
2. **The exact minimum expert envelope is too pessimistic for transfer.** v48.20 defines base benefit as `min(expert_1, expert_2)`. One source expert that is mismatched for a candidate can erase useful Near evidence even when the other expert is well calibrated. Balanced Near benefit collapses to 0.445 while the same frozen proposal still contains the opportunity almost always.
3. **Candidate tails do not directly supervise the group event.** Deployment first asks whether the frozen top-k contains any safe recovery and only then chooses a candidate. Candidate BCE and a setwise winner objective provide an indirect and unstable signal for this rare group-level admission event.
4. **Early stopping used fixed dev thresholds that were inactive.** The v48.20 metric used fixed opportunity/harm thresholds. Positive admission recall was zero through almost all dev epochs, making the selection risk nearly constant and causing Precision to select epoch 1. This cannot choose the checkpoint most likely to have a useful fit/verify frontier.
5. **Sampler semantics disagreed with safe-benefit training.** Raw PCD-positive but component-harmful overlap groups were placed in the positive stratum even though the safe-benefit group loss labelled them negative. In the uploaded index, Near has 16 raw positive groups but only 11 safe-positive groups; Contact has 44 raw and 41 safe-positive groups.
6. **Frozen proposal diagnostics were mixed with learned selector diagnostics.** The reported 0.86--0.93 non-positive false-switch rate and approximately 0.38--0.41 harmful ranked-switch rate are frozen tournament metrics and are identical across ablations. They diagnose the need for admission evidence, but they are not the learned gate's false-intervention rate. The learned gate currently abstains everywhere.
7. **Main/D ablation reproducibility was not exact.** Main and ablation jobs used different batch sizes/worker settings and non-deterministic cuDNN behavior, weakening causal comparison.

### v48.21 algorithm: CONCORD-BRIDGE

**CONCORD = CONservative Consensus Opportunity and Non-Compensatory Risk Decoupling.** It remains one continuous, bucket-invariant mechanism across Safe, Near and Contact; regime labels are never model inputs or inference routers.

1. **Permutation-invariant expert consensus.** Replace the exact minimum with `mean(expert benefit) - lambda * expert range` (`lambda=0.15` by default). This preserves transferable evidence when one expert is locally pessimistic while retaining an explicit disagreement penalty. The representation uses symmetric expert statistics, so expert ordering cannot act as a hidden regime identifier.
2. **Decoupled benefit and component-risk adapters.** Sparse safe-benefit and dense risk supervision receive separate zero-initialized bounded MLPs. Benefit keeps consensus transfer; harm remains an absolute semantic reset with DRS/deployability/gap component heads and exact `max` veto.
3. **Safe-benefit candidate target.** Opportunity supervision is `raw benefit AND not component harmful`. A candidate may still be a raw-benefit opportunity for certificate accounting, but the trained admission score is not rewarded for unsafe benefit/harm overlap.
4. **Frozen-top-k multiple-instance opportunity objective.** A noisy-OR loss directly supervises whether the deployed proposal top-k contains any safe beneficial candidate. Candidates outside frozen top-k receive no group-opportunity gradient. The existing deployment-exact safe-set objective remains primary; candidate tails and ranking terms remain auxiliary.
5. **Safe-positive stratified sampling.** When safe-benefit training is enabled, raw-positive/component-harmful groups are no longer sampled as positive admission groups. The teacher-index audit now reports safe-positive candidate/group/scene support explicitly.
6. **Threshold-free checkpoint selection.** `direct_concord_selection_risk` uses soft top-k safe-opportunity NLL, soft recall, false-admission mass, harmful policy mass, safe-candidate mass and safe top-1 regret. Near/Contact are only robust validation strata; the model receives no regime ID. Fixed 0.65/0.30 thresholds remain diagnostic and do not drive early stopping.
7. **Primary certificate semantics preserved.** The preregistered Natural gate continues to count raw PCD benefit and independently veto component harm (`OPPORTUNITY_LABEL_MODE=raw_benefit`). `safe_benefit` is supported as a separate audit mode but is not silently substituted into the primary gate, because current Near fit/verify support is sparse.
8. **Deterministic attribution.** Main and D use the same default batch size (72), deterministic algorithms, and `cudnn.benchmark=false`. Both variants must complete; all non-0/20 lower-level failures normalize to `RC=30`.

### Non-repeated v48.21 ablations

1. `A_safe_target_legacy_trunk`: safe target, safe sampler, group MIL and soft checkpoint metric, but retain the v48.20 shared trunk/exact-min architecture. This isolates the architecture correction.
2. `B_concord_candidate_only`: consensus plus decoupled adapters, but no group MIL or safe-set objective. This tests whether architecture alone is sufficient.
3. `C_concord_group_mil_aggregate`: consensus, decoupled adapters and group objectives with one aggregate harm head. This isolates component heads.
4. `D_full_concord`: full consensus, decoupled component risk, safe-positive sampler, frozen-top-k group MIL, deployment-exact safe set and threshold-free checkpoint selection.

Balanced and Precision run in separate waves. Each wave launches four tasks concurrently, two tasks per A30 and one DataLoader worker per task. Do not repeat exact-min unified benefit, one shared benefit/harm trunk, raw-positive sampler under safe targets, fixed-threshold early stopping, threshold-grid-only tuning, proposal retraining, regime-selected calibrators, or signed-total-PCD harm.

### Decision rules

- `RC=0`: run only the authorization-checked stress command generated after the independent certificate; then run multi-seed confirmation and the final Safe paired non-interference experiment.
- `RC=20`: do not read test. Compare A/B/C/D to determine whether the remaining failure is consensus transfer, group opportunity learning, or component calibration. Do not relax the Natural gate in the same output directory.
- `RC=30`: repair the stage-specific engineering failure before making any algorithm conclusion.
- Dataset regeneration is still deferred. The index/sampler/reporting fixes do not modify the three regime datasets.

### Local validation

- `pytest`: 203 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every `scripts/*.sh`: passed.
- The delivery environment does not contain the real WOMD/Waymax datasets or the two A30 GPUs. No v48.21 Natural-gate or closed-loop result is claimed.

## v48.20 — OC-TRAC-UNISON-BRIDGE (2026-07-30)

### v48.19 result attribution and CCF-A readiness

- The uploaded v48.19 dedicated run completed both variants, non-empty scene-disjoint certificate fitting/verification, protocol-manifest checks, support-feasibility checks, and the test-root seal. Its controller records `pipeline_valid=true`, `test_roots_read=false`, and `RC=20`. Unlike the historical v48.18 Near specification, all v48.19 fit/verify support bounds are mathematically feasible; this is a genuine certificate rejection rather than a parameter guard or impossible-gate artifact.
- The frozen recovery proposal is already high recall: oracle-best top-k hit is approximately 0.982–0.991 in the main run and approximately 0.98–1.00 across ablations. The failure is therefore downstream of candidate generation.
- Near retains a useful benefit signal (main candidate benefit AUC 0.708–0.759; the strongest shared-only ablation reaches 0.800), but harmful evidence and group admission do not generalize. Contact benefit/harm AUC remains close to random or inverted, and every main/ablation certificate has zero deployable verify selections.
- Safe is ready only for the non-interference claim: the same mechanism must remain nominal when recovery is unnecessary. Near and Contact are not yet ready for a CCF-A main-result claim because no gate-authorized closed-loop OC-RAP result exists. Overall submission readiness is therefore **not reached**.
- Available external references are used only as progress anchors. Safe nominal/log/Wayformer artifacts are complete; Contact has complete 50-scene closed-loop baselines; Near offline baselines are complete, but all uploaded Near closed-loop summaries are incomplete or count-inconsistent and are excluded from paper-ready comparisons.

### Root defects found in v48.19

1. **Candidate classification replaced the deployed group decision.** `direct_value_ordinal_evidence_balanced_replaces_erm=true` allowed candidate-level balanced BCE to replace group ERM, while setwise admission, top-k, and intragroup terms were zero or negligible. Training optimized candidate labels, whereas deployment chooses nominal versus one recovery candidate per group.
2. **Training and deployment used different action scores and candidate supports.** The old setwise path scored every recovery candidate using frozen PCD plus log-sigmoid tails. Certificate/closed-loop first freezes proposal top-k and then reranks only that set with `sigmoid(benefit)-sigmoid(harm)`. Candidate AUC could improve without improving the actual deployed action.
3. **Harm semantics changed without resetting the source prior.** FACET component-veto harm was still added as a residual to the old signed-PCD source harm logit. The old base and new target do not represent the same event, so zero initialization was not semantic identity.
4. **Factorized supervision remained partly contaminated by signed three-class masks.** Class weighting, hard mining, and intragroup harm masks still used signed total PCD labels in several paths instead of component-veto labels.
5. **Regime-conditioned routing remained in the model.** Shared plus bucket-selected residual calibrators still consumed regime/bucket identity, so the learned policy was not a single continuous mechanism across Safe/Near/Contact.
6. **One aggregate harm tail discarded the observed physical structure.** DRS, deployability, and gap degradation account for nearly all harmful examples; hard-violation and `harm_proxy` positive increments are too sparse to support learned tails.
7. **The normalized smooth envelope was not conservative.** Normalized soft-min/soft-max lies inside the input range, so it can overestimate the weakest benefit expert and dilute one high-risk component with several low-risk components.
8. **External-baseline completion was not uniformly audited.** Several Near closed-loop summaries reported totals inconsistent with progress or scene journals. These artifacts must not enter a paper table.

### v48.20 algorithm: UNISON-BRIDGE

**UNISON = Unified Non-regime-specific Intervention Selection with Observation-consistent Non-compensatory evidence.**

1. **One bucket-invariant evidence model.** Inference does not receive a regime ID and does not select a Near/Contact calibrator. Both frozen source experts are evaluated for every candidate. The shared calibrator consumes their outputs, means, absolute disagreement, frozen policy margins, and tournament context.
2. **Conservative benefit transfer.** The source benefit is the exact lower envelope `min(expert_1, expert_2)`, followed by one zero-initialized bounded shared residual. This retains transferable Near signal while treating expert disagreement as lack of confidence without first classifying the regime.
3. **Componentwise harm semantic reset.** Three explicit zero-initialized bounded heads estimate nominal-relative DRS, deployability, and gap risk. The aggregate harm logit is the exact `max` across heads. No old signed-PCD harm base is added. Hard violation and `harm_proxy` remain deterministic certificate vetoes until their positive support is sufficient.
4. **Deployment-exact safe-set admission.** The frozen tournament first forms proposal top-k. The teacher safe set is `beneficial AND not component-harmful` within that top-k. If empty, nominal is the sole group target; otherwise a temperature-weighted distribution is formed over safe recovery candidates. The loss uses the exact deployed score `sigmoid(benefit)-sigmoid(harm)` and gives no safe-set gradient to candidates outside frozen top-k.
5. **Group objective is primary.** Candidate balance, component BCE, top-k auxiliary, and intragroup ranking remain auxiliary. They can no longer replace group ERM. Global balancing is bucket-agnostic.
6. **Safe is an invariant boundary, not a routed strategy.** Nominal rows remain pinned, and stress/test remains sealed until the same Natural gate authorizes execution.
7. **Worst-regime metrics are evaluation-only.** `direct_unison_selection_risk` can select a checkpoint using the worst Near/Contact validation behavior, but regime labels are not passed to the model or used for inference routing.

### Engineering corrections

- Set `ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM=false` in the v48.20 pipeline.
- Use factorized component labels for component weights, hard masks, and intragroup supervision; use strict `margin > 0` for harmful membership.
- Use exact `min` benefit and exact `max` harm envelopes, preserving legacy DUET/FACET behavior when UNISON is disabled.
- Add a float64 gradient-norm reducer before clipping to prevent finite float32 gradients from overflowing the norm reduction and silently zeroing an update.
- Persist UNISON model flags and component-head geometry in checkpoints and inference bundles.
- Keep protocol preflight, manifest SHA256 binding, teacher-index contract/rebuild, both-variant completion, normalized return codes, and test-root sealing.
- Add `tools/audit_external_baseline_artifacts_v48_20.py`; a closed-loop summary is paper-eligible only when progress is complete and progress/summary/journal scene counts agree.

### Non-repeated v48.20 ablations

- `A_candidate_tail_only`: unified experts plus component candidate tails, no safe-set group objective.
- `B_safe_set_aggregate_harm`: deployment-exact safe-set objective with aggregate harm, no component heads.
- `C_component_safe_set_no_balance`: component heads plus deployment-exact safe-set, no global auxiliary balance.
- `D_full_unison`: component heads, deployment-exact safe-set, global auxiliary balance, and robust checkpoint selection.

Run Balanced and Precision as two waves. Each wave launches four tasks concurrently, two tasks per A30, one DataLoader worker per task. Do not repeat v48.19 separate/shared/regime-residual comparisons, threshold-only tuning, signed-PCD harm, raw-context expansion, or candidate-BCE replacement.

### Decision rules

- `RC=0`: run stress/closed-loop only through the automatically generated `NEXT_COMMANDS.txt` authorization.
- `RC=20`: the v48.20 protocol and artifacts are valid but the algorithm is rejected; run the four new ablations without reading test.
- `RC=30`: inspect the stage-specific failure JSON/log and fix engineering only. Do not interpret it as an algorithm result and do not mutate protocol settings in the same output directory.

### Validation

- `pytest`: 196 passed, 5 warnings.
- `python -m compileall -q src tests tools`: passed.
- `bash -n` for every `scripts/*.sh`: passed.
- The delivery environment did not contain the real WOMD/Waymax datasets or two A30 GPUs. No v48.20 Natural-gate or closed-loop outcome is claimed.

## v48.19 — OC-TRAC-FACET-BRIDGE (2026-07-30)

### Result attribution corrected before further tuning

- A recovered v48.17 `RC=20` is an algorithmic Natural-gate rejection only when the dedicated certificate is non-empty, scene-disjoint, and the controller records `pipeline_valid=true`. The earlier missing-report/`RC=30` failure was the separate 78,630-vs-20,000 parameter-guard bug already documented in v48.18.
- The uploaded v48.18 dedicated run completed adaptation and a non-empty independent certificate, returned `RC=20`, did not read test roots, and selected no verify groups for either variant. However, that result is **not a clean algorithm-only rejection** because the historical Near-fit specification was unsupported by its own split.
- Near fit contained only eight positive opportunities but required at least twelve selections and `precision LCB90 >= 0.50`. With the historical `z=1.6448536`, even an oracle selecting all eight positives plus four non-harmful negatives has LCB `0.43149`; no model or threshold could pass.
- The code labelled those directional bounds as 90% LCB/UCB while using the central two-sided 90% critical value. v48.19 declares the convention explicitly and uses one-sided 90% Wilson bounds (`z=1.2815516`). Under the new, separately preregistered protocol, Near fit uses 10 selections (`8/10` optimistic LCB `0.60160`) and verify retains 8 (`6/8` optimistic LCB `0.52371`, zero-harm UCB `0.17033`). This is a new protocol and must not be used to retroactively relabel v48.18 as passing.

### v48.18 ablation conclusions

1. `A_dual_scalar` preserved the only robust signal: Near benefit AUC (0.817 Balanced / 0.756 Precision). Harm AUC remained near random and Contact remained weak.
2. `B_dual_tournament` modestly improved harm ordering (especially Near) but reduced benefit ordering. It did not make Contact separable and therefore does not justify the current tournament context as a standalone improvement.
3. `C_dual_tournament_balanced` was unstable: Balanced nearly reverted to A; Precision gained only small harm AUC while losing benefit AUC and increasing intervention/harm on dev.
4. `D_full_duet` selected the same epochs and produced numerically identical certificate metrics as C for both variants. The v48.18 cross-regime checkpoint metric had no observed causal effect.
5. All v48.18 variants remained all-abstain on verify. Threshold search cannot solve this because the underlying Contact evidence and harm supervision are not discriminative.

### Root algorithm defect fixed

- v48.18 made the network outputs independent but generated both labels from one signed total PCD delta. Consequently benefit and harm were still mutually exclusive in supervision, even though a Contact recovery candidate can improve total deployability while worsening DRS, hard violation, gap quality, or post-contact stability.
- v48.19 introduces **FACET-BRIDGE: Factorized Advantage and Componentwise Evidence Transfer with a shared cross-regime bridge**.
- The benefit tail remains total PCD advantage. The harm tail is a non-compensatory component veto over nominal-relative DRS, deployability-gate probability, gap discount, hard violation, and `harm_proxy`. Benefit and harm can now be simultaneously positive.
- Component harm uses strict tolerance exceedance. Equality at the tolerance boundary is non-harmful; the default normalized deadband is 0.05 for each component. This removes the previous `component_margin == 0` soft-label ambiguity.
- Near and Contact share one zero-initialized bounded calibrator, with a small bounded regime residual (`scale=0.25`). This partially pools sparse evidence across regimes while retaining phase-specific corrections. Safe is the nominal boundary condition and remains protected by the verified nominal lock.
- Default trainable evidence-correction parameters are 2,298: three 766-parameter modules (one shared plus two regime residuals), far below the v48.17 raw-context calibrator and below the architecture-aware 8,000-parameter guard.

### Engineering and statistical safeguards

1. Add train/certificate-shared target implementation in `src/ocrap/algorithms/evidence_targets.py`; no duplicated harm definition is permitted.
2. Add explicit Wilson confidence-level/bound-type implementation and optimistic certificate-support preflight. Unsupported gates return protocol/artifact failure rather than algorithm rejection.
3. Freeze `GATE_SPEC.json` before certificate scoring. It now binds the full statistical protocol and SHA256 identities of Safe/Near/Contact manifests; changing data or gate settings requires a new output directory.
4. Bind teacher-index reuse to train-root paths, manifest SHA256 values, PCD parameters, macro set, and component-veto tolerances. A stale index is automatically rebuilt.
5. Add a pre-training FACET target-support audit. Near and Contact must each contain positive and negative examples for both benefit and harm tails; absence of overlap is reported as a warning rather than fabricated.
6. Main runs require both Balanced and Precision adaptation branches by default. One failed branch yields normalized `RC=30`; partial variants are allowed only with the explicit debugging flag `ALLOW_PARTIAL_VARIANTS=1`.
7. Normalize controller semantics: `RC=0` is a valid Natural-gate pass, `RC=20` is a valid supported-protocol algorithm rejection, and every other lower-level failure becomes `RC=30` with a stage-specific JSON artifact.
8. Stress/test execution remains sealed unless an independently certified run creates `NEXT_COMMANDS.txt`.

### v48.19 non-repetition ablations

- `A_component_veto_separate`: factorized component-veto targets with separate regime calibrators.
- `B_shared_component_veto`: A plus shared cross-regime partial pooling and bounded regime residuals.
- `C_shared_only_no_regime_residual`: shared bridge only; isolates whether regime residuals are necessary.
- `D_full_facet`: B plus the FACET checkpoint metric, which prioritizes minimum cross-regime recall subject to harm/false-intervention budgets.

Run four tasks concurrently per wave: two tasks on each A30, one DataLoader worker per task, Balanced wave followed by Precision wave. Do not repeat simplex labels, signed-total-delta harm labels, raw 4,890-D context, auxiliary-only balancing, sampler-only balancing, or v48.18 C/D comparisons.

### Validation

- `pytest`: 185 passed, 5 warnings.
- `python -m compileall -q src tools`: passed.
- `bash -n scripts/*.sh`: passed.
- No real Waymax/WOMD/A30 experiment was executed in the delivery environment; no v48.19 Natural-gate or closed-loop result is claimed.

## v48.18 — OC-TRAC-DUET-BRIDGE (2026-07-30)

### v48.17 result audit and corrected Natural-gate status

- The uploaded package does not contain the main `runs/ocrap_v48_17_bridge_dedicated_4817` directory, so its controller log cannot be inspected directly. However, the main controller invokes the same adaptation script used by the uploaded B/C ablations, and all four B/C logs show completed training followed by the same post-check failure: 78,630 `direct_evidence_calibrators.*` state parameters rejected against a hard-coded maximum of 20,000. The deterministic early-exit path therefore explains the reported missing main-run artifacts.
- `run_v48_17_bridge_dedicated.sh` then exited with code 30 before invoking the certificate controller and before running `check_v48_16_learning_gates.py`. This directly explains the simultaneous absence of `learning_gates_v48_17.json` and `NEXT_COMMANDS.txt`.
- The A_simplex_scalar ablation is the only v48.17 component with a valid, non-empty, scene-disjoint certificate. Both variants were genuinely rejected: Near used 290 groups/123 scenes and Contact 764 groups/215 scenes, but both selected zero verify groups and had zero positive recall.
- B_context_simplex and C_full_bridge cannot be assigned a Natural-gate result because their completed checkpoints were blocked before certificate. Their dev curves are diagnostic only.

### v48.17 algorithm attribution

1. **Retain the frozen top-k recovery proposal and Safe nominal lock.** The proposal continues to expose useful positive candidates, and the 120-scene paired Safe experiment passed all available non-inferiority checks with zero candidate-minus-baseline deltas.
2. **Context contains useful signal, but the v48.17 representation is statistically inefficient.** Relative-context BRIDGE improved Near dev positive recall in several epochs and modestly improved Contact recall in Precision, but it fed roughly 4,890 raw relative features into a 78,630-parameter calibrator despite only 60 deployable-positive adaptation groups.
3. **The three-class simplex correction is not appropriate for the observed target ambiguity.** It forces harm, dead and benefit to compete for unit mass. In Contact, a candidate may carry both benefit evidence and unresolved harm evidence; forcing one tail down can create false-safe decisions.
4. **The advertised batch-balanced objective was only auxiliary.** v48.17 added per-regime/class-balanced loss on top of the original top-1/top-k group ERM, so dead/mixed groups still dominated the primary gradient. The implementation did not match the intended “replace dead-zone-dominated ERM” contract.
5. **Checkpoint selection remained vulnerable to one-regime collapse.** Fold-robust risk could select an epoch with Near improvement but near-zero Contact recall, or vice versa.

### Engineering corrections

1. Replace the fixed 20,000-parameter v48.17 post-check with a configurable architecture-aware cap (`MAX_EVIDENCE_CALIBRATOR_PARAMS`, default 100,000 for recovery of existing BRIDGE checkpoints).
2. Always emit a learning-gate report and controller completion record, including on adaptation failure/exit 30. A missing report is no longer overloaded with “gate failed”.
3. Add `recover_v48_17_after_param_guard.sh` to reuse already-trained v48.17 checkpoints and run the withheld certificate without retraining.
4. Persist the evidence context source and new objective flags in checkpoints/inference configuration.
5. Add tests for identity initialization, tournament-context dimensionality, independent dual tails and cross-regime checkpoint risk. Full local validation: 176 tests passed; Python compileall and all Shell syntax checks passed.

### New algorithm: DUET-BRIDGE

**DUET-BRIDGE = Dual-tail Uncoupled Evidence Transfer with frozen tournament context and balanced target adaptation.**

1. **Frozen tournament context instead of raw relative features.** The evidence calibrator consumes the 48-dimensional contextual recovery embedding already produced by the frozen set tournament. This preserves proposal semantics while reducing the default two-regime calibrator from 78,630 parameters to 1,532 parameters.
2. **Independent benefit and harm residual tails.** A zero-initialized bounded residual is added independently to source benefit and harm logits. The model is no longer forced onto a three-class simplex; ambiguous candidates may have both tails elevated and are conservatively rejected by the harm veto. Nominal rows are explicitly pinned back to zero logits after correction so trained residual biases cannot alter nominal semantics.
3. **Independent-tail supervision.** Beneficial candidates supervise `(benefit=1, harm=0)`, harmful candidates `(0,1)`, and dead-zone candidates `(0,0)` using two BCE losses.
4. **Strict per-regime/per-class balanced replacement.** In calibrator-only adaptation, the minibatch-balanced objective replaces the dead-zone-dominated evidence ERM rather than being added as a weak auxiliary term.
5. **Cross-regime feasibility checkpoint metric.** `direct_duet_selection_risk` adds the minimum Near/Contact recall shortfall and worst-regime harm/false-intervention penalties to the held-out dev certificate risk. This changes only early stopping; the final Natural gate remains unchanged and scene-disjoint.

### Required v48.18 ablations

- A_dual_scalar: independent benefit/harm tails with the four source scalar inputs.
- B_dual_tournament: A plus frozen tournament context.
- C_dual_tournament_balanced: B plus stratified batches and strict balanced replacement.
- D_full_duet: C plus cross-regime feasibility checkpoint selection.

All eight tasks (four groups × Balanced/Precision) launch together. Tasks are assigned round-robin so each A30 runs four low-memory jobs; each task defaults to one DataLoader worker to limit CPU/I/O contention.

### Decision and non-repetition rules

- Do not create `NEXT_COMMANDS.txt` or run test/stress closed loop on exit 20 or 30. Stress remains authorized only by a valid independent Near+Contact certificate.
- Do not repeat the 78,630-parameter raw-context calibrator, simplex-only target correction, “balanced as auxiliary” loss, full Evidence retraining, threshold relaxation, or test-guided tuning.
- If repaired v48.17 returns 0, run its authorized stress experiment before v48.18 and preserve it as a valid comparison. If it returns 20, treat that as a true v48.17 algorithmic rejection and proceed to v48.18 without reading test results.


## v48.14 — OC-TRAC-PRISM (2026-07-29)

### Evidence from the completed v48.13 TERRA experiment

- Neither balanced nor precision passed the joint Near+Contact Natural gate. No v48.13 stress closed-loop result is attributable to the learned policy.
- TERRA's top-k proposal objective was the clearest success: on the main split, positive-group oracle-best hit was about 0.959 Near / 0.970 Contact and any-positive hit was 1.000 / 0.985. The high-recall proposal should be retained.
- Exact top-1 remained weak (Near negative or unstable, Contact only slightly positive), but this is no longer the primary bottleneck once proposal recall is high.
- Proposal evidence did not transfer: Contact harm AUC was approximately 0.39–0.54 and evidence/teacher correlation was near zero or negative. Non-zero Contact selections had low precision, high conditional harm, and negative mean exact-teacher advantage.
- The dedicated calibration diagnostics prove a train-to-target contract shift. Near/Contact calibration roots are far closer to val/test than train in `r_dep_star`, hard violation, candidate count, recoverability, and artifact rate. The legacy `harm_proxy` is non-zero in train but identically zero in calibration/val/test.

### Engineering defects fixed before further attribution

1. **Missing standard calibration artifacts.** Staged v48.13 training used `SKIP_POST_TRAIN_CALIBRATION=1`, so `gamma_rec_by_bucket_v48.json` and the standard calibration JSONs were never produced. v48.14 atomically generates them from the independent certificate pool.
2. **Safe nominal-only dependency bug.** The runner checked gamma and calibration before entering the Safe nominal-lock branch. Safe paired non-inferiority now requires only the checkpoint; Near/Contact stress execution still requires a valid certificate.
3. **Incomplete dedicated recalibration.** The uploaded source run had no completed `dedicated_candidates` artifacts. The new finalizer writes temporary outputs, verifies every required file, atomically installs the calibration directory, and writes `CERTIFICATE_CALIBRATION_COMPLETE.json`.
4. **Invalid v48.13 ablation scheduler.** `GROUPS` is a Bash special array containing Unix group IDs; only `1012_balanced/precision` ran instead of A/B/C/D. The scheduler now uses `ABLATION_SPECS` and requires all eight task markers. Consequently, the uploaded v48.13 ablation cannot support causal algorithm claims.
5. **Ordered-NLL option propagation.** The staged script computed `ORDERED_TOP1/ORDERED_ALL` but passed unrelated fallback defaults to the generic trainer. Parameter names and effective values are now unified.

### New algorithmic contribution: PRISM

**PRISM = Proposal-aligned Risk adaptation with Independent Scene-disjoint certification Model.**

1. **Freeze the proven high-recall proposal policy.** The v48.13 recovery tournament and encoder are frozen. v48.14 does not repeat exact-winner pairwise/listwise attempts that previously degraded Near.
2. **Scene-disjoint calibration-stage evidence adaptation.** Dedicated Near/Contact calibration roots are split by scene into 45% evidence-adaptation train, 15% adaptation dev, and 40% certificate pool. Only the small regime-specific `direct_delta_adapters` are updated. Test roots remain sealed.
3. **Dynamic false-safe hard-harm mining.** Ordered three-state NLL dynamically upweights harmful proposal members that the current adapter predicts as safe, plus a weaker missed-benefit weight to prevent all-abstain collapse. Hardness weights are detached.
4. **Independent certificate pool.** Standard OC-MERO calibration/gamma, policy-rule fit, scene-disjoint verify, and Natural gate are performed only on certificate-pool scenes not used by adaptation or early stopping.
5. **Target-distribution-aligned checkpoint selection.** Adaptation early stopping uses the same top-k evidence-rerank certificate semantics as deployment.

### Required v48.14 ablations

- A: v48.13 frozen checkpoint + dedicated certificate recalibration only.
- B: dedicated target-domain evidence adaptation without dynamic hard mining.
- C: target adaptation + dynamic hard-harm/missed-benefit mining, no same-group pair objective.
- D: full PRISM, adding same-group counterfactual evidence.

Per variant, all four tasks run concurrently: A/C on GPU0 and B/D on GPU1. Balanced and precision remain separate waves to limit CPU and storage contention.

### Decision gates

- Proposal oracle-best/any-positive hit must not materially regress from v48.13.
- Contact policy harm AUC should improve to at least 0.60 and remain directionally consistent between adaptation dev and certificate verify.
- Near benefit AUC should remain at least 0.70; Contact at least 0.75.
- Natural gate still requires non-zero verify coverage, positive mean exact-teacher advantage, unchanged precision/harm confidence bounds, recall/support, and opportunity-normalized macro constraints.
- Stress closed loop is allowed only when `NEXT_COMMANDS.txt` is generated from the independent dedicated certificate pool.

### Non-repetition note

Do not repeat all-pairs recovery ranking, cross-scene bipolar evidence as the primary harm objective, conformal calibration on a non-discriminative evidence model, threshold relaxation, absolute macro caps, or full train-set reconstruction at this stage. PRISM reuses the successful top-k proposal and specifically targets the empirically proven evidence-domain shift while preserving an independent statistical certificate.


## v48.13 — OC-TRAC-TERRA (2026-07-29)

### Evidence from the completed v48.12 TRIDENT experiment

- Neither balanced nor precision passed the joint Near+Contact Natural gate, so no OC-RAP stress closed-loop result is attributable to v48.12.
- Under the correct policy-first/no-fallback contract, three-seed recovery ranking remained asymmetric: Near group top-1 correlation was negative for both variants on average (about -0.054 balanced and -0.035 precision), while Contact was consistently positive but insufficient (about 0.077 balanced and 0.101 precision, versus the internal 0.20 readiness target).
- Contact benefit detection remained strong (candidate-positive AUC about 0.82 and policy-top1 benefit AUC near 0.80), but harmful-vs-dead evidence did not transfer across scenes. Fit rules with positive mean exact-teacher advantage collapsed on verify to high harmful rates and negative mean advantage.
- Near near-miss rules were sparse but sometimes safe: selected groups could have positive mean exact-teacher advantage and no harmful actions, yet support, Wilson precision lower bounds, recall, and cross-seed stability were below the Natural gate.
- External baselines establish the eventual closed-loop bar. Safe is dominated by nominal/log replay non-intervention. Near predictive-safety filtering offers a strong DRS/FRA/ODG/NUP trade-off. Contact restoration/MPC baselines recover more aggressively but pay substantial intervention and NUP cost. v48.12 has no gate-authorized closed-loop result and therefore has not surpassed these baselines.

### Causal conclusions from the complete v48.12 ablation

1. **The standalone recovery-set tournament remains useful but exact winner supervision is underidentified.** Candidate rank correlation is positive, especially in Contact, but it does not reliably become exact top-1. The v48.12 all-pairs teacher-gap loss degraded Near and did not materially improve Contact.
2. **Bipolar cross-group evidence is not a sufficient harm solution.** It improved some Near harm AUC values, but Contact harmful-vs-dead discrimination remained near random and selected verify actions retained negative average teacher advantage. Cross-scene pairwise losses can exploit scene severity and are noisy under minibatch sampling.
3. **Opportunity-normalized macro support is an engineering-correct certificate, not the current bottleneck.** Precision/harm transfer fails before macro excess becomes decisive.
4. **Threshold relaxation remains contraindicated.** Natural-gate rejection is consistent with the observed harmful verify actions.

### Engineering defects fixed before v48.13 attribution

1. **Parent-controller policy-contract loss.** The staged child process exported policy-first/no-fallback internally, but the parent calibration process reverted to default false values. The v48.12 main run therefore calibrated a different selection contract than its multi-seed run. Every staged variant now writes `POLICY_CONTRACT.env`, and the controller sources it before calibration.
2. **Checkpoint-selection/deployment mismatch.** Stage-E early stopping evaluated only the tournament rank top-1, while TERRA deploys evidence reranking within a frozen top-k proposal. Validation now uses the same proposal and evidence-reranking candidate, including certificate regret, harm, false intervention, recall, and evidence margin.
3. **Calibration/runtime contract propagation.** Proposal size and evidence-rerank semantics are now stored in calibration JSON selector overrides and consumed by offline and closed-loop selectors.
4. **Dedicated and multi-seed recalibration parity.** Both scripts source the immutable per-variant contract and use the same support, conditional-harm, macro, and proposal settings as the main run.
5. **Legacy packaging regression.** The missing historical v47 orchestration file required by the existing regression suite was restored; all historical tests now execute.

### New algorithmic contribution: TERRA

**TERRA = Top-k Evidence-Reranked Recovery with Abstention.**

1. **Set-valued recovery proposal**
   - Retains the independent permutation-equivariant recovery tournament.
   - Replaces noisy all-pairs exact-winner supervision with a differentiable top-k inclusion objective: at least one exact-teacher acceptable recovery must enter the proposal.
   - Proposal quality is measured on positive-opportunity groups by oracle-best hit rate and any-positive hit rate, separately from exact top-1 correlation.

2. **Proposal-distribution ordinal evidence**
   - Freezes the tournament and trains regime-specific ordered harmful/dead/beneficial evidence on every member of the actual top-k proposal, with rank-decayed weights.
   - This removes the v48.12 mismatch in which only rank-1 evidence was trained although useful or harmful runner-up candidates determined failure analysis.

3. **Same-group counterfactual evidence**
   - Adds beneficial-vs-nonbeneficial and harmful-vs-nonharmful comparisons only within the same scene-time proposal.
   - Shared scene severity cancels in these comparisons, reducing the train/dev shortcut that harmed Contact cross-scene transfer.
   - The v48.12 cross-group bipolar pair loss is disabled in the TERRA main experiment.

4. **Evidence reranking with abstention**
   - Runtime order is: physical recovery candidates → frozen rank top-k proposal → evidence thresholds → choose the highest evidence member within the proposal → abstain if none passes.
   - This is not an out-of-distribution runner-up fallback because Stage E explicitly trains all proposal members.
   - The same evidence score and margin are used by checkpoint selection, calibration, offline evaluation, and closed loop.

### Required layered validation

1. **Proposal gate:** on positive-opportunity groups, proposal oracle-best hit rate should be at least 0.75 and any-positive hit rate at least 0.90 in Near and Contact. Exact top-1 remains diagnostic rather than the sole Stage-P success condition.
2. **Proposal-evidence gate:** evidence-reranked proposal top-1 benefit AUC should reach at least 0.70 Near / 0.75 Contact, harm AUC at least 0.60, and evidence/teacher correlation at least 0.10.
3. **Natural gate:** only non-zero held-out coverage with positive mean exact-teacher advantage, unchanged precision/harm confidence bounds, recall/support, and macro-excess constraints may authorize stress closed loop.
4. **Multi-seed:** run 4801/4802/4803 only on an immutable checkpoint after proposal/evidence diagnostics are promising.
5. **Closed-loop comparison:** Safe must remain nominal-noninferior; Near must improve safety/recovery relative to predictive filtering without excessive intervention; Contact must approach restoration/MPC recovery while materially improving intervention and NUP trade-offs.

### Required ablations and GPU scheduling

- A: top-1 contract baseline.
- B: top-k proposal training only, deployment remains top-1.
- C: proposal-distribution evidence and evidence reranking on the old tournament.
- D: full TERRA.

The v48.13 scheduler runs all four groups concurrently per variant wave. A/C share GPU0 and B/D share GPU1, permitting two approximately 1-GB jobs per A30 while preserving separate processes and outputs. Balanced and precision waves remain sequential to limit host I/O contention.

### Non-repetition note

Do not repeat v48.12 all-pairs recovery ordering, cross-group bipolar evidence as the main harm objective, threshold relaxation, absolute macro caps, inherited value residual ranking, or untrained runner-up fallback. TERRA changes the identifiable policy object from a noisy exact winner to a small recovery proposal and aligns evidence training with every candidate that deployment may execute.


## v48.12 — OC-TRAC-TRIDENT (2026-07-28)

### Evidence from the completed v48.11 CASTER experiment

- No balanced or precision candidate passed the joint Near+Contact Natural gate; stress closed loop was correctly withheld.
- The standalone recovery set tournament produced a real but incomplete Contact ranking gain. Across calibration seeds 4801/4802/4803, balanced Contact top-1 correlation was 0.0735/0.0850/0.1026 (mean 0.0870), and precision was 0.0208/0.0455/0.0777 (mean 0.0480). Near remained approximately zero or negative.
- Candidate recovery signal remained strong, especially Contact (three-seed candidate-positive AUC about 0.829 balanced and 0.813 precision), but policy-top1 harm discrimination remained weak and unstable (Contact candidate-harm AUC about 0.536).
- Balanced Near exposed a useful near-miss: a verify rule selected 10 groups with 0.70 point precision, no harmful selections, 0.28 positive recall, and +0.146 mean exact-teacher advantage. It was rejected partly because every selected action was macro 5. However, the teacher-positive training distribution itself was about 88% macro 5, so the old absolute 0.85 macro-share constraint confounded policy shortcut with opportunity support.
- Contact fit-to-verify transfer remained the main safety failure: representative fit rules had positive mean advantage and moderate coverage, but verify precision collapsed and harmful rate increased sharply.

### Engineering defects fixed before further algorithm attribution

1. **Conditional checkpoint semantic mismatch.** v48.11 trained a recovery-only tournament, but `PREFERENCE_CONDITIONAL_MODE` was not enabled in the staged script. Early stopping therefore penalized nominal false-switch terms even though nominal was not part of the tournament and centered recovery scores make one recovery positive by construction. v48.12 sets this flag explicitly.
2. **Ablation scheduler fail-fast bug.** Under `set -e`, failure of the first `wait` stopped the entire suite. The uploaded ablation package therefore lacked C-precision and D-precision and could not isolate all CASTER modules. The new scheduler records failures, continues all eight tasks, and creates `ABLATIONS_COMPLETE.json` only when all tasks finish.
3. **Macro certificate contract.** The absolute selected-macro cap is replaced in the main v48.12 experiment by an opportunity-normalized excess concentration: selected concentration is penalized only when it exceeds the exact-teacher positive-policy concentration by more than a configured allowance. Raw macro share remains reported.

### New algorithmic contribution: TRIDENT

**TRIDENT = Teacher-gap Recovery tournament with Inter-regime Discriminative Evidence and Normalized-support cerTification.**

1. **Teacher-gap recovery-pair tournament**
   - Retains the recovery-only, permutation-equivariant set tournament that improved Contact.
   - Adds exact-PCD gap-weighted pair supervision only when a recovery pair is materially ordered.
   - Near ties below the configured gap remain unordered, avoiding artificial winner labels; clear pairs receive direct top-1 gradients.

2. **Bipolar cross-group ordinal evidence**
   - Retains the proper harmful/dead-zone/beneficial ordered simplex and frozen policy-top1 training distribution.
   - Adds regime-local cross-group pairwise AUC surrogates for beneficial-vs-nonbeneficial and harmful-vs-nonharmful policy selections.
   - Harm separation receives the larger weight because Contact verify harm inversion, rather than benefit detection, is the current certification bottleneck.

3. **Opportunity-normalized support certificate**
   - Reports both raw macro concentration and oracle-positive macro concentration.
   - The deployability constraint uses positive-policy excess concentration by default in TRIDENT experiments, preventing an impossible diversity requirement when the available teacher opportunities are intrinsically concentrated.
   - This is not threshold relaxation: precision, harmful-switch, support, recall, positive mean advantage, and scene-disjoint verification requirements are unchanged.

4. **Layered experimental attribution**
   - A: conditional-contract fix only.
   - B: recovery-pair tournament.
   - C: bipolar evidence.
   - D: full TRIDENT.
   - All eight variant tasks are attempted with at most two concurrent single-GPU jobs.

### Required validation order

1. Stage R: Near top-1 correlation must become consistently positive and Contact should exceed the v48.11 mean; inspect exact regret as well as correlation.
2. Stage E: policy-top1 benefit AUC should be at least 0.70 Near / 0.75 Contact, and harm AUC at least 0.60 in both regimes.
3. Certificate: non-zero verify selections must have positive mean exact-teacher advantage, precision LCB and harmful UCB within the unchanged Natural gate, and macro excess within budget.
4. Only a candidate passing both Near and Contact may enter stress closed loop. Safe paired non-inferiority remains a separate experiment.

### Non-repetition note

TRIDENT does not repeat threshold relaxation, handwritten recovery rules, the inherited value-plus-residual rank, global conformal saturation, or a shared harm classifier. It deepens the only v48.11 component with positive evidence (the standalone set tournament), directly optimizes the failing cross-group harm tail, and corrects support certification relative to the opportunity distribution.

## v48.11 — OC-TRAC-CASTER (2026-07-28)

### Evidence from the completed v48.10 COPE experiment

- The main v48.10 run and all eight causal ablation tasks completed. Neither variant passed the joint Near+Contact Natural gate, so no stress closed-loop result is attributable to COPE.
- Monotone ordinal evidence was the only consistently useful module. Relative to the A reference, the C evidence ablation raised Contact candidate-positive AUC from 0.724/0.670 to 0.806/0.826 for balanced/precision and improved Contact harm AUC to 0.583/0.597. The full model reached Contact candidate AUC 0.834/0.814 and policy-top1 benefit AUC 0.808/0.768.
- Conditional Option Preference did not solve groupwise ranking. Main Near/Contact top-1 correlation stayed around zero (-0.003/0.012 balanced, 0.013/0.001 precision). The conditional objective also increased non-positive false switches to 0.37-0.53 and harmful ranked switches to 0.14-0.19.
- Precision Contact was the only main result with non-zero verify coverage: 13/193 groups, precision 0.308 (LCB90 0.146), harmful rate 0.462 (UCB90 0.675), recall 0.121, mean exact-teacher advantage -0.180, and macro concentration 0.846. It is not deployable.
- The inherited candidate value remained inside the preference score as `rank_base + residual`. The trained residual changed the chosen recovery in only a small fraction of groups, so the high-AUC but groupwise-wrong candidate value continued to dominate option ordering.

### Engineering correction: policy-first, no-fallback semantics

Stage-E evidence was trained on the frozen preference top-1 candidate, but calibration and runtime first filtered candidates by evidence and then selected the highest-ranked survivor. A failed top-1 could therefore fall through to a runner-up that Stage E never trained on. v48.11 makes the policy contract explicit and identical everywhere:

1. rank physically supported recovery candidates;
2. choose the preference top-1 candidate;
3. evaluate rank margin and evidence only for that candidate;
4. abstain if it is uncertified; never fall through to rank 2.

The calibration JSON now records `direct_value_policy_first_no_fallback`, and the runtime selector exposes the same option.

### New algorithmic contribution: CASTER

**CASTER = Conditional Attention Set Tournament with Evidence Routing.**

1. **Recovery-only set tournament**
   - Replaces, rather than residualizes, the inherited candidate-level value ranking.
   - Uses a small permutation-equivariant self-attention tournament over nominal-relative recovery tokens.
   - Nominal is pinned to zero and excluded from the tournament; admission is isolated in Stage E.
   - Group scores are centered to remove an unidentifiable common offset.

2. **Policy-conditioned regime evidence**
   - Stage E freezes the complete set tournament.
   - Separate Near and Contact evidence experts consume nominal-relative candidate features plus the frozen policy score and top1-vs-runner-up gap.
   - The evidence model therefore learns the distribution actually encountered by each regime rather than averaging incompatible harm/dead boundaries.

3. **Proper ordered three-state likelihood**
   - The ordered logits induce a valid simplex: harmful, dead-zone, beneficial.
   - A class-weighted three-class NLL replaces two independent focal BCE losses.
   - Harmful examples receive the largest weight because harm-vs-dead separation is the current certification bottleneck.

4. **Strict attribution and speed**
   - Added immutable staged architecture/checkpoint contracts.
   - Added dynamic ablation summarization and a calibration-only policy-semantics ablation.
   - Four ablations are queued together with at most one model per A30; the policy-first ablation reuses the reference checkpoint and avoids duplicate training.

### Required validation order

1. Stage T: Near and Contact recovery-only top-1 correlation should both exceed 0.10 before evidence results are interpreted.
2. Stage E: policy-top1 benefit AUC should reach at least 0.70 Near / 0.75 Contact and harm AUC at least 0.60 in both regimes.
3. Natural gate must produce non-zero verify coverage with positive mean exact-teacher advantage and unchanged confidence bounds.
4. Run seeds 4801/4802/4803 only after stages 1-2 pass.
5. Run stress closed loop only when the controller creates `NEXT_COMMANDS.txt`.

### Non-repetition note

CASTER does not repeat threshold relaxation, another residual MLP on top of the candidate value, independent harm BCE, or evidence-first runner-up fallback. Its novelty is the combination of recovery-only set competition, policy-conditioned regime evidence, ordered three-state certification, and one consistent no-fallback policy contract.

## v48.10 — OC-TRAC-COPE (2026-07-27)

### Evidence from the completed v48.9 PACER experiment

- The main v48.9 run and fixed-checkpoint calibration seeds 4801/4802/4803 completed. Neither variant passed the scene-disjoint Near+Contact Natural gate, so the controller correctly did not create `NEXT_COMMANDS.txt`; no Near/Contact stress closed-loop result is attributable to PACER.
- Candidate evidence remained detectable, but policy quality did not improve enough. Main policy-top1 benefit AUC was about 0.67/0.74 for balanced Near/Contact and 0.67/0.73 for precision, while policy-top1 harm AUC was only about 0.49–0.57. Group top-1 correlation remained near zero in Near and slightly negative in Contact.
- The closest Near fit rules could reach nominal precision around 0.75 on only eight selected groups, but verify precision fell to roughly 0.38–0.50, recall to about 0.08–0.12, and selected actions were concentrated in macro 5. Contact fit-to-verify transfer was worse and could become predominantly harmful with negative mean teacher advantage.
- Policy-top1 conformal calibration did not solve certification. One-sided overprediction quantiles remained approximately 0.60–0.62, forcing zero verified coverage. Exact teacher advantage is strongly tri-modal (`harmful/dead-zone/beneficial`, with many exact zeros and boundary masses), whereas the continuous delta regressor collapsed near zero.
- The uploaded ablation suite was incomplete: only balanced A/B/C artifacts were available. More importantly, Stage C for A/B instantiated a 128-wide preference context while Stage P used width 32, causing preference-adapter checkpoint shape mismatches and discarding learned Stage-P context. Final A/B attribution is therefore invalid even though the main run did not contain this mismatch.

### What v48.9 established

- **Intervention-aware preference is useful for suppression but not sufficient for ranking.** Relative to the old nominal-inclusive objective, the Stage-P audit reduced non-positive false switches from about 0.65–0.71 to 0.12–0.15 and harmful ranked switches from about 0.21–0.23 to 0.05–0.07. However, Contact conditional recovery ordering regressed to near zero/negative correlation.
- **Policy-aligned certificate sampling is directionally useful.** Compared with all-candidate training, the available balanced ablation improved policy-top1 benefit AUC, especially in Contact, but harm discrimination remained near random and no rule passed verification.
- **Conformal calibration is not a substitute for a discriminative evidence model.** Correcting the conformal sampling scope cannot rescue a regressor whose residual scale is comparable to the full teacher-advantage range.

### Engineering corrections

1. Added `training.strict_init_prefixes`. Stage E now aborts unless the complete Stage-P preference adapter loads with exactly matching geometry; silent loss of learned preference context is forbidden.
2. The staged architecture writes `STAGE_ARCHITECTURE.json`, and completion markers include immutable preference/evidence checkpoint hashes.
3. The v48.10 ablation controller propagates the same preference width to both stages, creates one immutable `TASK_COMPLETE.json` per task, resumes completed tasks, and refuses to write the suite summary until all eight `(4 ablations × 2 variants)` tasks exist.
4. Calibration, checkpoint metrics, offline evaluation, selector semantics, and closed-loop execution now share the same conditional-recovery ranking and ordinal-evidence interpretation.

### New algorithmic contribution: COPE

**COPE = Conditional Option Preference with Monotone Ordinal Evidence.** It separates the two logically different questions that PACER still mixed inside the preference target and continuous certificate.

1. **Conditional Option Preference (COP)**
   - Stage P ranks recovery options only; nominal is excluded from the option-ordering loss and from the conditional rank margin.
   - Exact teacher-PCD defines an ambiguity-aware acceptable recovery set. The loss maximizes mass on that set and minimizes exact expected recovery regret.
   - Positive-opportunity groups receive full weight. No-opportunity and harmful groups receive a lower weight and teach only the least-bad recovery ordering; whether any recovery should be executed is deferred entirely to Stage E.
   - This preserves the experimentally useful nominal-relative low-capacity context while preventing nominal-suppression gradients from destroying Contact recovery ordering.

2. **Monotone Ordinal Evidence (MOE)**
   - Stage E freezes the complete preference policy and models the frozen policy-top1 candidate as one of three ordered states: beneficial, dead-zone, or harmful relative to nominal.
   - Two ordered cumulative logits parameterize `P(beneficial)` and `P(non-harm)`, with the architecture enforcing `P(beneficial) <= P(non-harm)` and `P(harm)=1-P(non-harm)`.
   - Focal ordinal supervision is concentrated on policy-top1 candidates, with only a weak all-candidate regularizer. This matches the deployment distribution and the tri-modal exact teacher target without regressing advantages toward zero.
   - Admission uses opportunity probability, harm probability, evidence score, conditional recovery rank margin, support, recall, and macro concentration under the unchanged fit/verify Natural gate. Thresholds are not relaxed.

### Required causal ablations

1. A: v48.9-style nominal-inclusive preference + continuous delta evidence.
2. B: conditional option preference + continuous delta evidence.
3. C: nominal-inclusive preference + monotone ordinal evidence.
4. D: full COPE.

The first attribution question is whether B improves conditional recovery top-1 and regret without relying on nominal switching. The second is whether C/D improve policy-top1 benefit and harm AUC and create transferable non-zero verification coverage. Multi-seed and stress closed loop remain forbidden until the fixed checkpoint passes the diagnostic learning gates and unchanged Natural gate.

### Local validation

- 144 tests passed.
- Python compileall passed.
- The main controller and all modified v48.10 shell scripts passed `bash -n`.
- Real WOMD/Waymax/A30 training and closed-loop evaluation are not available locally; COPE is an experimentally testable design, not a claim that the publication thresholds have already been reached.

## v48.9 — OC-TRAC-PACER (2026-07-27)

### Evidence from the completed v48.8 experiment

- The v48.8 main run, eight ablation jobs, and the paired Safe probe were audited. Natural gate failed for both variants and both stress regimes; every calibrated rule selected zero actions, so no Near/Contact closed-loop improvement can be attributed to SCOPE.
- Candidate-level signal remained usable (main candidate-positive AUC: Near 0.643–0.730, Contact 0.765–0.785), but policy ordering remained insufficient. Main top-1 correlation was 0.053/0.125 for balanced Near/Contact and 0.163/0.185 for precision Near/Contact, below the internal 0.20 readiness target.
- The conflict-free preference ablation did not improve top-1 over the engineering-fixed reference. Near/Contact correlation changed from 0.022/0.079 to 0.006/0.058 for balanced and from 0.048/0.102 to 0.006/0.058 for precision.
- The split-conformal certificate was over-conservative for the wrong reason: residuals were fitted over every recovery candidate, although deployment evaluates only the frozen preference top-1 candidate. One-sided overprediction quantiles were about 0.57–0.61, causing all scored rows to saturate at opportunity=0 and harm=1; no strict threshold grid or near-miss frontier existed.
- The paired Safe probe used only eight scenes. Collision/offroad, bounded NUP, and intervention were identical to nominal, but route progression, jerk p95, and yaw-rate p95 were unavailable. It is diagnostic only and not a paper-ready Safe claim.

### Root-cause corrections

1. **Partial-label set mass rather than uniform-set KL.** The v48.8 target forced all acceptable candidates to equal logits. PACER minimizes negative probability mass on the acceptable set, preserving ambiguity without inventing an ordering inside the set.
2. **Nominal-only target for no-opportunity groups.** Dead-zone recoveries are no longer treated as equally acceptable deployment actions. They receive a weak intervention-cost margin below nominal; materially harmful recoveries receive a stronger margin.
3. **Policy-induced certificate training.** Stage C now trains the relative-gain head strongly on the recovery candidate actually selected by the frozen Stage-P preference policy, while retaining a low-weight all-candidate regularizer. This removes the train/deploy distribution mismatch.
4. **Policy-top1 conformal scope.** Optional conformal calibration fits residuals on one frozen-policy candidate per group, not all unused candidates. The default main experiment uses empirical direct-delta admission; conformal remains a controlled ablation until it demonstrates non-zero verified coverage.
5. **Non-empty failure diagnostics.** Calibration now writes a diagnostic frontier even when all predicted opportunity/harm values violate hard bounds, while explicitly marking probability-bound deficits so diagnostic rows cannot pass the Natural gate.

### New algorithmic contribution: PACER

**PACER = Policy-Aligned Candidate Evidence for Recovery.** It couples two isolated stages through the policy-induced candidate distribution rather than through shared gradients.

- **Intervention-aware partial-label preference:** Stage P uses only nominal-relative set context. Positive groups maximize mass on the exact-teacher equivalent recovery set; no-opportunity groups choose nominal; dead-zone and harmful alternatives are separated by different margins.
- **Policy-aligned evidence:** Stage C freezes the whole preference path and learns exact candidate-minus-nominal PCD gain on Stage P's selected candidate, with smooth-L1 and tri-state sign supervision. The all-candidate loss is only a weak representation regularizer.
- **Auditable abstention:** empirical fit/verify precision, conditional harmful-switch bounds, rank margin, recall, support, and macro concentration remain unchanged. PACER does not lower gate thresholds to manufacture coverage.

### Required validation and ablations

1. Old uniform-set preference + old all-candidate certificate.
2. Intervention-aware set-mass preference only.
3. Policy-aligned certificate only.
4. Full PACER.

The first learning checkpoint is whether Near and Contact top-1 correlation improve without increasing non-positive false switches. The second is whether policy-top1 gain AUC and near-miss precision improve. Multi-seed and stress closed loop remain forbidden until a fixed checkpoint passes the held-out Natural gate.

### Engineering and speed notes

- Main and ablation runs may reuse the v48.8 proxy split and exact teacher-PCD index; no dataset rebuild is required.
- Four ablations are submitted together but the scheduler runs at most one job per A30, so two jobs execute concurrently without GPU oversubscription.
- The Stage-P adapter width is reduced from 48 to 32, Stage C uses only the delta adapter, BF16/TF32 and persistent data workers remain enabled, and calibration uses group-batched inference.

### Local validation

- 140 tests passed.
- Python compileall passed.
- Modified shell scripts passed `bash -n`.
- Real WOMD/Waymax/A30 results are not available in the local environment; no claim is made that v48.9 already passes Natural gate or closed-loop publication targets.

## v48.8 — OC-TRAC-SCOPE (2026-07-27)

### Evidence from the completed v48.7 proxy experiment

- Stage P did not reliably learn policy top-1. Candidate-level rank correlation was positive (about 0.12–0.16), but unconstrained group top-1 correlation remained slightly negative in both Near and Contact. Acceptable-set accuracy was only about 0.53–0.64 and positive-group regret remained 0.12–0.19.
- Stage C did not learn deployable execution evidence. Candidate-positive AUC was 0.66–0.77 and risk-harm AUC only 0.55–0.61. Every Natural-gate rule selected zero actions. The closest rules had low precision, high conditional harmful-switch UCB, negative or near-zero mean teacher advantage, and 0.85–1.00 maximum macro share.
- Ablations isolate two partial ideas: staged training improves Contact relative to joint single-winner training, and set-valued supervision improves both regimes under joint training. Their v48.7 combination regressed because the loss simultaneously treated near-tied candidates as equivalent and as ordered best-vs-rest competitors, while checkpoint selection was dominated by a sparse fold.
- The Safe probe used only eight nominal-locked scenes. It confirms zero intervention and NUP=1 for that probe, but does not establish paired collision/offroad non-inferiority, confidence intervals, route progression, jerk/yaw-rate, or the publication Safe target.

### Engineering corrections required for clean attribution

- Checkpoint improvement is now strict with `training.best_metric_min_delta`; equal validation metrics no longer overwrite the earlier best checkpoint.
- Fold selection is support-aware. Preference/certificate checkpointing ignores folds below a configurable positive-group floor and uses the mean of the worst supported K folds instead of a noisy single-fold maximum.
- Preference risk now includes harmful top-1 and non-positive nominal-switch penalties, rather than evaluating only positive-group regret.
- Training summaries retain the exact `trainable_param_prefixes` and metric tolerance.
- Calibration always writes unconstrained top-1 diagnostic rows even when no rule passes and evaluates each near-miss fit rule on the held-out verify fold.
- Proxy splits and exact teacher-PCD indexes can be prepared once and reused by all ablations. The controller supports one-variant jobs, shared assets, and a two-GPU queue without oversubscribing either A30.
- Safe evaluation can now run a scene-paired scalar/nominal reference and the nominal-locked model on the two A30s, then report paired bootstrap non-inferiority intervals. A duplicate-loop syntax error in the legacy summary block was also fixed; route/jerk/yaw metrics are marked unavailable rather than silently proxied.

### New algorithm: SCOPE

**SCOPE = Support-aware Conflict-free Ordinal Preference with Conformal Evidence.**

1. **Conflict-free nominal-inclusive set preference**
   - Every scene-time group is supervised, not only groups with a positive recovery opportunity.
   - Material-positive groups target a teacher-equivalent recovery set; no-opportunity groups target nominal plus only dead-zone alternatives; harmful recoveries are explicitly pushed below nominal.
   - When enabled, this objective replaces the contradictory single-winner/listwise family instead of being added on top of it.

2. **Invariant low-capacity preference context**
   - The trainable Stage-P adapter receives only candidate-minus-nominal, recovery-mean, and recovery-max relative blocks. Absolute candidate features are excluded to reduce severity/macro shortcuts under train/dev contract drift.
   - Only the context residual is trained; inherited pointwise preference remains frozen. Hidden width is reduced to 48 and the residual remains zero-initialized.

3. **Robust relative-gain learning**
   - Stage C trains only the relative delta adapter with smooth-L1 regression and soft positive/harm sign supervision. Heteroscedastic NLL is disabled, preventing the severe train-negative/validation-positive variance-collapse pattern observed in v48.7.
   - The delta log-variance remains at a fixed conservative initializer and is not treated as learned epistemic confidence.

4. **Split-conformal execution evidence**
   - On the calibration fit scenes, one-sided finite-sample residual quantiles form a lower confidence bound for candidate-minus-nominal gain. Rule search and held-out verification use this bound.
   - Selector, offline evaluator, and closed-loop runner consume the same conformal quantile and score semantics if a rule passes.

### Stepwise validation protocol

- Stage-P audit must first show positive Near/Contact top-1 correlation and acceptable-set accuracy before certificate results are interpreted.
- Stage-C discrimination then checks positive/harm AUC and regret independently of Natural-gate coverage.
- Only after both pass is fixed-checkpoint multi-seed calibration run; stress closed loop remains forbidden unless the held-out Natural gate is valid.
- Four controlled groups are required: engineering-fixed v48.7 reference, conflict-free preference only, robust/conformal certificate only, and full SCOPE. The queue runs at most two jobs concurrently on two A30 GPUs.

### Non-repetition note

SCOPE does not repeat stronger Harm-head weighting, joint preference/certificate gradients, minibatch GroupDRO, threshold relaxation, shared NASC, or another additive single-winner ranking loss. It specifically removes contradictory supervision, absolute-feature shortcuts, learned-variance collapse, and sparse-fold checkpoint noise exposed by the completed v48.7 experiment.

## v48.7 — OC-TRAC-SPIRE (2026-07-26)

### Evidence from the completed v48.6 experiment

- All v48.6 training, three-seed recalibration, and four core ablations were audited.
- Candidate recovery signal remained usable (three-seed mean AUC: Near 0.702–0.706, Contact 0.818–0.824), but the policy layer regressed: Near top-1 correlation was -0.039 to -0.002 and Contact was -0.087 to -0.054; every verify fold selected zero actions.
- The only positive ablation was preference-only relative context: it improved rank correlation and made balanced Contact top-1 slightly positive on seed 4801. Direct-delta-only and the full joint RPGC objective worsened Contact ordering, demonstrating negative transfer from certificate learning into the shared preference representation.
- Rank-margin correctness AUC was informative only for Contact (about 0.63–0.65) and weak for Near (about 0.42–0.43). The direct-delta/harm channel remained insufficiently transferable (risk-harm AUC about 0.55–0.58).

### Engineering corrections before further algorithm attribution

- Validation checkpointing now computes opportunity/harm from the same Gaussian direct-delta CDF used by calibration; the v48.6 raw-delta threshold was not deployment-equivalent.
- Harmful *population exposure* UCB and conditional harmful-switch UCB among selected actions are now reported and constrained separately. A low exposure UCB caused by zero/rare selections can no longer be mistaken for a low harmful-switch rate.
- `run_ocrap_v48_trac_sr.sh` no longer silently falls back to the obsolete `runs/ocrap_v48_trac_sr_regime_balanced` path. `BASE_RUN` or explicit checkpoint/calibration paths are mandatory.
- Natural-gate failure writes `GATE_FAILED.json` and exits before producing closed-loop commands. It does not imply that the trained candidate checkpoints are missing.
- Added a Safe-only nominal-locked probe that does not require Near/Contact certificates and cannot authorize stress-regime intervention.
- Added partial dedicated-calibration merge support so completed Safe/Near worker pairs can be filtered and atomically installed under the evaluation root before Contact is finished.
- Teacher-PCD data-quality reports now distinguish all-macro opportunities from deployable-macro opportunities; quality gates use the actual selector allowlist.
- Added parameter allow-list freezing for auditable staged optimization.

### New algorithm: SPIRE

**SPIRE = Set-valued Preference with Isolated Relative-gain Evidence.** It explicitly separates the three policy objects that v48.6 failed to identify jointly.

1. **Preference stage — who should be selected?**
   - The encoder and value surface are frozen. Only the pointwise and nominal-relative preference residuals are trainable.
   - Exact teacher-PCD supervision is ambiguity-aware: Near and Contact use regime-specific acceptable sets around the teacher optimum rather than forcing arbitrary single winners in near-tied groups.
   - The loss combines acceptable-set KL, set-versus-nominal/worse-candidate margin, confidence-paced best-vs-rest preference, exact expected regret, and a small rank-gap term.
   - Early stopping uses worst-fold tie-aware preference risk, not candidate AUC or total loss.

2. **Certificate stage — should the selected recovery be executed?**
   - The complete preference path, encoder, and value heads are frozen. Only the direct candidate-minus-nominal delta adapter is trained.
   - This removes the v48.6 negative transfer in which delta-NLL gradients degraded Contact ordering.
   - Early stopping uses a fixed deployment-aligned certificate risk based on direct-delta opportunity probability, harm probability, rank margin, harmful admitted actions, false interventions, and missed positive opportunities. Always-abstain therefore no longer receives a deceptively good checkpoint score.

3. **Evidence and gate semantics**
   - Calibration keeps exact preference top-1 and direct-delta admission separate.
   - It reports both strict single-winner accuracy and acceptable-set accuracy/tie-aware regret.
   - Proxy calibration may use development conditional-harm UCB limits; paper promotion requires the larger dedicated set and substantially tighter conditional harmful-switch bounds.

### Required ablations

1. Joint single-winner v48.6 objective.
2. Staged optimization with the old single-winner preference target.
3. Joint optimization with set-valued preference.
4. Full SPIRE: staged optimization plus set-valued preference.

This design isolates whether gains come from ambiguity-aware supervision, gradient isolation, or their combination.

### Closed-loop promotion requirements

- Stress closed loop remains forbidden unless both Near and Contact certificates pass.
- First screening target: all three seeds positive top-1 correlation, mean >=0.10, non-zero verify selections, and no uncontrolled conditional harmful-switch UCB.
- Paper-readiness remains stricter: top-1 correlation >=0.20, precision LCB90 >=0.60, positive recall >=0.35, conditional harmful-switch rate/UCB approaching the 5–10% target on a sufficiently large dedicated calibration set, and scene-paired closed-loop improvements without Safe degradation.

### Non-repetition note

SPIRE does not repeat joint value/rank/delta optimization, stronger Harm-head weighting, ordinary single-winner listwise training, GroupDRO, threshold relaxation, or handwritten rescue rules. The new contribution is ambiguity-aware set preference plus stage-isolated evidence certification under a deployment-aligned checkpoint and Natural gate.

# OC-RAP Algorithm Changelog

## v48.6 — OC-TRAC-RPGC (2026-07-26)

### Evidence from the completed v48.5 experiments

The completed main run, four controlled ablations, and fixed-checkpoint calibration seeds
4801/4802/4803 change the diagnosis from “ranking is uniformly broken” to a more specific
two-stage failure. The independent ECPR preference design is directionally valid: Contact
within-group top-1 correlation is positive for both variants on every calibration seed
(approximately 0.124–0.174), whereas v48.4 Contact ranking was negative on every seed.
Near remains split-sensitive (-0.092 to 0.062 balanced and -0.070 to 0.028 precision), and
no calibrated rule selects a recovery action. Candidate-positive AUC remains useful but is
not sufficient (multi-seed mean about 0.682/0.728 for Near and 0.781/0.796 for Contact).
The calibrated downside AUC is only moderate (roughly 0.54–0.62), and the closest Contact
rules still have low precision, high harmful selection, and >0.93 single-macro share.

The ablations identify the source of the ranking improvement. `C_exact_ecpr` is the only
standalone module that makes Contact top-1 correlation positive. Legacy NASC alone does not
improve ranking, and combining shared NASC with ECPR restores candidate AUC but drives
policy top-1 back toward zero or negative. Therefore v48.6 retains independent preference
learning and removes legacy shared-feature set context from the main path.

### Engineering conclusions and fixes

- Verified that the executed v48.5 configuration used `direct_value_output_mode=score` in
  both validation and calibration. A suspected raw-logit/probability mismatch was ruled out
  as the root cause; the generic validation path is nevertheless made mode-explicit.
- The v48.5 “delta distribution” subtracts two absolute value predictions and adds their
  variances as if their errors were independent. Candidate and nominal share the same scene
  encoder, so this approximation can overestimate uncertainty and collapse opportunity/harm
  gates to zero coverage.
- Calibration now enforces the macro-concentration constraint on the fit fold, not only as a
  held-out warning. Near-miss optimisation also includes macro-share deficit.
- Added explicit rank-margin abstention to calibration and runtime, plus diagnostics for
  rank-margin correctness AUC, direct-risk harm AUC, and the legacy Harm-head AUC.
- Checkpoint selection uses the worst scene-hash fold across Near and Contact rather than a
  single pooled development mean, reducing sensitivity to the proxy calibration split.

### New algorithmic contribution: Relative Preference and Gain Certificate (RPGC)

- **Preference-only relative context:** nominal-relative, recovery-mean, and recovery-max
  context augments only the independent rank residual. The absolute value representation is
  no longer rewritten by NASC. The new residual projection is zero-initialized, preserving
  the v48.5 checkpoint exactly at warm start.
- **Direct relative-gain distribution:** a dedicated head predicts
  `PCD(candidate)-PCD(nominal)` mean and log-variance from paired relative features. This
  replaces the independence approximation formed by subtracting two absolute predictions.
  The same output drives delta NLL training, checkpoint admission metrics, calibration, and
  closed-loop risk control.
- **Confidence-paced listwise preference and rank-gap calibration:** exact teacher-PCD
  supervises the complete recovery ordering and the best-vs-runner-up margin. Near-ties are
  downweighted; high-confidence positive groups receive stronger gradients. The learned
  margin becomes a deployable abstention certificate rather than an uncalibrated score.
- **Exact-opportunity macro-balanced sampling:** positive groups are reweighted by inverse
  teacher-best-macro frequency using only the training teacher index. This addresses the
  observed macro-5 shortcut without the high variance of minibatch GroupDRO or any use of
  validation/test distribution statistics.
- **Fold-robust policy checkpointing:** early stopping minimises the worst Near/Contact
  scene-hash-fold policy risk, combining positive-group exact regret, harmful switches, and
  false interventions.

### Required attribution protocol

1. `A_v485_ecpr_reference`: effective v48.5 ECPR path, no legacy NASC and no direct delta.
2. `B_preference_context_only`: A plus preference-only relative context.
3. `C_direct_delta_only`: A plus the direct candidate-vs-nominal gain distribution.
4. `D_full_rpgc`: preference context plus direct gain distribution (v48.6 main).

All runs use the same exact teacher-PCD, initialization, scene split, macro-balanced
positive sampler, and Natural-gate constraints. The main, multi-seed calibration, ablations,
and closed-loop probes must run sequentially and only immutable completed checkpoints may
be compared.

### Decision gates

- Near and Contact top-1 correlation should be positive on all three proxy seeds; initial
  target >0.10 and publication target >=0.20.
- Rank-margin correctness AUC should exceed 0.65 before margin-based coverage is trusted.
- At least one variant must produce non-zero held-out selections in both regimes with
  precision LCB90 >=0.40 during development, then >=0.60 for paper readiness.
- Positive recall should reach >=0.35, harmful-selection UCB90 <=0.10, and selected macro
  share <=0.85 before development closed loop.
- Safe, Near, and Contact paper targets remain closed-loop requirements; zero-action
  abstention is safe but supplies no evidence of recovery benefit.

### Non-repetition note

This iteration does not repeat shared NASC, threshold relaxation, Harm-head-driven ranking,
minibatch GroupDRO, or another generic pairwise loss. It deepens the experimentally supported
independent preference idea and makes relative gain, uncertainty, confidence, and macro
coverage separately identifiable.

## v48.5 — OC-TRAC-ECPR (2026-07-25)

### Evidence motivating this iteration

The re-uploaded v48.4 artifacts confirm a persistent candidate-to-policy gap. Across proxy calibration seeds 4801/4802/4803, candidate-positive AUC remains useful (Near roughly 0.750–0.791; Contact 0.732–0.880), but Contact group top-1 correlation is negative for every seed (-0.094 to -0.072), Near top-1 is unstable (-0.044 to 0.115), Harm AUC remains near random (about 0.512–0.561), and every verified policy selects zero actions. The completed A/SRC-reference ablation also selects zero actions. The uploaded archive contains seven complete main-training epochs and an interrupted eighth epoch; only ablation A is complete, B is partial, and C/D are absent, so v48.4 component attribution is limited to supported evidence rather than inferred from unfinished runs.

### Engineering isolation fixes

- Unified training targets, validation checkpoint metrics, and calibration on the same **exact teacher-PCD shared-option contract**. v48.4 trained/checkpointed against a differentiable soft shared-success approximation but calibrated against exact best-shared-option PCD; this objective mismatch could reverse within-group order.
- Added an independent zero-initialized preference head. Warm start exactly preserves the inherited value ranking while allowing policy ordering to specialize without changing the calibrated gain scale.
- Calibration and runtime now use preference logits for recovery-candidate top-1 and value mean/std for candidate-vs-nominal admission. Deployment therefore matches training and calibration semantics.
- Added output-root and GPU locks, per-variant and aggregate training-completion markers, immutable checkpoint SHA256 manifests, strict multi-seed source checks, and a completion auditor. Multi-seed calibration can no longer consume an actively changing checkpoint.
- Fixed the candidate-selection result key from `harmful_rate_selected` to `harmful_selected_rate`.
- Added a calibration near-miss frontier so zero-selection failures report which statistical constraint is closest to passing.

### New algorithmic contribution: Exact-Contract Preference Recovery (ECPR)

- **Exact Policy Contract:** the exact OC-MERO q table chooses one globally shared option; hard success is evaluated on teacher margins and converted to PCD identically in training, validation, calibration, and deployment diagnostics.
- **Independent Preference Ranking:** a dedicated set-aware rank residual learns which recovery candidate is best; the value distribution is reserved for whether that candidate should challenge nominal. Candidate AUC and policy top-1 are no longer forced through one scalar.
- **Confidence-paced best-vs-rest preference:** only exact-teacher ordering gaps above a minimum are supervised strongly. Near-ties are downweighted, reducing sensitivity to train/validation contract drift and ambiguous teacher order.
- **Expected preference regret:** the rank distribution is penalized by exact teacher advantage regret on positive-opportunity groups.
- **Distributional candidate-minus-nominal gain:** value mean/log-variance model the gain delta. Opportunity and downside probabilities are derived analytically from the delta distribution, replacing the non-transferable Harm head as the main deployment risk source. Harm/opportunity heads remain optional auxiliary diagnostics.
- **Risk-focused checkpoint selection:** early stopping minimizes the worse-regime sum of positive-group top-1 regret, harmful selected-candidate rate, and false-intervention rate. Always-nominal behavior cannot hide incorrect ranking.
- Pseudo-environment minibatch GroupDRO is disabled by default because v48.4 did not complete the required attribution and sparse minibatch domains can amplify noise.

### Required attribution protocol

1. `A_exact_pointwise`: exact teacher-PCD, pointwise value only.
2. `B_exact_zi_nasc`: A plus zero-initialized set context.
3. `C_exact_ecpr`: A plus independent preference head and confidence-paced preference regret.
4. `D_full_ecpr`: C plus set context and distributional delta NLL.

All four use the same scene split, initialization, exact teacher target, and distributional calibration. Run them sequentially and require `completion_audit.json` to report `comparable=true` before attribution.

### Decision gates

- Development target: positive-group top-1 correlation/accuracy and regret must improve before Natural-gate thresholds are changed.
- Screening target: Near and Contact top-1 correlation >0.10 initially; publication target >=0.20, verify precision LCB90 >=0.60, positive recall >=0.35, and harmful-selection UCB90 <=0.10.
- Only a frozen checkpoint stable across calibration seeds 4801/4802/4803 may enter closed loop.
- Safe remains strict nominal non-inferiority; Near and Contact must pass the existing regime-specific closed-loop gates before test evaluation.

### Non-repetition note

This iteration does not repeat threshold relaxation, handwritten rescue certificates, bucket-conditioned routing, ordinary candidate classification, or the unverified minibatch GroupDRO setting. It changes the supervised decision object and makes ranking, admission, and risk estimation separately identifiable.

## v48.4 — OC-TRAC-SRGR (2026-07-25)

### Evidence motivating this iteration

The uploaded v48.3 proxy-calibration run did **not** resolve the policy-level failure.
Candidate-positive AUC remained informative (Near 0.7249–0.7268; Contact 0.7634–0.7906),
but group top-1 correlation stayed negative (Near about -0.02; Contact -0.14 to -0.068),
and every fit/verify policy selected zero recovery actions. Relative to v48.1, v48.3 improved
only balanced-Near candidate AUC and slightly reduced the precision-Contact top-1 error; it
regressed precision-Near top-1 and Contact candidate AUC. Natural-gate abstention remained
correct, but no Near/Contact recovery benefit was demonstrated.

Two implementation defects were found in the executed v48.3 path:

- NASC used a non-zero random residual at warm start (`sigmoid(-1.5)≈0.18`), so loading the
  v48.1 checkpoint did not initially reproduce the inherited selector.
- `training.best_metric=loss_direct_recovery_value_worst` was never emitted by validation;
  the trainer silently fell back to total loss, so checkpoint selection did not optimize
  the intended worst-regime policy objective.

### New algorithmic contribution: Shift-Robust Groupwise Recovery (SRGR)

- **ZI-NASC:** zero-initialized nominal-anchored set context. The inherited pointwise policy
  is now an exact initialization, while the set residual learns only evidence-supported
  corrections. The gate is also made more conservative.
- **DRA-RCD:** decoupled ranking-admission regret distillation. Value-only logits learn the
  within-group teacher ordering and expected regret; opportunity/harm logits remain in a
  separate admission distribution. A weak/non-transferable harm head can therefore block
  unsafe execution without corrupting candidate ranking gradients.
- **Soft opportunity/downside supervision:** continuous teacher advantage is converted to
  soft labels around the positive/negative margins, reducing contradictory labels caused
  by small train/dev contract shifts.
- **Pseudo-environment GroupDRO:** group losses are robustly aggregated over
  `(regime, nominal-severity bin, opportunity state, teacher-best macro)` environments.
  This reduces domination by train-specific severity or macro pockets without using
  calibration/test distributions for training.
- **Policy-regret checkpointing:** validation now reports exact teacher-PCD group regret,
  top-1 accuracy, positive recall and harmful-switch rate for Near and Contact. Early
  stopping uses worst-regime mean regret and raises an error if the configured metric is
  absent; silent fallback is removed.

### Engineering and experiment protocol

- The v48.1 precision checkpoint loads into v48.4 with no shape mismatch; training from
  scratch or from v47 is not recommended for this iteration.
- Added `scripts/run_v48_4_core_ablations.sh` with four explicit runs: SRC reference,
  ZI-NASC only, DRA-RCD only, and full SRGR. Each run has its own output directory.
- Added `scripts/recalibrate_v48_4_multiseed.sh`. `CALIBRATION_SEED` values 4801/4802/4803
  produce separate proxy splits and separate output roots while reusing the same trained
  checkpoint; checkpoints are not retrained for calibration-seed robustness.
- Added aggregation tools for ablation and multi-seed summaries.
- Direct-only and full training paths now call the same direct-value loss helper.

### Required decision gates before closed loop

- Near and Contact group top-1 correlation should become positive and preferably exceed 0.10
  in screening; publication readiness remains >=0.20.
- At least one variant must produce non-zero verify selections in both regimes with finite
  precision/harm bounds.
- Candidate AUC should not fall more than 0.03 below v48.1 while group regret improves.
- The same checkpoint should be stable across calibration seeds 4801/4802/4803; output
  directories must not be shared.

### Non-repetition note

This iteration does not repeat threshold relaxation, handwritten rescue rules, ordinary
pairwise/listwise ranking, or bucket-conditioned routing. Its new contribution is the
combination of warm-start-safe set interaction, ranking/admission gradient separation,
shift-robust pseudo-environment optimization, and policy-regret checkpoint selection.

## v48.3 — OC-TRAC-NASC/RCD (2026-07-25)

### Evidence motivating this iteration

The uploaded screening run completed training and calibration diagnostics despite its
`v48_1` output-directory name. Checkpoint configs show the v48.2 SRC, encoder anchor,
exact teacher-PCD sampler and robust experts were active. Candidate AUC remained useful
(Near 0.696–0.729; Contact 0.786–0.822), but unconstrained group top-1 correlation was
near zero or negative and every calibrated policy abstained. Therefore this is not a
threshold problem: the pointwise direct head still lacks explicit candidate-set context.

### New algorithmic contribution

- **NASC (Nominal-Anchored Set Context):** a permutation-equivariant adapter compares
  every recovery candidate with the nominal embedding and exchangeable mean/max
  summaries of the recovery set. A learned conservative residual gate preserves the
  prior pointwise solution at initialization.
- **RCD (Regret-Consistent Distillation):** the composite admission policy is trained
  against the full teacher advantage distribution, not only a hard argmax, and directly
  minimizes expected teacher top-1 regret while retaining SRC harmful-mass and coverage
  constraints.
- Calibration now scores a complete scene-time candidate set in one batched call, matching
  training and closed-loop deployment semantics. Singleton APIs deliberately fall back to
  the legacy pointwise path.

### Engineering changes

- Added checkpoint/config support for the NASC adapter.
- Passed `(bucket, scene, time)` group keys and nominal masks through training.
- Updated `calibrate_policy_risk_v48.py` to batch each group via `predict_samples`.
- Enabled NASC/RCD in `train_ocrap_v48_trac_sr.sh`; existing v48.2 can be reproduced by
  setting `model.direct_recovery_set_context=false`, `POLICY_DISTILL_WEIGHT=0`, and
  `POLICY_REGRET_WEIGHT=0`.

### Required ablations

1. v48.2 SRC baseline.
2. NASC only.
3. RCD only.
4. NASC + RCD (main).
5. Remove harm/SRC constraint from NASC + RCD.

### Non-repetition note

This does not repeat prior pointwise, pairwise, listwise, top-rank, expert-routing, or
threshold-search attempts. The new element is architectural cross-candidate interaction
plus policy-level expected-regret supervision.

This root log is the canonical index for future iterations. Historical detail is retained in `ALGORITHM_CHANGELOG_V48.md` and `ALGORITHM_CHANGELOG_V48_1.md`; do not repeat an item below unless its implementation or experimental conclusion changed.

## v48.2 — OC-TRAC-SRC (2026-07-24)

### Why this iteration was necessary

An earlier static audit of the pre-fix v48.1 source found a missing `os` import and sampler-key mismatch. The uploaded screening artifacts now prove those fixes were already present in the executed job: both variants trained, the exact teacher-PCD sampler reported positive-group coverage, and v48.2 SRC settings were stored in the checkpoints. The historical engineering fixes below therefore describe prerequisites that were effective in this run, not a failure of the uploaded run itself.

### Engineering correctness fixes

- Added the missing `import os` in `src/ocrap/cli/train.py`.
- Made `group_batch_positive_advantage_{macro_ids,bucket_ids,gain_min}` the canonical sampler keys, retaining legacy aliases only for backward compatibility.
- Fixed the positive-group scanner to use the bucket stored in the exact teacher-PCD index instead of re-inferring it from the file path.
- Added `training.group_batch_require_positive_advantage_groups`; v48.2 training fails before epoch 1 when a requested positive boost resolves to zero groups.
- Added regression tests covering the canonical sampler configuration and exact-index bucket routing.
- Added atomic `calibration_build_status.json` breadcrumbs for `preflight`, `build_safe`, `build_near_contact`, `build_contact`, `merge_filter_audit`, failure, and completion.
- Added `START_STAGE=safe|near|contact|merge` to resume a failed dedicated calibration build without unnecessarily restarting completed stages; merge preflight now requires all six shard manifests.
- Replaced per-sample Safe diagnostic spam about missing targeted futures with regime-aware source requirements and one aggregate warning. Safe/nominal datasets require replay+reactive; targeted futures are required only when configured.

### New algorithm: Selective Risk-Coverage regularization (SRC)

- Added a differentiable policy distribution over the explicit nominal abstention class and all recovery candidates using the same score/opportunity/harm composition as setwise admission.
- Added a harmful-selection probability budget: probability mass assigned to teacher-harmful recovery candidates is penalized above `direct_value_selective_harm_budget`.
- Added a positive-group recovery-coverage floor so risk minimization cannot collapse to always selecting nominal.
- New controls:
  - `training.direct_value_selective_risk_weight`
  - `training.direct_value_selective_harm_budget`
  - `training.direct_value_selective_coverage_weight`
  - `training.direct_value_selective_coverage_target`
- Default v48.2 settings are risk weight 2.0, harm budget 0.05, coverage weight 1.0, and positive-group coverage target 0.65.
- The contribution is policy-level rather than another candidate classifier: it optimizes the calibrated risk/coverage trade-off under explicit abstention and complements the existing tri-state, harm-head, and robust-expert design.

### Required ablation protocol

Run both of the following from the same initialization and data split:

1. Fixed v48 baseline: `SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0`.
2. v48.2 OC-TRAC-SRC: defaults enabled.

Do not attribute gains to SRC unless both runs pass the same fit/verify Natural gate and are compared on scene-paired closed-loop evaluation.

### Local validation

- Targeted v48/v46 tests pass, including new SRC and sampler-regression tests.
- Full test-suite, compile, and shell validation status is recorded in `V48_2_VALIDATION_STATUS.txt` in the delivered package.
- Real WOMD/JAX/GPU results are not available in the local audit environment.

## v48.1 — Existing-data-first and calibration isolation

See `ALGORITHM_CHANGELOG_V48_1.md`. Key items already tried: proxy scene-disjoint calibration/dev split, dedicated validation-tail calibration construction, exact teacher-PCD coverage indexing, manifest repair, and existing-data-first screening.

## v48 — OC-TRAC-SR

See `ALGORITHM_CHANGELOG_V48.md`. Key items already tried: tri-state supervision, nominal setwise abstention, harm head, conservative two-expert aggregation, encoder fine-tuning, exact teacher-PCD alignment, joint calibration, and disabling handwritten rescue rules in the main v48 policy.
- Added `tools/inspect_calibration_build_v48.py` to classify the first incomplete stage from shard manifests and explain whether contact logs are expected.
- Added an explicit `SEED` override to the v48.2 training command so multi-seed publication experiments are reproducible rather than relying on an implicit config default.
- Added normalized L2-SP encoder anchoring during direct-only fine-tuning (`training.encoder_anchor_weight`, default 0.02). This limits drift of the shared representation away from the loaded OC-MERO/root-margin model while still allowing policy-level adaptation; without it, zero-weight root/margin losses and an unfrozen encoder could silently invalidate the pretrained core heads.
- Added an output-root `flock` guard to the dedicated calibration controller. The two commands in the supplied request are identical; launching both concurrently against the same shard/log paths can corrupt or race the build, so v48.2 rejects a second controller.

## v48.15 — OC-TRAC-PRISM-CC (2026-07-29)

### Evidence and correction of the v48.14 conclusion

The uploaded v48.14 ablation package did **not** evaluate the Natural gate.  Every
certificate worker terminated at shell line 23 with `variant: unbound variable`:
`local variant="$1" gpu="$2" run="$OUTPUTDIR/candidates/$variant"` expanded
`$variant` before the local assignment under `set -u`.  The controller then
misclassified missing risk JSONs as `GATE_FAILED.json`.  Consequently, absence of
`NEXT_COMMANDS.txt` in that run means *calibration artifact failure*, not a valid
algorithmic gate rejection.  v48.15 separates exit code 30 / `CALIBRATION_FAILED.json`
from exit code 20 / `GATE_FAILED.json` and provides a no-retraining recovery script.

The Safe paired package also had zero matched calibration targets because the runner
forced `closed_loop.bucket_split=test` on `calibration_safe`; it silently fell back to
eight arbitrary WOMD scenes.  v48.15 removes the forced split, requires non-empty
bucket targets, disables stale resume by default in the Safe wrapper, and emits
scene-level jerk/yaw-rate p95.  The uploaded 8-scene Safe result is therefore a
nominal-lock smoke test, not a calibration-safe non-inferiority result.

### v48.14 algorithm evidence that remains valid

- The dedicated scene-disjoint adaptation/dev/certificate protocol is retained.
- Target-domain adaptation reduced harmful-switch/false-intervention diagnostics on
  adaptation dev, but the full `direct_delta_adapters` update trained roughly 0.39M
  parameters from only 16 deployable-positive Near groups (10 scenes) and 44 Contact
  groups (17 scenes).  Positive admission recall collapsed to 0–0.33 Near and
  0–0.036 Contact, indicating overfit/over-conservative forgetting rather than a
  calibrated deployable certificate.
- Dynamic hard-harm mining reduced some false-safe diagnostics but further suppressed
  positive recall.  It remains a moderate auxiliary weight, not the main adaptation
  mechanism.
- The same-group counterfactual term produced no consistent gain over hard-harm-only
  adaptation and is disabled in the v48.15 main experiment.

### New algorithm: PRISM-CC

**PRISM-CC = Proposal-aligned Risk adaptation with Independent Scene-disjoint
certification and low-Capacity Correction.**

1. **Frozen proposal and frozen source evidence.**  The high-recall v48.13 top-k
   recovery proposal and the source ordinal-evidence experts are both frozen.
2. **Tiny regime-specific residual evidence calibrator.**  A zero-initialized MLP
   consumes the frozen source evidence center/width and frozen policy score/gap, then
   produces a bounded residual correction.  The two regime calibrators contain 132 state parameters in total, versus approximately 392k trainable parameters
   in v48.14.  Initial predictions exactly reproduce the source checkpoint.
3. **Balanced three-state correction.**  Ordered harmful/dead-zone/beneficial NLL is
   retained, but hard-harm amplification is reduced and missed-benefit importance in
   checkpoint selection is increased to avoid an always-abstain optimum.
4. **Independent certificate pool unchanged.**  Natural-gate thresholds, scene
   disjointness, Wilson bounds, harmful-selection bounds, support requirements, and
   opportunity-normalized macro checks are not relaxed.

### Engineering and attribution changes

- Fixed the certificate worker local-variable expansion bug.
- Added `VARIANTS` filtering so a single-variant ablation task does not launch or report
  a nonexistent sibling variant.
- Distinguish calibration/controller failure from a genuine Natural-gate rejection.
- Added `scripts/recover_v48_14_certificate_pool.sh` to evaluate already-trained v48.14
  checkpoints without retraining.
- Added strict Safe target matching and removed arbitrary-scene fallback.
- Added scene-level jerk and yaw-rate p95 to Safe paired output.
- Added `scripts/run_v48_15_parallel_ablations.sh`; four ablations run concurrently per
  variant wave, two processes per A30 as supported by the measured memory footprint.
- Added layered `tools/check_v48_15_learning_gates.py` diagnostics.

### Required v48.15 ablations

1. `A_source_dedicated`: fixed source checkpoint, dedicated recalibration only.
2. `B_full_adapter_prism`: v48.14 high-capacity target adaptation.
3. `C_tiny_calibrator`: low-capacity residual correction without hard mining.
4. `D_full_prism_cc`: low-capacity correction with balanced hard-harm/missed-benefit
   supervision.

### Non-repetition and stopping rule

Do not repeat all-pairs recovery ranking, shared NASC, minibatch GroupDRO, continuous
relative-gain regression, broad conformal radii, strong hard-harm weighting, or
full-adapter target adaptation unless new evidence invalidates the prior conclusions.
First recover and evaluate the already-trained v48.14 certificates.  Run stress
closed-loop only when the controller creates `NEXT_COMMANDS.txt`; no gate threshold is
lowered to force that file to appear.

## v48.16 — OC-TRAC-ANCHOR (2026-07-29)

### Correction of the v48.15 experimental conclusion

The uploaded v48.15 certificate result with `rc=20` was not a valid Natural-gate
rejection.  The dedicated partition deliberately labels samples as
`evidence_adapt_train`, `evidence_adapt_dev`, and `certificate_pool`, while both
standard calibration and policy-risk calibration accepted only literal
`calibration`/`val`.  Every certificate NPZ was therefore discarded:
`num_groups=0`, `num_scenes=0`.  The controller installed the empty JSON files and
misclassified the risk tool's failure as a gate rejection.  v48.16 introduces
protocol-aware split roles, requires non-empty scene-disjoint certificate data, and
uses exit code 30 for artifact/protocol failure.  A Natural gate is considered
evaluated only when both Near and Contact contain non-zero groups, scenes, fit folds,
and verify folds.

The uploaded Safe paired run was also invalid: 120 offline targets were loaded but
zero were matched after scanning only 2,000 raw validation scenarios.  The correct
WOMD validation shard specification is `validation_tfexample.tfrecord@150`, and
sparse dedicated target IDs require scanning the complete validation set.  v48.16
validates all 150 shard files, defaults `SAFE_RAW_MAX_SCENARIOS=0`, and hard-fails
instead of writing an empty apparently valid result when no target is matched.

### Evidence retained from the adaptation-dev ablation

Final certificate metrics are unavailable because of the split-role bug, but the
adaptation-dev results still reveal the optimization failure mode:

- the high-capacity v48.14 adapter substantially reduces admissions and often
  destroys Contact positive recall;
- the 132-parameter v48.15 calibrator preserves the source model structurally and
  lowers harmful/false interventions, but collapses positive admission recall to
  0--0.11 Near and approximately 0.036 Contact;
- the v48.15 hard-harm/hard-benefit configuration selected exactly the same best
  epoch metrics as the plain tiny calibrator, so the same weighting is not repeated;
- the frozen high-recall top-k proposal, source ordinal evidence, scene-disjoint
  adaptation/certificate protocol, and zero-initialized bounded residual correction
  remain the useful foundation.

### New algorithm: ANCHOR

**ANCHOR = Adaptation with Nominal-preserving Class-balanced Held-out Ordinal Risk.**

1. **Class-balanced ordered evidence.**  Proposal evidence loss is averaged within
   harmful, dead-zone, and beneficial classes before averaging present classes.
   Dead-zone prevalence can no longer make all-abstain the lowest-loss solution.
2. **Bipolar probability margins.**  Beneficial proposals are explicitly pushed to a
   minimum benefit probability and harmful proposals to a minimum harm probability.
   This trains both tails required by the selective certificate.
3. **Source-residual anchoring.**  The target-domain calibrator residual receives an
   L2 anchor, retaining the source evidence unless dedicated data supports a bounded
   correction.
4. **Lower-capacity correction.**  Hidden width is reduced from 8 to 4 and residual
   scale from 0.30 to 0.20.  Strong hard-harm mining is replaced by moderate
   harm/benefit weights; the missed-opportunity checkpoint penalty is increased.
5. **No proposal retraining in this round.**  The top-k recovery proposal is frozen so
   any change in Natural-gate performance is attributable to target-domain evidence
   correction rather than another ranking modification.

### Engineering and attribution changes

- Added semantic split-role aliases in `ocrap.models.data`.
- Dedicated standard calibration explicitly accepts `certificate_pool` and disables
  validation fallback.
- Policy-risk calibration accepts an explicit `--allowed-splits` contract and returns
  an artifact-failure code for empty data.
- Certificate completion now validates non-zero samples/groups/scenes and non-empty
  fit/verify folds before installing results.
- Added dedicated protocol role/scene-leakage audit.
- Main and ablation controllers distinguish exit 0 (gate pass), 20 (valid gate
  rejection), and 30 (pipeline/artifact failure), and capture adaptation log tails.
- Safe WOMD shard preflight requires 150 validation shards; complete-set scanning is
  the default for sparse target matching; zero matched targets now hard-fail.
- Generated `NEXT_COMMANDS.txt` invokes an authorization-checking stress wrapper.
- Four v48.16 ablations run concurrently per variant wave, two light jobs per A30.

### Required v48.16 ablations

1. `A_source`: fixed v48.13 source evidence plus valid dedicated certificate.
2. `B_old_tiny`: the v48.15 tiny-calibrator objective under the repaired pipeline.
3. `C_balanced_margin`: class-balanced ordinal evidence and bipolar margins.
4. `D_full_anchor`: class-balanced margins plus source-residual anchoring.

Do not claim a v48.15/v48.16 gate result unless `certificate_data_valid=true` and
both risk JSON files contain non-zero independent scenes.  Do not run test/stress
closed loop after exit 20; development-only qualitative diagnostics may be used, but
must be isolated from paper metrics and threshold selection.

## v48.17 BRIDGE — 2026-07-30

**BRIDGE: Batch-balanced Regime-conditioned Identity-preserving Discriminative Group Evidence**

### Why this version was necessary

The completed v48.16 ablation bundle contains eight valid, non-empty dedicated
certificates (four components times balanced/precision).  Every run returned a real
Natural-gate rejection (exit 20), not an artifact failure: the held-out verify folds
contained 163 Near groups with 6 positive opportunities and 380 Contact groups with
14 positive opportunities, but every accepted rule selected zero groups.  The
uploaded Safe paired run contains 120 matched scenes and is identical on its available
metrics, while route progression was not emitted and jerk/yaw-rate did not yet carry
non-inferiority margins.

The source proposal is not the dominant bottleneck.  On the balanced source
certificate, top-3 contains an oracle-best or another positive candidate for all
positive Near groups and all positive Contact groups.  Positive-group top-1 accuracy
is 0.643 for Near and 0.594 for Contact.  In contrast, Evidence has weak harmful
ranking and severe false-switch exposure: proposal-Evidence harm AUC is below 0.5 in
both regimes, and the unconstrained non-positive false-switch rate exceeds 0.90.
Contact additionally exhibits a strong fit-to-verify reversal: the closest fit rule
selected 1/20 positive and 2/20 harmful candidates, while its verify counterpart
selected 0/24 positive and 14/24 harmful candidates.

v48.16 B/C/D changed the dedicated certificate metrics only at approximately floating
point noise.  Code audit identified three reasons:

1. The target calibrator observed only four summary scalars, so candidates with
   similar source center/width and rank margins but opposite target-domain outcomes
   were conditionally indistinguishable.
2. The advertised class-balanced Evidence loss was balanced inside each scene-time
   group.  Because most groups contain a single teacher class, it often collapsed to
   ordinary NLL; dead-zone groups still dominated across the minibatch.
3. Weighted replacement increased the probability of rare groups but did not ensure
   beneficial, harmful and dead-zone evidence was simultaneously present in a batch.
   Bipolar margins and class balance were therefore frequently inactive.  Checkpoint
   selection could still prefer the early always-abstain solution.

### Algorithm changes

1. **Identity-preserving tri-simplex residual.**  Added
   `direct_recovery_evidence_calibrator_mode=simplex_context`.  A zero-initialized,
   bounded residual is added to the frozen source log-probabilities of the harmful,
   dead-zone and beneficial classes, followed by a softmax.  At initialization the
   model is exactly the source Evidence model; unlike the old center/width correction,
   the beneficial and harmful tails may be corrected independently while retaining a
   valid probability simplex.
2. **Frozen candidate-vs-nominal context.**  The small calibrator can consume the
   source relative feature vector in addition to source class summaries and proposal
   rank margins.  Context is detached by default, preserving proposal/source Evidence
   attribution and keeping target adaptation low capacity.
3. **Batch- and regime-balanced ordinal Evidence.**  Beneficial, harmful and dead-zone
   candidate losses are accumulated over the whole minibatch and separately by
   regime, then averaged over the classes/regimes that are present.  Bipolar benefit
   and harm probability margins are applied at the same batch scope.
4. **Evidence-stratified scene-time batches.**  The group sampler builds exact teacher
   strata from best candidate-vs-nominal PCD: beneficial, harmful-only and dead/mixed.
   Replacement sampling is performed within each stratum and batches are interleaved,
   with default group fractions 0.35/0.35/0.30.  Scene-time grouping remains intact.
5. **Recall-constrained checkpoint selection.**  Added a configurable minimum positive
   recall and a shortfall penalty in the direct-policy metric.  The default v48.17
   target is recall >= 0.25 on adaptation dev; this prevents an always-abstain epoch
   from winning only by avoiding harm.
6. **Conservative bounded adaptation.**  BRIDGE freezes the source model and proposal,
   uses an 8-wide calibrator, a bounded residual scale of 0.75, an L2 source anchor of
   0.02, and no selective-risk, hard-mining or pairwise objectives.  Those objectives
   were intentionally disabled because previous versions did not provide stable
   incremental evidence.

### Engineering changes

- Added full checkpoint/config compatibility for calibrator mode, context input and
  context detachment; legacy `center_width` checkpoints remain loadable.
- Added exact evidence-stratum accounting to training summaries and hard failure when
  stratification is requested without an exact scene-time group index.
- Fixed Natural-gate checker field names (`precision_wilson_lcb90` and
  `teacher_advantage_mean`) so reports no longer silently read missing metrics.
- Fixed final candidate selection to read `teacher_advantage_mean` (with legacy
  fallback), so a dual-pass run is not ranked with a silently zeroed advantage.
- Rewrote the ablation summarizer, corrected dedicated-certificate paths and proposal
  metric names, and made the reported version explicit.
- Added signed fixed-route progression at scene level.  Waymax SDC routes are used
  when available; otherwise the already constructed logged-future route proxy is
  transformed once to global coordinates and its source is reported explicitly.
- Added 5% paired non-inferiority margins for jerk and yaw-rate; the Safe paper-ready
  flag now requires route progression, jerk and yaw-rate to be available and pass.
- Added authorization-checked v48.17 stress execution and exit-code separation:
  0 = valid Natural-gate pass, 20 = valid algorithmic rejection, 30 = engineering or
  artifact failure.
- Added four focused unit tests for simplex identity/bounds, calibrator capacity,
  evidence-stratified batching and signed route progression.

### Required v48.17 experiment and ablations

Main experiment: `run_v48_17_bridge_dedicated.sh`, with balanced on GPU0 and precision
on GPU1.

Component ablations compare against the already completed v48.16 `D_full_anchor`
baseline and therefore do not rerun old failed designs:

1. `A_simplex_scalar`: tri-simplex residual with the legacy four scalar inputs.
2. `B_context_simplex`: add frozen relative context, keep the old sampler/loss scope.
3. `C_full_bridge`: add evidence-stratified batches, batch/regime balance and the
   recall-constrained checkpoint metric.

### Decision and stopping rules

- Exit 0 and `NEXT_COMMANDS.txt` present: run authorized stress closed loop, rerun Safe
  paired evaluation with route progression, then perform multi-seed confirmation.
- Exit 20: do not inspect test/stress.  Use the three component ablations and explicitly
  labelled validation-only trajectory diagnostics to determine whether the remaining
  limitation is conditional Evidence capacity or irreducible positive support.
- Exit 30: no algorithm conclusion is allowed; repair the pipeline first.
- Do not relax the Natural-gate statistical constraints merely to create coverage.
- Do not retrain the proposal unless v48.17 shows that top-3 positive-hit rate itself
  degrades under the corrected protocol; current uploaded evidence supports freezing it.
- Do not rebuild the three regime datasets in this round.  Sparse positive support is
  addressed through sampler/loss/checkpoint logic so the next result remains
  attributable to the algorithm rather than a changed dataset.

## v48.28 — PROVENANCE-MARGIN-BRIDGE

### Motivation

v48.27 returned a valid `RC=20`, but two independent defects prevented a clean interpretation:

1. adaptation-dev shadow targets were built from standard WOMD `validation`, while the closed-loop audit defaulted to `validation_interactive`; all 16 targets missed after scanning 43,479 raw scenarios;
2. the stage-1 factor checkpoint metric was constant across epochs and selected epoch 0 for both Balanced and Precision, so the five harm-factor heads were frozen at their semantic prior.

The component-harm parameterization also used `prior=-2, scale=2`, which bounded each candidate component logit to at most zero and therefore could not represent `p(harm)>0.5` for strong veto violations.

### Engineering changes

- Added an official WOMD `scenario/id` preserving Waymax loader, following the custom-loader contract used by Waymax when string IDs are required.
- Persisted `official_scenario_id`, `legacy_scenario_id`, source scenario index, source role, source pattern, and `max_num_objects` into RawScenario metadata, sample NPZ files, and manifests.
- Changed adaptation-dev shadow default from `validation_interactive` to the same standard `validation` TFRecord family used by the calibration-regime builder.
- Added fail-closed target/source-role provenance audit before shadow execution.
- Restricted legacy source-order matching to the same declared source role. It is migration-only; official `scenario/id` is the primary identity.
- Added `repair_v48_27_dev_shadow_with_v48_28.sh` so existing v48.27 checkpoints can be re-evaluated without retraining.
- Added model-contract validation for component count, prior, frontier mode, bounded admission, and component scale.
- Added factor-transfer integrity validation: stage-1 and stage-2 checkpoints must be post-epoch-0, factor heads must be nonzero and frozen during admission training, and the admission head must be trained.
- Added a structured `GATE_FAILURE_DECOMPOSITION.json` separating proposal infeasibility, development-rule fitting failure, certificate generalization failure, and pass.

### Algorithm changes

- Replaced the stage-1 checkpoint metric `direct_factor_selection_risk` with `direct_factor_supervised_risk`, which includes the actual supervised factor loss and therefore changes when the benefit and component-risk heads learn.
- Disabled initial-checkpoint eligibility in both factor and admission stages.
- Increased the default component-harm residual scale from 2.0 to 6.0 while retaining the semantic prior of -2.0. The representable logit range changes from approximately `[-4, 0]` to `[-8, 4]`.
- Kept five non-compensatory factors: DRS, deployability, oracle-to-deployable gap, hard rule, and harm proxy.
- Kept top-3 frozen proposals, categorical one-action admission, bounded identity-preserving admission, and legacy Noisy-OR disabled.
- The v48.28 main model uses two-stage factor→admission training with deployment-exact safe-utility regression only. Listwise/frontier terms are no longer in the main objective and remain an ablation because v48.27 showed no consistent benefit.

### Ablations

The four groups are:

1. `A_three_factor_wide_range` — three factors, scale 6;
2. `B_five_factor_old_range` — five factors, scale 2;
3. `C_five_factor_wide_range_regression` — five factors, scale 6, regression-only main design;
4. `D_add_listwise_frontier` — C plus listwise/frontier terms.

All eight Balanced/Precision jobs are launched concurrently: four jobs per 24 GB A30. Per-task data workers and host threads are limited to one to reduce TFRecord and CPU contention.

### Protocol decisions

- The certificate concept is retained. Complete certificate oracle support is feasible for Near and Contact in v48.27, so the dominant failure is learned development-rule fitting, not mathematical gate impossibility.
- Gate thresholds are not reduced post hoc.
- Safe remains nominal-first with held-out non-inferiority checks; Near and Contact use the registered Natural gate.
- Existing v48.27 shadow outputs contain no valid physical rollouts and must not be interpreted as zero collision/exposure/intervention.

### Non-claims

The local environment does not contain the user's WOMD/Waymax runtime or two A30 GPUs. v48.28 has passed static and unit tests, but no claim is made that it already obtains `RC=0`, passes the Natural gate, or reaches the Near/Contact closed-loop publication targets.

## v48.29 — VETO-RANK-PHYSICS-BRIDGE

### Motivation

v48.28 returned a valid `RC=20`. The proposal-constrained oracle remained feasible on the complete Near and Contact certificates, while all Balanced/Precision branches failed during adaptation-dev rule fitting. The nearest rules had low safe-positive precision/recall and excessive harmful selection, so the certificate was correctly rejecting an unsafe learned selector.

The v48.28 shadow matched eight paired scenes per branch, but audit found a runtime alias defect: dataset buckets were named `evidence_adapt_dev_near_contact` and `evidence_adapt_dev_contact`, whereas selector overrides, calibrated `gamma_rec`, and Contact physics recognized only bare `near_contact`/`contact`. Consequently every scene ran with `gamma_rec=0`; Contact was not marked as a post-contact target and its anchor/free-space/escape/re-contact metrics were missing or misleading. The matched shadow therefore established provenance only, not physical efficacy.

Runtime timing also showed that online `selected_topk` OC-MERO audit labels consumed 98.48%–98.57% of scene wall time. Model selection and Waymax step metrics were a small fraction of total cost.

### Engineering changes

- Added a shared canonical regime parser for dataset provenance prefixes. `evidence_adapt_dev_*`, `certificate_pool_*`, calibration, validation and test names now resolve to `safe`, `near_contact` or `contact` without misclassifying Near as post-contact.
- Applied the same alias contract to all selection `*_by_bucket`/`*_by_regime` overrides and to `gamma_rec_by_bucket`.
- Added explicit `canonical_regime`, `bucket_aliases`, `post_contact_target` and runtime-contract metadata to closed-loop results.
- Contact physics now recognizes provenance-prefixed Contact buckets, creates a finite causal contact anchor, and enables re-contact, overlap, post-contact free-space, escape and stable-stop metrics.
- Added `check_v48_29_shadow_runtime_contract.py`. Shadow execution fails closed unless Near/Contact regimes are correct, every scene has a finite positive calibrated gamma, Contact anchors are finite, post-contact semantics are active, and metrics are valid.
- Fixed the runtime-contract auditor itself to serialize invalid/non-finite values as JSON `null` rather than crashing while reporting an invalid legacy result.
- Added `repair_v48_28_dev_shadow_with_v48_29.sh` so v48.28 checkpoints can be re-evaluated without retraining.
- Changed physical dev-shadow default to `label_mode=fast`, zero online audit labels. Policy execution and Waymax physical metrics are unchanged. A separate suffix directory can run a sparse `selected_topk` teacher audit when needed.
- Kept official WOMD `scenario/id`, source-role provenance and legacy source-index migration checks from v48.28.
- Added fail-closed checkpoint/inference validation for admission prior mode, bounded admission, frontier, component count, component scale and semantic risk prior.
- Added eight v48.29-specific tests. Full regression result: 259 passed, 5 warnings.

### Algorithm changes

1. **Independent five-factor veto.** DRS, deployability, oracle-to-deployable gap, hard rule and harm proxy remain separately supervised non-compensatory risk factors with semantic prior -2 and scale 6.
2. **Benefit-only admission prior.** Added `direct_recovery_evidence_admission_prior_mode=benefit_only`. Admission inherits detached raw-benefit evidence but no longer subtracts the same maximum risk a second time. The five factors remain an independent calibrated hard veto, and harmful actions still receive negative safe-utility targets.
3. **Hardest-negative safe ranking.** For every proposal group with a safe-positive opportunity, the teacher-best safe action must outrank nominal and the hardest non-safe proposal by a registered margin. Groups with no safe opportunity push every recovery score below nominal.
4. **Two-stage training retained.** Stage 1 learns dense raw-benefit ordering and five harm factors only. Stage 2 freezes them and learns bounded admission with deployment-exact safe-utility regression, categorical one-action supervision and hardest-negative ranking.
5. **Listwise/frontier removed from the default main model.** v48.28 did not show stable incremental benefit. A small frontier term remains only in the D ablation.
6. **Top-3 frozen proposal retained.** Complete certificate oracle support remains feasible, so proposal expansion is not justified.
7. **Legacy Noisy-OR remains disabled.** Deployment selects exactly one recovery action or nominal.

### v48.29 ablations

1. `A_risk_centered_reference` — old risk-centered admission prior;
2. `B_veto_decoupled` — independent veto plus benefit-only admission;
3. `C_add_safe_hard_negative` — B plus hardest-negative safe ranking; v48.29 main design;
4. `D_add_frontier_to_hard_negative` — C plus a light frontier term.

All eight Balanced/Precision jobs launch concurrently: four tasks per 24 GB A30. Each task is limited to one DataLoader worker and one OMP/MKL/OpenBLAS thread to control CPU and filesystem contention.

### Protocol and decision rules

- The Natural certificate is retained. Oracle feasibility means the current gate is not mathematically impossible; thresholds are not reduced post hoc.
- `RC=20` means a valid algorithmic rejection. `RC=30` is reserved for engineering, provenance, checkpoint, index, artifact or runtime-contract failure.
- A physical shadow result is interpretable only when `SHADOW_RUNTIME_CONTRACT.json` is valid.
- If v48.29 improves offline precision/risk but valid physical shadow does not improve, the next change must be a preregistered candidate-level temporal physical teacher, not threshold tuning on certificate or held-out stress.
- No claim is made locally that v48.29 already passes the gate or meets CCF-A Near/Contact targets; WOMD/Waymax execution on the user's two A30s is required.

## v48.30 — SLACK-RANK-BRIDGE

### Motivation

v48.29 returned a valid `RC=20`. All four Balanced/Precision Near/Contact branches still failed at `development_rule_fit`, while the complete top-3 proposal-constrained certificate oracle remained feasible. The failure was therefore not a mathematically impossible certificate or missing proposal support.

The hardest-negative objective produced a real local gain on adaptation-dev—Near safe-opportunity recall increased—but certificate selection became substantially more aggressive and harmful. Joint audit found a population-prior contract error in the admission stage: only 52 of 1,167 training groups (4.46%) were safe-beneficial, while stage 2 forced 50% safe-positive groups with replacement and applied no importance correction. The model learned a recovery-heavy resampled prior that was incompatible with the natural development/certificate population and the low-intervention Natural gate.

v48.29 paired shadow execution was technically valid, but it did not establish publication-level physical gains. Near produced only small TTC changes with lower NUP and nontrivial intervention. The eight Contact targets were floor/ceiling saturated for overlap, re-contact, escape and stable-stop events, so those event metrics were not informative; continuous clearance/free-space changes were negligible or adverse.

### Unified algorithm change

SLACK-RANK-BRIDGE uses one regime-agnostic physical semantic for Safe, Near and Contact. It does not expose a regime identifier to the Evidence model and does not dispatch to separate policies.

For each recovery candidate relative to nominal, the model predicts five signed non-degradation margins:

1. DRS margin;
2. deployability margin;
3. oracle-to-deployable-gap margin;
4. hard-rule margin;
5. harm-proxy margin.

Each target margin already includes its preregistered tolerance. A value at or below zero is inside the allowed envelope; a positive value crosses an uncompensated safety boundary. The unified safety slack is

```text
s(a) = max_k m_k(a)
```

and the admission prior is

```text
U_safe(a) = B(a) - lambda * relu(s(a))
```

where `B(a)` is detached raw-benefit evidence. The independent component veto remains fail-closed. The continuous hinge supplies stable ordering close to the boundary, while the hard veto prevents benefit from compensating for a true violation.

This semantic protects Safe because unnecessary non-nominal actions lack positive benefit or violate at least one non-degradation margin; it permits Near recovery only when benefit is obtained inside the common physical envelope; and it permits Contact escape/stabilization only when deployability, recovery gap, hard-rule and harm-proxy coordinates do not deteriorate beyond the same registered tolerances.

### Training changes

1. **Natural-population stage 2.** Admission training now uses every scene-time group at most once per epoch:

   ```text
   GROUP_BATCH_STRATIFIED=false
   GROUP_BATCHING_REPLACEMENT=false
   ```

   Positive weighting remains inside the loss; it no longer alters deployment prevalence through replacement sampling.

2. **Signed component-margin regression.** Binary component targets are retained, and stage 1 additionally regresses the continuous distance to each veto boundary:

   ```text
   predicted_margin_k = factor_temperature * component_logit_k
   L_margin = SmoothL1(predicted_margin_k, teacher_margin_k)
   ```

3. **Population-aware checkpoint metric.** Added `direct_population_safe_rank_risk`, evaluated on the natural adaptation-dev population. It combines safe top-1 regret, harmful recovery mass, false-admission mass, safe-recall shortfall and safe-mass shortfall. Near and Contact are used as worst-stratum reports only; no regime ID enters the model.

4. **Hardest-negative retained under the corrected prior.** Best-safe-vs-nominal-and-hardest-non-safe supervision remains in the full design, but it is now trained on natural groups rather than an 11x positive-oversampled population.

5. **Default objective simplification.** Safe-utility listwise and frontier contrast remain disabled in the main model because v48.29 C/D ablations showed no stable incremental benefit. Legacy Noisy-OR, unbounded admission and top-8 proposal remain disabled.

### Engineering and attribution safeguards

- Added checkpoint/inference persistence for slack temperature and slack penalty.
- Added `TRAINING_CONTRACT.json`, which fails closed unless stage 2 is natural and without replacement, the population checkpoint metric is finite and varies across epochs, signed margin regression is enabled, factor transfer is valid, five factors are present, no regime routing is used and legacy Noisy-OR is disabled.
- Main runner explicitly pins five factors, natural stage-2 defaults, zero listwise/frontier weights, separate factor/admission epoch budgets and the safety-slack model contract. This prevents ambient environment variables from silently changing the registered main algorithm.
- Added `PHYSICAL_TARGET_SUPPORT.json`. It warns when Contact event targets are floor/ceiling saturated; continuous physical deltas remain reportable, but event non-improvement cannot be interpreted as success.
- Added structured `GATE_FAILURE_DECOMPOSITION.json` for proposal infeasibility, development fitting, certificate generalization and engineering failures.
- Retained exact train/dev index separation, official WOMD provenance, canonical regime aliases, positive calibrated gamma checks and Contact anchor checks from v48.28/v48.29.
- Corrected the v48.30 controller event name and pinned factor/admission epoch variables in the main runner.

### Ablations

Eight jobs are launched concurrently, four per 24 GB A30:

1. `A_natural_population_reference` — natural population, benefit-only admission;
2. `B_add_signed_component_margin` — A plus continuous five-factor margin regression;
3. `C_add_safety_slack_projection` — B plus unified safety-slack prior;
4. `D_full_slack_rank` — C plus hardest-negative, the v48.30 main design.

Per-task workers and host threads remain limited to one during eight-way execution. If filesystem/CPU contention dominates, reduce `TASKS_PER_GPU` to two rather than increasing DataLoader workers.

### Protocol decisions

- The Natural certificate and registered gate thresholds are unchanged. Complete oracle feasibility means post-hoc gate relaxation is not justified.
- `RC=0` alone authorizes held-out stress. `RC=20` remains a valid algorithmic rejection; `RC=30` remains engineering/protocol failure.
- If natural-population training improves precision but reduces recall, future changes must use loss weighting or better representations without changing the sampling prior.
- If development passes but certificate fails, focus on scene-level generalization and slack calibration.
- If offline safe ranking improves but valid physical shadow does not, the next step is a preregistered candidate-level temporal physical teacher, not certificate threshold tuning.

### Validation and non-claims

Local validation:

```text
265 passed, 5 warnings
compileall PASS
all shell bash -n PASS
```

The local environment does not contain the user's WOMD/Waymax runtime or two A30 GPUs. No claim is made that v48.30 already obtains `RC=0`, passes the Natural gate or reaches the Near/Contact CCF-A closed-loop targets.

## v48.35 — CONTINUOUS-FRONTIER

### Motivation and audited failure

v48.34 produced a valid algorithmic rejection (`pipeline_valid=true`, `certificate_executed=true`, `gate_evaluated=true`, `certificate_exit_code=20`, `test_roots_read=false`). The top-5 proposal still contained safe recovery opportunities, but the learned selector did not transfer them into a stable scene-disjoint certificate policy. Near retained useful candidate discrimination but had only one clean certificate hit; Contact selected no safe-positive action and many harmful actions. The v48.34 barrier/boundary ablations did not solve this failure.

The audit found four attribution errors that had to be removed before another algorithm claim:

1. Near and Contact were fitted with separate frozen threshold rules, creating a deployment policy fork despite the network not receiving a regime ID.
2. `proposal_exact_eligible_*` used fixed diagnostic thresholds rather than the frozen deployed rule, so “exact” diagnostics could disagree with actual selection.
3. the hard-boundary training continuation used semantic thresholds that were not constrained to the final fitted rule domain, making the corresponding ablation nearly inert;
4. post-gate commands referenced stale/missing scripts and could turn an algorithmic RC=3/20 into an engineering RC=30.

### Unified algorithm

v48.35 keeps one continuous mechanism for Safe, Near and Contact. Regime labels are not model inputs and are not used to choose a rule. Near and Contact names appear only as certificate audit strata for worst-stratum constraints.

For each candidate, the compact trainable evidence bridge now receives executable prefix physics relative to the nominal candidate:

- prefix parameters;
- macro identity;
- prefix state trajectory;
- control sequence.

It excludes absolute ego state, utility/hard/harm/feasibility audit scalars, nominal/time flags, agents, map and BEV suffixes. This restores action identity without exposing scene or regime shortcuts.

The five signed component logits define a continuous worst-component safety frontier. The free admission logit is capped by the safety frontier using a differentiable smooth minimum:

```text
free(a) = benefit(a) + residual(a)
cap(a)  = - max_k component_logit_k(a)
admission(a) = smooth_min(free(a), cap(a))
```

Therefore benefit or a large learned residual cannot compensate for a predicted component violation. The shared rule fitter is additionally restricted to the semantic domain:

```text
opportunity_threshold >= 0.5
harm_threshold        <= 0.5
score_threshold       >= 0.0
```

This is required for the cap to remain non-compensatory at deployment, not only during training.

### One shared deployment rule

`calibrate_shared_continuous_rule_v48_35.py` pools adaptation-dev proposal rows and fits exactly one four-threshold rule. Audit strata are used only to require that the same rule satisfies every stratum's minimum support, precision LCB, harmful-exposure UCB and macro-concentration constraints. Both certificate workers consume the byte-identical frozen JSON and record its SHA256.

A failed shared development fit exits with RC=3 and is preserved as an algorithmic rejection. The controller maps only missing/corrupt/protocol-inconsistent artifacts to RC=30. Held-out test commands are generated only after certificate RC=0.

### Engineering and protocol fixes

- real deployed-rule diagnostics are emitted as `proposal_deployed_rule_*`;
- legacy `proposal_exact_eligible_*` fields remain only as explicitly deprecated aliases;
- duplicate argparse registration in the mixed source snapshot is removed;
- model, training, metric/calibration and continuous-frontier fail-closed contract checks are added;
- factor-cache identity includes the context source and verifies source/copy checkpoint hashes;
- training metadata no longer claims that the train-time semantic boundary equals the final fitted threshold;
- Safe and stress wrappers verify `V48_35_COMPLETE.json`, gate authorization and candidate identity;
- stress execution verifies that Near and Contact certificates reference the same shared frozen-rule SHA and expose identical selector overrides;
- generated command dependencies are regression-tested;
- repository-local pytest import configuration is added.

### Preregistered ablation

The v48.35 ablation is a 2x2 design with one shared rule in every task:

1. legacy relative context + compensatory safety slack;
2. executable physical-relative context + compensatory safety slack;
3. legacy relative context + non-compensatory frontier cap;
4. executable physical-relative context + non-compensatory frontier cap (main).

The design isolates representation and admission geometry without creating Near/Contact-specific policies. Compatible Stage-1 factor caches are reused only after exact semantic-contract validation.

### Decision rules and non-claims

- `RC=0`: shared development rule and independent certificate pass; only then Safe paired non-inferiority and held-out stress are authorized.
- `RC=20`: valid algorithmic rejection; inspect shared-rule deficits and certificate rows, then improve representation/losses without reading test.
- `RC=30`: engineering, provenance, cache, checkpoint, script, artifact or protocol failure; no algorithm comparison is valid.

Local CPU validation checks code and contracts only. WOMD/Waymax and the user's A30 environment are unavailable locally, so v48.35 is not claimed to obtain `RC=0` or publication-ready closed-loop gains before the registered experiments are run.

## v48.35.1 — RC30-TRAINING-CONTRACT-HOTFIX

### Scope and attribution

This release is an engineering-only hotfix for the uploaded v48.35 run. It does **not** change the OC-RAP model, candidate set, training objective, checkpoint-selection metric, shared-rule fitter, certificate thresholds, gate, datasets, or Safe/Near/Contact semantics. The algorithm remains one network, one continuous physical representation, one non-compensatory frontier, and one shared deployment rule; Near and Contact remain audit strata only.

The uploaded run stopped with:

```text
failure_stage=training_contract
raw_exit_code=4
normalized_exit_code=30
balanced_adaptation_rc=0
precision_adaptation_rc=0
certificate_executed=false
gate_evaluated=false
```

The sole failed check was `exact_eligibility_all_stages`. The trainer did enable
`POLICY_METRIC_EXACT_ELIGIBILITY=true`, which was persisted as
`cfg.training.direct_policy_metric_exact_eligibility=true` in every checkpoint, but
`adapt_ocrap_v48_35_continuous_frontier_single_stage.sh` wrote only the older
`semantic_frontier_eligibility_metric` metadata key. The training-contract checker
looked only for the never-written `exact_deployment_eligibility_metric` key. This
metadata/checker mismatch converted a completed adaptation into RC=30 and prevented
all certificate execution.

### Engineering changes

1. New stage metadata writes both:
   - `semantic_frontier_eligibility_metric=true`;
   - `exact_deployment_eligibility_metric=true`, with checkpoint-config provenance.
2. `check_v48_35_training_contract.py` now verifies the actual exact-eligibility bit
   in the factor, identity, and final checkpoint configs. A legacy v48.35 stage file
   is accepted only when its semantic metadata is present **and** every trusted
   checkpoint independently proves exact eligibility. Removing the check or trusting
   metadata alone is not allowed.
3. Added `check_v48_35_resume_contract.py`. No-retraining reuse is authorized only
   for the exact known signature: training-contract raw RC=4 normalized to 30, both
   adaptations RC=0, no certificate/gate/test access, unchanged source/protocol,
   matching checkpoint/support hashes, valid stage transfer, unified physical
   semantics, and exact eligibility in every checkpoint config.
4. Added `RESUME_AFTER_ADAPTATION=1` to the v48.35 controller. Authorization occurs
   before stale status cleanup. A valid resume skips both GPU adaptations, refuses
   index rebuilds, reruns protocol/index/model/training contracts, and then executes
   the original shared-rule certificate path.
5. Added `repair_v48_35_rc30_training_contract_with_v48_35_1.sh` as the operator-facing
   no-retraining repair wrapper.
6. Completion metadata records `adaptation_reused_without_retraining` and the resume
   contract path. Different RC=30 signatures, algorithmic RC=20, changed data/index
   contracts, changed checkpoint bytes, or prior certificate artifacts are rejected.
7. Added regression tests for new metadata, checkpoint-proven legacy repair,
   rejection when one checkpoint disables exact eligibility, resume ordering before
   cleanup, no-retraining behavior, and index fail-closed behavior.
8. Tightened legacy compatibility: an absent new metadata key may be repaired from
   checkpoint evidence, but an explicitly false new key is treated as a contradiction
   and is rejected even when the old semantic key is true.
9. The resume contract additionally checks the final checkpoint hash against the
   controller-level completion record and refuses any pre-existing calibration/GATE
   artifacts that could indicate prior certificate access.
10. The repair wrapper now implements `--help`, rejects positional arguments, and
    cannot accidentally interpret a help request as an experiment launch.

### Interpretation rule

The uploaded v48.35 result is not an algorithm result because the certificate never
ran. Training diagnostics may be used only as debugging signals; they cannot establish
Near/Contact gate effectiveness. The correct next action is to reuse the byte-identical
adaptation checkpoints through the v48.35.1 resume path. Only a resulting valid RC=0
or RC=20 may be used for algorithm attribution. No additional algorithm modification
is introduced before that certificate evidence exists.

### Local validation for v48.35.1

- 17 focused v48.35/v48.35.1 tests passed.
- 174 supported release-matrix tests passed with 6 non-fatal PyTorch warnings.
- 57 shell scripts passed `bash -n`.
- `python -m compileall -q src tools tests` passed.
- The continuous-frontier contract preflight passed all finite-gradient, physical-relative input-isolation, non-compensation, no-regime-ID, and one-shared-rule checks.
- The uploaded result archive does not contain checkpoint `.pt` bytes. Therefore the no-retraining authorization cannot be executed against the archive alone; it is intentionally rechecked on the original experiment machine, where the registered checkpoints must still exist and match their stored SHA256 values.



## v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX

**类别：工程终态修复；算法不变。**

- 未修改 `src/`、`configs/`、模型、损失、训练目标、数据集、certificate 阈值或 gate。
- 未引入 Safe/Near/Contact regime routing；统一连续物理余量机制保持不变。
- 新增 pre-attempt re-entry contract，活动 RC=0/20 重复调用幂等返回。
- 修复拒绝 resume 覆盖已完成 RC=20 的问题；拒绝仅写旁路诊断，不创建 attempt 或替换 terminal markers。
- 新增精确 archived RC=20 恢复，要求 attempt/gate/certificate/Safe/test-seal 一致，失败 byte-for-byte rollback。
- 历史 archive 缺少 `NEXT_COMMANDS.txt`，故 archived RC=0 自动恢复显式 fail closed。
- 新增统一工程恢复入口与 6 个定向测试。


## v48.40 DCFR — Decoupled Context Frontier Reserve (2026-08-07)

### Evidence from v48.39
- Main/A/B/C all terminated as pipeline-valid `RC=20`; the dominant layer is `development_rule_fit`, not an engineering failure.
- Unbounded benefit is rejected: Precision Near opportunity AUC fell from about 0.89 (A/B bounded benefit) to 0.64/0.57 (C/D) on development and from about 0.73 to 0.44/0.38 on certificate.
- Unbounded harm is at most weakly positive and does not repair the shared-rule gate. Dynamic-range expansion is therefore not retained.
- With bounded benefit, Near safe-positive opportunity discrimination remains strong, but certificate safe-positive actions are systematically predicted harmful; conditional safe-positive-vs-harmful harm AUC falls below random in Near certificate.
- Cross-head changes under one-head parameterization ablations indicate gradient coupling through the trainable shared OCAF interaction bridge. This reintroduces a shared bottleneck after v48.21 had already shown benefit/risk adapter decoupling was useful.

### Algorithm changes
1. **Dual task-specific OCAF interaction contexts, still one regime-free policy.** The same candidate-minus-nominal executable action and the same nominal observation feed two independently trainable `ObservationConditionedActionFrontierBridge` branches. Benefit and harm no longer rotate the same interaction parameters. No regime id, regime branch, or regime-specific threshold is introduced.
2. **Frontier-normalized component-margin regression.** Component BCE keeps the exact harmful/safe sign target. The dense physical regression target optionally becomes `s*tanh(m/s)` with `s=0.10`, an odd monotone transform preserving zero and near-boundary resolution while preventing very large violations from dominating the regression geometry. Deployment margins and measured hard vetoes remain in the original physical semantics.
3. **Retained:** bounded HAF benefit factor, factor-preserving/reserve-only deployment, support reliability, aligned deterministic noncompensatory joint reserve, OCAF observation-conditioned physics, top-k=5, and one shared Natural rule.
4. **Explicitly not retained/repeated:** v48.39 unbounded benefit/harm factors, v48.38 one-sided tail losses, learned admission residual, full identity-stage factor updates, grid densification, top-k expansion, aggressive positive oversampling, generic pairwise/listwise restacking, or any Safe/Near/Contact routing.

### Preregistered ablation
- A: shared OCAF + raw component-margin regression.
- B: dual OCAF + raw regression.
- C: shared OCAF + frontier-normalized regression.
- D/main: dual OCAF + frontier-normalized regression.

The intended causal readout is: B>A supports gradient-decoupling; C>A supports boundary-focused target geometry; D>B,C supports complementarity. If D does not improve rare-frontier harm discrimination, do not stack additional losses; inspect observation/teacher identifiability and the dataset statistical-power ceiling.

## v48.45.2 — LOST-SOURCE RECONSTRUCTION SUPPORT (engineering/source protocol; SOWR algorithm unchanged)

### Trigger

The uploaded v48.45 A/B/C archives all terminated before adaptation with normalized `RC=30` at `source_checkpoint_contract`. The historical source run directory no longer exists and both required `candidates/{balanced,precision}/model_v48_trac_sr/best.pt` files are missing. No certificate or gate was executed, so those archives are engineering failures and provide no SOWR algorithm evidence.

The historical v48.13 source also cannot be reproduced byte/recipe-identically from the current archive: `tests/test_v48_13_terra.py` references `scripts/train_ocrap_v48_13_terra.sh`, but that historical training script is absent. v48.45.2 therefore does **not** pretend to recover v48.13.

### Protocol repair

1. `train_ocrap_v48_trac_sr.sh` now supports scratch initialization only behind explicit `ALLOW_SCRATCH_INIT=1 INIT_CKPT=`. Normal source/adaptation training remains fail-closed on a missing checkpoint.
2. Added `rebuild_v48_45_shared_source.sh` with a two-stage reconstruction:
   - S0: one pooled Safe/Near/Contact recovery backbone/witness from scratch, using train and val only; direct proposal loss disabled;
   - S1: the exact same S0 checkpoint is frozen while Balanced/Precision direct proposal/evidence heads are fit on Near/Contact candidate groups.
3. S0 requires both `best.pt` and `TRAINING_COMPLETE.json`. A partial `best.pt` without completion metadata is deleted and retrained instead of being silently reused.
4. The rebuilt source writes `SOURCE_REBUILD_COMPLETE.json` containing source identity and SHA256 for the S0, Balanced and Precision checkpoints.
5. `check_v48_36_source_checkpoint_contract.py` verifies manifest hashes whenever a rebuilt-source manifest exists. Every A/B/C/D arm must therefore consume byte-identical source checkpoints.
6. Added `check_v48_45_rebuilt_source_quality.py`, a train/validation-only structural source preflight that rejects missing/non-finite summaries, empty data, missing checkpoint artifacts, a non-scratch S0, or S1 checkpoints that did not warm-start from the single shared S0. It introduces no calibration/test threshold.
7. v48.45 source resolution now prefers a completed `ocrap_v48_45_source_rebuild_s7` source before the absent historical v48.13 source when `SOURCE_RUN` is not explicitly set. Explicit `SOURCE_RUN` still wins.
8. Recommended A/B/C/D concurrency is `MAX_PARALLEL_ARMS=1`: each arm already launches Balanced on GPU0 and Precision on GPU1, so this uses exactly two GPU training processes. `MAX_PARALLEL_ARMS=2` would launch up to four training processes and is not recommended for the first rebuilt-source round.

### Attribution contract

- SOWR A/B/C/D definitions, dual-ROCT settings, top-k, shared Natural rule, harm budgets, calibration/certificate protocol and gate are unchanged.
- All four arms must use the same rebuilt source manifest hashes.
- Results are valid for causal comparison **within the rebuilt-source round**.
- Absolute metrics must not be interpreted as directly comparable to historical v48.44/v48.45 runs whose source checkpoint was different and is now lost.
- No per-arm random warm start and no per-arm backbone retraining is allowed.
- Source rebuild reads no calibration/certificate/test roots; those remain downstream protocol-only data.

### Unified-regime note

S0 pools Safe/Near/Contact for recovery-witness learning and does not add any new SOWR regime router or regime-specific admission rule. The inherited v48 source-policy checkpoint geometry still contains the legacy two `direct_delta_adapters`; this pre-existing design debt is held fixed in the rebuilt round to avoid confounding source recovery with a separate algorithm refactor. If strict removal of all bucket-conditioned policy internals is required for the final paper, it must be evaluated as a separate controlled experiment after SOWR attribution.
