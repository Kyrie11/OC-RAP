# OC-RAP v48.52 结果归因与 v48.53 CSE 设计

## 1. 结论先行

v48.52 是一个有效、可归因的单轴实验，但其结果明确 **reject teacher-only Physical Sign Alignment (PSA)**。A/B 都是 pipeline-valid authoritative RC20，certificate 与 Natural gate 实际执行，test roots 未读取；A/B protocol/source/gate identity一致，A/B factor contract只在 teacher physical sign factor上不同。因此 B-A 可以作为算法证据，而不是工程噪声。

v48.52 的负结果不是“physical certificate没有意义”，而是说明了一个更严格的必要条件：**仅将 teacher hard sign 替换为 physical certificate，同时让 student/deployment hard DRS继续用 q>=0 event，会造成监督目标与可部署 representation 的结构不对称。**

因此 v48.51 的两个核心结论仍被保留，但需要进一步精炼：

1. hard/exact coordinate必须参与 upstream calibration；
2. hard coordinate只负责 material sign，smooth q geometry负责 hard-equivalence class 内的 continuous order/magnitude；
3. **新增约束：teacher、student-training 与 deployment 的 hard certificate必须具有相同的 composition，而不仅仅“都拥有一个 zero crossing”。**

这就是下一版的 **Certificate Structural Equivalence (CSE)**。

当前不应该进入 Boundary-Complete Evidence Centering，因为 v48.52 没有先解决 physical sign correctness；相反，它让 physical/native geometry变差。必须先检验双边结构同构，再决定 final evidence centering是否成为下一层瓶颈。

---

## 2. v48.52 attribution validity

本轮 A/B audit 的 attribution contract valid：

- source checkpoint SHA 一致；
- protocol seal一致；
- calibration/certificate dataset manifests一致；
- shared deployment rule一致；
- proposal top-k仍为 5；
- option execution semantics均为 `observation_class`；
- `strategy_regime_conditioning=false`；
- `test_roots_read=false`；
- 两臂均为 authoritative RC20 / pipeline valid；
- B-A 唯一算法因子是 teacher-side PSA。

历史 v48.51-B 没有被错误复用：reuse contract检测到 protocol-seal SHA mismatch，因此 launcher自动跑了 fresh v48.52-A。这反而让本轮 A/B 的 identity 更干净。

---

## 3. v48.52 的定量归因

### 3.1 Near-contact

| metric | A: BC-FC + smooth NAP | B: A + teacher PSA | 方向 |
|---|---:|---:|---|
| certificate recall | **0.333** | 0.222 | 明显变差 |
| certificate harmful UCB90 | **0.042** | 0.074 | 变差 |
| certificate candidate safe-positive AUC | **0.448** | 0.397 | 变差 |
| proposal safe-positive AUC | **0.491** | 0.444 | 变差 |
| development recall | **0.375** | 0.125 | 明显变差 |
| development raw precision | **0.088** | 0.032 | 变差 |
| development joint semantic sign | **4/19** | 1/19 | 明显变差 |
| development boundary-complete positive | **5/19** | 2/19 | 变差 |
| development exact-positive | **2/19** | 1/19 | 变差 |
| development DRS safe-positive veto | **0/19** | 2/19 | 变差 |
| development DRS harmful false-safe | **57/136** | 83/136 | 变差 |

Near 的结论非常清楚：PSA没有“提高物理正确性但影响 final centering”，而是先把上游 certificate geometry本身破坏了。因而不能触发 evidence-centering分支。

值得注意的是，A 仍维持 v48.51-B 的核心优势：Near recall 0.333、harmful UCB90 0.042。这意味着 **BC-FC 的 hard-sign / smooth-order decomposition没有被 v48.52推翻**；被推翻的是 teacher-only physical redefinition。

### 3.2 Contact

| metric | A | B / PSA | 方向 |
|---|---:|---:|---|
| certificate recall | **0.050** | 0 | 变差 |
| certificate harmful UCB90 | **0.351** | 0.473 | 明显变差 |
| candidate safe-positive AUC | **0.632** | 0.540 | 明显变差 |
| proposal safe-positive AUC | **0.611** | 0.521 | 明显变差 |
| development exact-positive | **7/37** | **0/37** | 关键 collapse |
| development boundary-complete positive | **7/37** | **0/37** | 关键 collapse |
| development final joint sign | 0/37 | 0/37 | 都未闭环 |
| DRS safe-positive veto | 11/37 | **7/37** | 单项改善 |
| DRS harmful false-safe | **341/432** | 370/432 | specificity变差 |

Contact 中 PSA 确有一个局部现象：safe-positive DRS veto从 11/37降到 7/37。但 harmful false-safe从 341/432升到 370/432，candidate/proposal ranking下降，而且最关键的 exact/native positive geometry从 7/37变为 0。因此它不是 Pareto improvement，不能作为值得保留的 mechanism。

更重要的是，这一结果证明 **Contact 的主要问题不能简单概括为“teacher physical label不够正确”**。A 中已经存在 7/37 latent exact-positive，但 final opportunity/pred-adv仍是 0/37；B 则连 latent physical sign都丢掉了。下一步必须先解释为什么 physical teacher target反而让 student representation更差。

---

## 4. 为什么 teacher-only PSA 会失败

代码级结构审计给出了与结果一致的解释。

### v48.52-B teacher hard certificate

```
teacher OC-MERO q
    -> 按 observation-class q 选择 recovery option
    -> 在所选 option 上读取 teacher m_star
    -> m_star >= 0 判 root physical success
    -> teacher root probability mass 聚合 DRS
    -> exact PCD
```

### v48.52-B student / deployment hard certificate

```
predicted OC-MERO q
    -> 每个 root 直接判断 q_best >= 0
    -> predicted root probability mass 聚合 DRS
    -> exact PCD
```

两边的 hard event 并不是同一个映射。

`q` 是对 observation-compatible roots 的 lower-tail robust recovery value，包含 robust aggregation与 observation-consistency coupling；`m_star(selected)` 是具体 root、具体被执行 action 的 physical margin。两者应当相关，但不可能被假定为同一个二元事件。

v48.52 把 target从 q-event改成 physical-margin event，却没有给 student/deployment提供相同的 composition；同时 order channel仍要求 q 保持 smooth depth。于是同一个 `margin_head` 同时收到：

- smooth channel：保持 q-depth / q-ranking；
- hard sign channel：用 q-hard student logit去拟合 m_star-selected teacher event。

这就是 supervision/representation structural mismatch。结果上它表现为 Near/Contact ranking、exact sign、DRS specificity一起退化。

因此 v48.52 的可靠结论是：

> **Semantic equivalence不能只定义在 teacher label上；它必须是 teacher–student–deployment 三方结构同构。**

---

## 5. 当前 dominant bottleneck 的重新定位

v48.50：主要矛盾是 boundary quantization vs local ordering。

v48.51：BC-FC 部分解决了这个问题，Near 出现明显正向结果。

v48.52：teacher-only PSA 暴露出更上游的 **certificate structural equivalence** 问题。

因此当前优先级为：

1. **Teacher–student–deployment certificate structural equivalence**；
2. DEP/GAP teacher/native component correctness 与 normalization；
3. predicted-root reliability（共同绝对瓶颈候选，但不能解释 v48.52 B-A，因为 root head在 A/B冻结且 source相同）；
4. **只有在 1 被支持后**，才能把 final evidence centering提升为 dominant bottleneck。

### 为什么现在不做 Boundary-Complete Evidence Centering

上一版预注册条件是：“如果 PSA 改善 physical sign correctness，但 learned opportunity/pred-adv仍负，则做 evidence centering”。

v48.52 实际发生的是：PSA **没有改善 physical sign**，甚至使 Contact exact-positive 7/37 -> 0/37。前置条件没有满足。因此现在进入 evidence centering会把 upstream certificate错配与 downstream centering混在一个实验里，失去清晰归因。

---

## 6. predicted-root reliability 和 DEP/GAP 该如何看

### predicted-root reliability

它仍可能限制绝对性能，但本轮 A/B 的 root-logit head在 witness stage冻结，source checkpoint一致，因此它不能是 PSA 负效应的 causal source。历史 changelog 中 root-logit recalibration已有 stop signal，因此 v48.53不重开“重新校准 root logits”路线。

如果 v48.53 CSE失败，下一步可以审计 root probability 的 reliability / uncertainty representation，但必须作为新的诊断问题，而不是换名字重复 root-logit recalibration。

### DEP/GAP

A/B 的 component geometry仍显示 DEP/GAP存在明显 sensitivity/specificity tradeoff，特别是 Contact safe-positive veto仍高。因此它们仍是后续可能瓶颈。

但 v48.53 暂不改 DEP/GAP formula/normalization，因为同时改它会让 CSE无法单独归因。若 CSE失败，才进入 DEP/GAP teacher normalization与 component correctness单轴实验。

---

## 7. Safe / Near / Contact 的统一 planning primitive 状态

### Safe

Safe standard calibration仍 valid，但 A/B都是 RC20，因此 authoritative paired Safe non-inferiority与 closed-loop正确地没有执行。当前不应降低 gate 去“解决 RC20”。否则会失去 critical regime 的 failure localization。

正确顺序仍是：critical mechanism correctness -> RC0 -> Safe paired non-inferiority + closed-loop。

### Near-contact

Near 已经明确有 recovery能力。A 继续达到 0.333 recall / 0.042 harmful UCB。这说明论文不是在证明“有没有 recovery signal”，而是在解决 unified primitive 如何同时保持：

- material sign；
- local ordering；
- precision / centering；
- component specificity。

PSA使这些指标退化，因此不进入主算法。

### Contact

Contact 仍应把 hard safe-benefit certificate作为最终 intervention authorization criterion，但它不应该是唯一 dense representation target。A 已有 candidate safe-positive AUC约0.632、proposal约0.611和 latent exact-positive 7/37，说明 model不是完全不会 prefix-action ranking。

现实目标仍应是同一个 planner对 prefix action的 post-contact recoverability进行连续排序，并在物理 certificate达到 material improvement时授权 intervention。二次碰撞/re-contact概率仍适合在 RC0后的 closed-loop endpoint验证；当前数据没有独立概率 label，不应凭空加入新监督。

---

## 8. v48.53：Certificate Structural Equivalence (CSE)

### 方法定义

v48.53 不新增一个新 policy module，而是把 BC-DE 的 semantic contract补完整。

Teacher physical certificate：

```
teacher q selects observation-consistent option
-> teacher selected m_star >= 0
-> teacher root mass
-> hard DRS
```

Student / deployment physical certificate：

```
predicted q selects observation-consistent option
-> predicted selected margin >= 0
-> predicted root mass
-> hard DRS
```

两边只把 hard sign composition结构对齐。

Smooth order channel保持完全不变：

```
smooth q-boundary DRS + smooth PCD
```

因此仍然贯彻：

> hard owns material sign; smooth owns local order.

但进一步要求：

> hard sign teacher and hard sign student must be structurally realizable by the same certificate composition.

### student-side gradient

- forward option selection：hard q argmax，与 teacher的 observation-consistent semantics一致；
- forward root success：`selected predicted margin >= 0`；
- backward：只在 q 已选中的 predicted margin上使用 sigmoid straight-through gradient；
- 不对 hard option selection做 soft router；
- q smooth order channel继续为 q提供 continuous gradient；
- physical margin zero crossing固定为 0，不使用 q gamma平移。

### 参数/容量

0 新 head、0 新参数、0 新 threshold、0 regime conditioning。

---

## 9. 为什么必须做 2×2，而不是只跑 symmetric D

v48.52 已知：teacher-only physical (B-A) 是负向。

若下一版只跑 D，我们无法知道 D 的变化来自：

- student physical factor本身；还是
- teacher/student symmetry interaction。

因此 v48.53 采用：

| Arm | teacher physical X | student/deployment physical Y |
|---|---:|---:|
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 1 |
| D/Main | 1 | 1 |

关键读取：

1. B-A：已知 teacher-only PSA；
2. C-A：student-only physical effect；
3. D-B-C+A：真正的 structural-equivalence interaction。

如果 B 与 C 单边均负，而 D显著正，这反而是最强的 CSE 证据：说明关键不是“physical factor本身”，而是双边结构匹配。

---

## 10. v48.53 预注册判断

### Near

- D certificate recall机制 screen：`>=0.25`；
- D harmful-selected UCB90：`<=0.15`；
- development joint sign：至少恢复到 A 的 `4/19` 量级；
- candidate/proposal safe-positive ranking不能继续 PSA 型 collapse；
- DRS safe-positive false-veto 与 harmful false-safe是首要 component readout。

### Contact

第一优先级不是立刻要求 RC0，而是检查 v48.52-B 的 physical collapse是否被结构同构修复：

- development exact/native positive应不再为 0/37；
- 恢复到 A 的 `7/37` 量级是强机制信号；
- candidate/proposal safe-positive AUC应明显恢复；
- harmful UCB应相对 B下降；
- certificate recall恢复到 `>=0.05` 说明恢复 A 信号，`>=0.10` 才视为明显进一步进展。

### 下一层分支

**如果 D 改善 physical/native geometry，但 final opportunity/pred_adv仍负偏：**

此时 teacher/student/deployment correctness才真正获得证据，下一版进入单轴 Boundary-Complete Evidence Centering。

**如果 D 仍失败：**

停止 BC-FC/CSE/transport family，不再调 hard/smooth loss weight；转 DEP/GAP teacher normalization、teacher component correctness、predicted-root reliability诊断。继续尊重 root-logit recalibration、threshold、top-k、oversampling、generic pairwise/listwise、learned admission residual等历史 stop signal。

---

## 11. 运行效率

v48.52 A/B 每臂约 42.4 / 43.0 分钟，并行总墙钟约44分钟。GPU平均利用率仍只有约18–20%，显存峰值约537MB；pipeline仍是小粒度 inference / CPU / I/O受限，而不是算力受限。

v48.53 保留已有 checkpoint/config-SHA standard-calibration prediction cache，不再引入有数值风险的优化。

更大的加速来自实验设计：默认严格复用刚得到的 v48.52 A/B，只新跑 C/D，各占一张 GPU，因此仍只需要大约一个 arm wave，而不是重新跑四臂。若 A/B identity/protocol不一致则 fail-closed自动回退 fresh四臂。

---

## 12. 论文主线建议

不建议论文最终写成：

`OC-MERO + NCP + DRFC + NAP + BC-FC + PSA + CSE`

更好的 CCF-A 级主线是一个原则：

**Observation-Consistent, Boundary-Complete, Structurally Decision-Equivalent Recoverability**

或更短：

**Physical Boundary-Complete Decision Equivalence**

其中三个必要条件是：

1. observation consistency；
2. hard-sign / smooth-order boundary completeness；
3. teacher/student/deployment certificate structural equivalence。

v48.50–v48.53 的消融不是“连续加模块”，而是在逐层验证这三个必要条件，并用失败实验排除更弱的定义。

这比继续增加 learned router、regime-conditioned policy或 residual head更能维持论文 novelty 与机制完整性。
