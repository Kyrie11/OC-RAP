# OC-RAP v48.53 结果归因与 v48.54 IPBD 设计

## 1. 结论摘要

v48.53 的 2×2 attribution contract 有效，因此可以严格解释 B-A、C-A 与 D-B-C+A。结果**不支持** v48.53 提出的强 CSE 命题：teacher/student/deployment 都采用 q-selected margin-physical hard certificate 并没有恢复 reference A，D/Main 在 Near/Contact 都没有形成 Pareto improvement。

截至本轮最稳健的机制原则仍是：

- observation consistency 是必要约束；
- deployed discontinuous **q-hard certificate 负责 material decision sign**；
- **smooth q geometry 负责 hard-equivalence class 内 continuous ordering**；
- deployed hard/exact boundary 必须进入 upstream learning/calibration；
- 但 decision equivalence **不要求内部 certificate computation 做 structural imitation**。

v48.53 新增的关键结论是：selected-option physical margin 有真实的信息价值，但最适合作为 **privileged training signal**，而不是 native/deployment hard DRS。

因此下一版不进入 evidence centering，也不继续 PSA/CSE，而做单轴 **Invariant-Preserving Physical Boundary Distillation (IPBD)**。

---

## 2. v48.53 2×2 严格归因

四臂：

- A：q-proxy teacher + q-hard student/deployment；
- B：teacher physical、student q-hard；
- C：teacher q-proxy、student/deployment physical；
- D：teacher + student/deployment 都 physical。

Precision 关键结果：

| regime | metric | A | B | C | D |
|---|---|---:|---:|---:|---:|
| Near | cert recall | **0.333** | 0.222 | 0.222 | 0.111 |
| Near | harmful UCB90 | 0.042 | 0.074 | **0.039** | 0.044 |
| Near | candidate safe-positive AUC | 0.448 | 0.397 | **0.535** | 0.427 |
| Near | dev recall | **0.375** | 0.125 | 0.250 | 0 |
| Near | dev joint sign | **4/19** | 1/19 | 2/19 | 0/19 |
| Near | dev DRS harmful false-safe | **0.419** | 0.610 | 0.603 | 0.868 |
| Contact | cert recall | **0.050** | 0 | 0 | 0 |
| Contact | harmful UCB90 | **0.351** | 0.473 | 0.473 | 0.591 |
| Contact | candidate safe-positive AUC | 0.632 | 0.540 | **0.655** | 0.608 |
| Contact | proposal safe-positive AUC | **0.611** | 0.521 | 0.600 | 0.558 |
| Contact | dev exact/native positive | **7/37** | 0/37 | 2/37 | 1/37 |
| Contact | dev DRS safe-positive veto | 11/37 | 7/37 | **0/37** | **0/37** |
| Contact | dev DRS harmful false-safe | **0.789** | 0.856 | 0.917 | 0.984 |

### B-A：teacher-only physical PSA 再次确认负向

B 延续 v48.52 负结果。Near recall、ranking、development sign 下滑；Contact recall 从 A 的 0.05 归零。没有理由重新调 PSA strength、temperature 或 loss weight。

### C-A：最重要的“局部正信号 + 明确副作用”

C 是本轮最有研究价值的 arm，但不是可直接进入 Main 的机制。

正面：

- Near candidate safe-positive AUC：0.448 → **0.535**；
- Contact candidate safe-positive AUC：0.632 → **0.655**；
- Near harmful UCB90 略从 0.0419 → **0.0394**；
- Near/Contact DRS safe-positive false-veto 均降到 0；
- Near development opportunity/pred-adv 的中心也比 B/D 更接近正侧。

负面：

- Near certificate recall仍下降到 0.222；
- Contact certificate recall从 0.05 降到 0；
- Contact harmful UCB90从 0.351 恶化到 0.473；
- Contact development DRS harmful false-safe从 0.789恶化到 0.917。

因此 physical margin 提高了某些 sensitivity / candidate discrimination，却显著损害 specificity。它**有信息但不适合接管 hard sign**。

### D interaction：不能误读为 CSE 成功

Contact recall interaction `D-B-C+A = +0.05`，但这是：

`0 - 0 - 0 + 0.05 = +0.05`。

也就是 A 唯一有 0.05 recall，B/C/D 全为 0 造成的 floor arithmetic。它不是 D 的协同恢复。

Contact exact/native-positive interaction约 +0.162 同理：D=1/37，虽然相对“两个负主效应的加法预期”更好，但仍远低于 A=7/37。与此同时 D 的 Contact harmful UCB90达到 0.591、DRS harmful false-safe达到 0.984。

因此“B、C各自负、D interaction正”只有在 **D 的绝对机制指标也恢复/超过 A，并且 safety/specificity 不恶化**时才能作为 CSE 证据。本轮不满足。

---

## 3. 哪些命题继续成立，哪些应被否定

### 继续成立

1. hard q-based coordinate负责 material sign，smooth q geometry负责 local order；
2. deployed hard boundary必须参与 upstream calibration；
3. Safe/Near/Contact 应由一个统一 primitive处理，不引入 regime-conditioned policy；
4. proposal availability不是首要瓶颈，当前 development 仍有可行 safe-positive proposals。

### 本轮进一步否定

1. **teacher physical hard target 单边替换会更正确**：已由 v48.52/v48.53-B否定；
2. **student/deployment margin-physical hard DRS 单边替换会更正确**：C 否定；
3. **teacher/student/deployment 只要 structural-equivalent 地都采用 margin-physical hard certificate 就会恢复**：D 否定；
4. **正 interaction 本身足以证明协同机制**：否定。interaction 必须与绝对性能、specificity、安全指标共同解释。

CSE 因此不应继续作为论文“必要条件”。它应作为一个高价值的 falsified ablation，帮助论文提出更精确的原则：

> Decision equivalence is preservation of decision-relevant invariants, not structural imitation of every privileged physical coordinate.

---

## 4. 当前 dominant bottleneck

当前第一瓶颈是 **q-hard specificity 与 root-level physical-margin sensitivity 之间的职责冲突**。

A 的 q-hard DRS 在 Near/Contact 保留更好的 harmful specificity 和最终 recall；C 的 margin-physical DRS消除了 safe-positive veto并改善 candidate discrimination，但把大量 harmful roots判成 false-safe。

所以现在不是“选 q 还是选 physical margin”，而是：

> 如何把 physical margin 中有用的 boundary information 注入 latent witness，同时不改变部署 q-hard certificate 的 material decision semantics？

第二层瓶颈仍包括 DEP/GAP component correctness/normalization；predicted-root reliability也可能限制绝对性能。但本轮 source/root head在因素间冻结，所以它们不能解释 C-A 的局部 ranking gain或 specificity collapse。

final evidence centering **还不是下一步**。只有 upstream q-hard certificate在保留 specificity的同时吸收 physical boundary信息后，若 opportunity/pred-adv仍系统性负偏，才能干净地把 centering定位为主因。

---

## 5. v48.54：Invariant-Preserving Physical Boundary Distillation (IPBD)

### 设计原则

部署 hard sign 完全回到 A/reference：

- teacher sign：q-hard proxy；
- student/deployment sign：q-hard DRS；
- smooth order：smooth q DRS/PCD；
- physical teacher alignment=false；
- physical student/native DRS=false。

只有训练期增加一个 privileged auxiliary：

1. teacher OC-MERO q选择 observation-consistent option；
2. 读取该 teacher-q-selected option 的 teacher `m_star`；
3. target=`1[m_star_selected >= 0]`；
4. 监督同一个 option 的 predicted margin zero crossing；
5. teacher root probability weighted + class-balanced BCE；
6. physical margin 0 是唯一 boundary；temperature复用 0.08，loss weight复用 0.50；
7. privileged target不参与 routing、不进入 native DRS、不改 threshold/admission。

因此它不会重复 v48.53 的 structural imitation：physical margin只教模型“被 q 选中的恢复动作在物理边界哪一侧”，但最终是否授权仍由 q-hard decision-equivalent certificate决定。

### 为什么它值得做

C 已经证明 physical margin含有 candidate ranking/sensitivity信息；A 已经证明 q-hard certificate拥有更好的 specificity。IPBD 是对这两条证据最小、最可证伪的组合，而不是参数搜索。

它也没有重复 changelog 中的：

- one-sided component penalty；
- generic pairwise/listwise；
- learned admission residual；
- BC-NAP / exact-only NAP；
- root-logit recalibration；
- threshold/top-k扩张；
- regime-conditioned policy。

---

## 6. v48.54 实验设计

严格 A/B：

- A：v48.53-A q-hard BC-FC + smooth NAP reference；
- B/Main：A + IPBD。

默认做 semantic reference reuse。旧 v48.53 launcher因为 protocol seal文件包含 transient creation metadata而无法复用 v48.52 A/B；v48.54 不再比较无关的 seal byte SHA，而比较：

- source checkpoint SHA；
- Safe/Near/Contact 五个 canonical manifest SHA；
- gate semantic protocol；
- reference factor contract。

任何真实 semantic mismatch仍然 fail-closed fresh A。

### 预注册 readout

Near：

- recall至少 >=0.25，优先恢复 A=0.333；
- harmful UCB优先 <=0.05；
- candidate safe-positive AUC希望 >A=0.448；
- development joint sign希望维持 A≈4/19。

Contact：

- candidate safe-positive AUC至少 >=A=0.632；
- proposal safe-positive AUC不应明显低于 A=0.611；
- harmful UCB不得重复 C/D 型恶化，参考 A=0.351；
- certificate recall首先保持/恢复 A=0.05；
- DRS harmful false-safe不能向 C/D方向恶化。

如果 physical/ranking evidence改善且 q-hard specificity保持，但 opportunity/pred-adv仍负偏：下一版才进入 Boundary-Complete Evidence Centering。

如果 IPBD失败：停止 physical-margin distillation family，进入 DEP/GAP teacher normalization / component correctness / root-uncertainty diagnosis；仍不重开 root-logit recalibration。

---

## 7. CCF-A 主线

建议论文主线从 v48.53 的“Certificate Structural Equivalence”继续推进，而不是维持该强命题：

**Observation-Consistent, Physical Boundary-Complete Decision Equivalence with Invariant-Preserving Privileged Boundary Distillation**。

论文不应写成一串模块名。方法层面可以收敛成：

- observation-consistent recovery abstraction；
- hard/smooth boundary role separation；
- invariant-preserving learned transport；
- privileged physical boundary supervision；
- calibrated non-compensatory deployment gate。

v48.52/53 的负消融反而可以成为方法论证的一部分：semantic/structural imitation都可能破坏 specificity；真正需要保留的是最终决策不变量，而非复制所有 privileged internal coordinates。

当前距离 CCF-A 主要仍在 empirical closure：所有 v48.53 arm仍 RC20，Contact recall最好仍只有 A 的 0.05，Near precision LCB远低于 gate要求，Safe authoritative paired non-inferiority/closed-loop尚未解锁。因此 v48.54 的目标仍是机制闭环，不应通过放宽 gate来“制造”Safe结果。
