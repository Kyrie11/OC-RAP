# OC-RAP V48.55 实验归因与 V48.56 TCSA 设计报告

日期：2026-08-19  
结论版本：`v48.56-DCP-DRFC-BCDE-TCSA`  
定位：**teacher/component semantic correctness release；不是 performance Main。**

---

## 0. 结论先行

这轮**不应该进入 Boundary-Complete Evidence Centering**。V48.55 的 TCBC 2×2 没有满足上一轮预注册的 centering 前提：B/C/D 并没有在 Near+Contact 上形成新的 component/native geometry Pareto front。相反，Near certificate recall 从 A 的 0.333 降到 B/C/D 的 0.222；Contact 只有 D 出现局部 recall/UCB interaction，但 development native geometry 完全不动。

因此严格执行上一轮 stop rule：

1. **STOP component-normalization family**；不再调 pooled RMS、target scale、per-component scale 或 nonlinear normalization。
2. **直接进入 teacher component correctness audit**；优先级高于 evidence centering，也高于 root uncertainty/root-logit recalibration。
3. V48.56 只做 correctness：审计 DEP/GAP/teacher source/component construction，并新增 `strict_min_slack` teacher shadow。
4. 在 strict teacher audit 通过之前，不把 legacy PCD 换成 R_dep rescue 直接训练，因为当前 R_dep 正边界出现高度可疑的 0.5 plateau。
5. 论文主线进一步收敛为：

> **Observation consistency + boundary-complete decision equivalence + decision-role-consistent calibration.**

“不同数学类型的 component 应使用不同 calibration geometry”仍然成立，但只是必要条件；**decision role 必须先于 coordinate type**。

---

## 1. 论文与代码主线对齐

论文的真正中心问题不是“碰撞后专用 controller”，而是：**当前 candidate prefix 是否保留了部署时可从 post-prefix observation 选择的 recovery affordance**。OC-RAP 的方法链是：

`scene/prefix → recovery-sufficient roots → post-prefix observation equivalence → recovery-option margin → OC-MERO R_dep/R_orc → calibrated admission/action selection`。

其中最关键的部署量是：

- `R_orc`：每个 latent root 可以先知道 branch identity 再选 recovery option；
- `R_dep`：先按 post-prefix observation collapse indistinguishable roots，再要求共享可执行 recovery option；
- `G = R_orc - R_dep`：oracle-to-deployable gap；
- `DRS`：共享/observation-consistent option 的 recovery success proxy。

论文 teacher margin 明确写成 active recovery constraints 的 normalized signed slack 的最小值；CRISP 的核心 admission 是 `R_dep >= gamma_rec`，再叠加 hard-rule / harm constraints。这个结构意味着：**R_dep 是 material deployability primitive；GAP/ODG 首先是 oracle-artifact diagnostic，除非另有证明，不应自动变成与 R_dep 同层级的 hard certificate veto。**

当前代码的 authoritative mainline已经坚持：

- `option_execution_semantics=observation_class`；
- `strategy_regime_conditioning=false`；
- Safe/Near/Contact 只是 dataset/evaluation strata；
- shared deployment rule；
- q-hard material sign + smooth local order；
- no test roots in fitting/calibration selection。

需要同步修论文的一处明显冲突：附录当前仍有 `Regime-conditioned recovery admission` / contact-only protective certificate 描述。若最终代码继续 shared/no-regime rule，这段必须删除、改成统一 rule 下的 regime-wise evaluation，或者明确仅作为未采用的 alternative；否则论文算法陈述和 authoritative experiment contract 会冲突。

---

## 2. V48.55 attribution validity

A semantic reuse contract有效：source checkpoint、canonical dataset manifests、gate semantics均一致；`strategy_regime_conditioning=false`、`test_roots_read=false`。B/C/D 的 factor contract 与 A 的 attribution identity一致，所以本轮可以按预注册顺序做因果归因：

- **B−A**：只测试 DRS sign-only / remove DRS continuous component regression；
- **C−A**：只测试 DEP/GAP pooled train-only linear RMS canonicalization；
- **D−B−C+A**：测试二者 interaction；
- hard veto、q-hard deployment、smooth order、source/data/top-k/gate均固定。

V48.55 pooled train RMS 本身确实存在量纲差异：DRS≈0.419、DEP≈0.223、GAP≈0.356；C/D 使用 DEP/GAP 线性 canonicalization，并保持 zero crossing / within-component order，不是 v48.40 的 tanh 重跑。因此该 2×2 对“scale mismatch”已经是一次干净、值得接受其否证结果的实验。

---

## 3. B−A：DRS sign-only 没有解决 deployed geometry

### Near

- cert recall：`0.333 → 0.222`，**−0.111**；
- harmful UCB90：`0.04189 → 0.04242`，轻微变差；
- candidate safe-positive AUC：`0.44753 → 0.44579`；
- proposal safe-positive AUC：`0.49113 → 0.48316`，明显更差；
- dev recall：`0.375 → 0.375`；
- dev DRS/DEP/GAP false-safe/false-veto：**逐项 0 变化**；
- dev opportunity/pred-adv：**0 变化**。

### Contact

- cert recall：`0.05 → 0.05`；
- harmful UCB90：完全相同 `0.350808`；
- candidate/proposal safe-positive AUC变化只有 `1e-5~1e-4`；
- dev component/native geometry：**逐项 0 变化**。

### 归因

X 不能作为“DRS 数学类型修正成功”的正机制。它最多说明：在当前 architecture/deployment overwrite 下，去掉 explicit DRS magnitude regression 并不足以推动 deployed DRS geometry，且 Near certificate selection受到负扰动。

但这**不否定 v48.50–51 的 BC-FC 原则**。历史证据支持的是更精确的命题：

> hard/discontinuous coordinate负责 material sign；smooth q geometry负责 hard-equivalence class 内 local order。

V48.55 X 测的是“把一个 component regression loss拿掉”是否足以改 deployed geometry，不等同于 BC-FC 本身。

---

## 4. C−A：DEP/GAP RMS canonicalization 不是 dominant bottleneck

### Near

- recall：`0.333 → 0.222`，−0.111；
- harmful UCB90：`+0.00053`；
- candidate AUC：`+0.00060`；
- proposal AUC：`+0.00089`；
- dev所有 native component geometry：0变化。

### Contact

- recall：0变化；
- harmful UCB90：`0.35081 → 0.35646`，**变差 +0.00565**；
- candidate/proposal AUC几乎不变；
- dev所有 native component geometry：0变化。

### 归因

这已经足以对 “cross-severity scale inconsistency 是主 causal bottleneck” 给出 stop signal。线性 RMS normalizer没有把 Near/Contact 的 component Pareto修正到新的位置。继续搜索 RMS target scale、per-regime RMS、robust scale、tanh/clip，只会重新回到已经被历史 changelog否决的 normalization/tolerance search family。

**STOP component-normalization family。**

---

## 5. D−B−C+A：Contact 局部 interaction 不构成 TCBC complementary evidence

Contact：

- recall interaction：`+0.05`，D 达到 0.10；
- harmful UCB interaction：`−0.02590`，D 从 A 的 0.35081 到 0.33056；
- 但 D 仍高于 Contact verify cap 0.25；
- candidate/proposal AUC无实质提升；
- Contact dev recall仍为 0；
- Contact dev joint semantic eligible仍为 0；
- DRS/DEP/GAP native component false-safe/false-veto全部与 A/B/C完全相同。

Near：

- D recall仍 0.222，低于 A 的 0.333；
- harmful UCB进一步轻微变差到 0.04297；
- interaction只在 proposal AUC有 +0.0084 的 arithmetic recovery，但绝对幅度仍不足以抵消 Near recall损失；
- dev native geometry仍然完全相同。

因此 D 的 Contact improvement 更像 downstream selection/ranking 的小 interaction，而不是 upstream certificate construction 获得了新正确几何。根据上一轮预注册标准，**positive arithmetic interaction不能替代 D 的 absolute component/native Pareto**。

结论：TCBC Main不成立；不进入 evidence centering。

---

## 6. 为什么现在不能做 Boundary-Complete Evidence Centering

上一轮定义的 centering前提是：

> upstream ranking / physical-boundary / component geometry 已经在 Near+Contact 同时改进，只剩 safe-positive opportunity / pred-adv 系统性落在负侧。

V48.55 不满足：

- Near recall下降；
- Contact dev recall仍 0；
- component/native geometry不动；
- C scale factor甚至让 Contact UCB变差；
- D 只有 certificate selection层局部收益。

此外代码路径还有一个更强的解释：当前 `native_*_preservation` 会把部分 learned component/benefit readout覆盖回 native q-hard/smooth-NAP路径，所以 X/Y 本来就没有直接获得“移动最终 deployed component geometry”的自由度。此时观察到 pred-adv中位数不变，不能当成“模型已经学对，只差偏置中心”的证据。

**所以现在做 centering会把上游 semantic mismatch藏到最后一层 bias/offset 中，论文机制会变成结果驱动的修补。**

---

## 7. 上轮四项 correctness audit：本轮答案

### 7.1 DEP target是否正确表达 deployment notion？——目前不是同一个 notion

当前 component DEP harm term是：

`sigmoid(R_dep_nom) - sigmoid(R_dep_cand) - tolerance`。

这表达的是 **nominal-relative deployability preservation / non-inferiority**，不是论文的 **absolute deployable recoverability admission boundary**。

两者都可能有价值，但必须分角色：

- `R_dep_cand >= gamma/0`：material deployability/admission；
- `R_dep_cand`相对 nominal的下降：secondary preservation/non-inferiority。

把第二个叫“deployability component”并让它和 absolute recovery notion混用，会导致语义不透明：candidate可以仍低于 absolute deployment boundary却因为“没比 nominal差太多”通过 relative DEP；也可以始终可部署却因 nominal更高而被处罚。

**V48.56 不立即改 target训练；先把角色审计并在 strict source上验证 support。**

### 7.2 GAP quality定义/方向/normalization？——方向正确，role比normalization更有问题

`gap_quality=exp(-max(gap,0))` 的方向是正确的：ODG越小，quality越高。线性/指数变换也保持单调。

真正的问题不是方向，而是：**GAP是否应是 hard non-compensatory veto**。

`G=R_orc-R_dep` 可以在两种完全不同情况下变小：

1. `R_dep`提高，真正更 deployable；
2. `R_orc`和`R_dep`都很差，但两者接近——gap小并不意味着安全。

反过来，candidate可以让 `R_dep`跨过 material boundary，同时 oracle上界提高得更多，于是 gap变大；若把 gap hard veto，就会拒绝真实 deployability rescue。

论文 CRISP 核心也没有把 ODG/GAP写成独立 material admission约束。因此当前证据更支持：

> GAP优先作为 **diagnostic / anti-oracle regularizer / tie-break order evidence**；若要作为 hard veto，需要另做专门因果实验和理论定义。

### 7.3 teacher source label是否与实际 deployed veto一致？——DRS execution semantics大体对齐，但 source freshness必须核对

teacher index的 DRS 是先 fresh OC-MERO q，再按 `observation_class`选 option并计算 shared recovery success；这与当前 deployment notion一致。

但是 index优先读取 stored `r_dep_star/r_orc_star`，没有在历史 pipeline中强制逐样本 assert它们和 fresh OC-MERO一致。因此 V48.56 新 audit工具提供 source dataset时会逐样本重算，任何 cached-source mismatch均进入 semantic conflict。

本地上传包没有 `/data0/...` NPZ source roots，所以本地只能完成 index-level role audit；**fresh source equality 必须在你的训练机上运行下一步指令得到最终结论。**

### 7.4 是否存在 component construction semantic mismatch？——是，而且现在是 dominant evidence

legacy positive target来自：

`PCD = DRS × sigmoid(R_dep) × exp(-GAP)`。

代码对这个函数的定义本来是“post-contact unavoidable-contact operating-point summary”；但 teacher index把它同时用在 Near+Contact作为统一 benefit。更重要的是，它是**补偿式乘积**，而 deployed harm label是**non-compensatory max-veto**。

因此一个 component的大改善可以覆盖另一个 component的实质恶化，从而让同一 candidate同时得到 positive benefit 与 harmful label。

上传的 v48.55 index实测：

| split/regime | candidates | PCD positive | positive & harmful | conflict | GAP culprit | no-GAP仍冲突 |
|---|---:|---:|---:|---:|---:|---:|
| train Near | 1425 | 45 | 20 | **44.4%** | 19 | 7 |
| train Contact | 4086 | 138 | 32 | **23.2%** | 30 | 22 |
| dev Near | 518 | 37 | 12 | **32.4%** | 12 | 3 |
| dev Contact | 1434 | 88 | 25 | **28.4%** | 25 | 10 |

Contact-train overlap样本的中位变化：

- DRS `+0.96`；
- R_dep `−0.300`；
- GAP `+1.300`。

这说明 PCD经常因为 DRS近乎从0跳到1而把 candidate标为 beneficial，即使实际 deployability和oracle gap都变差。**这是 semantic contradiction，不是 gradient scale问题。**

---

## 8. 新发现：不能直接把 benefit 改成 R_dep rescue——teacher m*还有 source-construction plateau

对现有 index做 paper-native screen：

`nominal R_dep < 0 <= candidate R_dep`。

得到：

| split/regime | R_dep zero-cross rescue | harmful | safe | safe groups | safe scenes | candidate R_dep==0.5 |
|---|---:|---:|---:|---:|---:|---:|
| train Near | 23 | 1 | 22 | 9 | 6 | **86.96%** |
| train Contact | 78 | 2 | 76 | 30 | 13 | **91.03%** |
| dev Near | 21 | 0 | 21 | 8 | 5 | **90.48%** |
| dev Contact | 38 | 0 | 38 | 13 | 5 | **68.42%** |

如果只看支持数，这非常诱人；但绝大多数 candidate `R_dep`精确等于 0.5，是异常平台。

代码追溯 teacher margin发现，物理/控制 component取 min之后，历史实现还会：

- 对部分 non-hidden `post_contact_stabilize / yield_rejoin / pull_over` 做 `max(m,0.6)`；
- secondary-threat `avoid_secondary` 做 `max(m,0.9)`；
- route-blocked yield做 post-min ceiling。

论文 teacher margin却写的是 active normalized constraint slack 的纯 min。因此当前 zero-cross evidence可能被这些 post-min heuristic manufacture/quantize。

所以 **V48.56 不能直接启用 R_dep rescue Main**。正确顺序是：先去掉 post-min correction做 shadow source；确认 support是真实的；然后才定义新的 performance target。

---

## 9. “不同 certificate component 的数学类型是否应决定 calibration geometry？”——本轮更具体的答案

### 答案：是，但“数学类型”不是充分条件

V48.50–51提供了正证据：

- discontinuous/hard coordinate不适合承担全部 continuous magnitude；
- hard boundary负责 material sign；
- smooth q geometry负责 hard-equivalence class内 local order。

V48.55则提供否证：

- 把 DRS从 continuous regression拿掉，不能自动修正 deployed geometry；
- 把 continuous DEP/GAP做 pooled RMS normalization，也不能自动修正 deployed geometry。

因此更一般、可写进论文的方法原则应是：

> **Calibration geometry = f(decision role, mathematical type, deployed interface/transform).**

其中优先级：

1. **Decision role**：material admission？non-inferiority？diagnostic/order？
2. **Mathematical type**：discontinuous sign、continuous signed margin、bounded probability、monotone gap statistic？
3. **Deployment interface**：hard veto、threshold、ranking、anti-oracle regularizer、final rerank？

具体到当前三个坐标：

- **DRS**：material/discontinuous → q-hard sign + smooth q local order；
- **R_dep / DEP**：continuous signed deployability；absolute boundary与relative preservation必须分离；
- **GAP**：continuous monotone diagnostic；应先确认是否真的有 hard-veto role，再讨论 normalization。

这比“给每个 component选一个不同 loss/scale”更有 CCF-A 方法论价值，因为它解释了为什么某些看起来数学上合理的 calibration 会失败。

---

## 10. V48.50 → V48.55 递进链与 dominant bottleneck迁移

### V48.50：Exactness不是答案

Exact-only NAP破坏 Near：hard exact coordinate不能承担所有连续 order/magnitude。

### V48.51：BC-FC成为第一个稳定正机制

得到稳定原则：**material sign必须匹配 deployed hard boundary；smooth geometry保留同一 hard class内 local order。**

### V48.52：teacher semantic “更 physical”不自动正确

PSA negative：只修 teacher physical sign而student/deployment q-hard没有形成正确机制。

### V48.53：structural imitation也不是答案

student physical / symmetric CSE失败：部署不需要复制 privileged physical certificate的内部实现。

### V48.54：training-only privileged physical signal也会跨 severity负迁移

IPBD Contact局部增益、Near系统崩塌；physical-margin distillation family STOP。

### V48.55：coordinate-type / scale hypothesis被否证

TCBC没有移动 upstream native geometry；normalization不是dominant bottleneck。

### V48.56：bottleneck正式收敛到 decision semantics

当前最准确的 dominant bottleneck 是：

> **teacher-to-deployment decision-semantic alignment / component-role correctness**。

不是 “physical vs q-hard”、不是 “RMS scale”、也还不是 “final centering”。

次级瓶颈是：

- Near positive scene/group support仍偏少；
- Contact dev semantic eligibility极低；
- root probability reliability可能存在绝对误差，但尚不能解释当前 target contradiction，故继续后置。

---

## 11. V48.56 应保留/删除/新增什么

### 保留为核心 Main backbone

- observation-consistent root/option semantics；
- q-hard deployed material DRS；
- BC-FC hard sign + smooth q local order；
- smooth NAP/native advantage；
- native certificate preservation；
- same source checkpoint / same top-k=5 / same shared gate protocol；
- no regime conditioning；
- Safe/Near/Contact的scene-disjoint train/val/calibration/test evaluation structure。

### 不作为 V48.56 Main保留

- TCBC X（DRS sign-only loss ablation）：可以保留历史代码可复现，但不进入新 core combo；
- TCBC Y（DEP/GAP RMS canonicalization）：STOP；
- TCBC D：不作为论文 Main mechanism。

### 继续禁止

- PSA/CSE/IPBD/selected-root physical-margin distill；
- physical student DRS、teacher-only physical sign；
- q/margin AND/OR、exact-only NAP、BC-NAP；
- root-logit recalibration；
- threshold relaxation/grid、top-k、candidate/macro expansion；
- Near/Contact router、regime-specific threshold/loss/budget；
- aggressive oversampling/hardest-negative distortion；
- generic pairwise/listwise stack、learned admission residual；
- broad encoder fine-tune。

### V48.56 新增

1. **TCSA semantic audit**；
2. **strict_min_slack teacher mode**；
3. **strict shadow calibration/protocol/index pipeline**；
4. **cached-vs-fresh OC-MERO source audit**；
5. **legacy PCD-veto contradiction audit**；
6. **R_dep zero-cross support/plateau audit**；
7. **GAP hard-veto role counterfactual readout**。

---

## 12. V48.56 实验设计：不要再做一个模糊的多机制 Main

### Stage 0 — Existing-source audit

直接在 v48.55 adaptation train/dev上运行 TCSA，同时在训练机读取 canonical source NPZ fresh重算 OC-MERO。

必须回答：cached `R_dep/R_orc`是否与当前 `m_star/root_probs/c_star`一致。

### Stage 1 — Strict-min-slack shadow

新建独立 shadow calibration root：

`/data0/senzeyu2/dataset/OCRAP_v48_56_strict_teacher`

保持：WOMD source、scene split、candidate/future/root/option数、Near/Contact mining、branch-intent component、no full-margin artifact override。

唯一关键变化：

`teacher_margin_semantics.mode=strict_min_slack`

即去掉 post-min floor/ceiling。

然后重新：

`calibration → scene-disjoint protocol → evidence_adapt train/dev → teacher index → semantic audit`。

**不读 test，不做 performance training。**

### Stage 2 — 分析 strict shadow 后才决定 performance factor

若 strict source：

- source equality通过；
- R_dep plateau显著减弱；
- Near/Contact仍有可用 scene/group-level rescue support；
- PCD conflict仍显著；

则下一 performance experiment测试 **Deployability-Boundary Rescue / role-separated evidence**，而不是 centering。

若 strict source让 rescue support几乎消失，则说明此前“恢复机会”相当部分由 teacher heuristic产生。下一步应重构 recovery teacher/option feasibility，而不是调 evidence head。

若 strict source后 PCD conflict显著消失且 native geometry变得合理，才重新评估是否可以把 benefit role修正后进入 centering。

---

## 13. 下一 performance mechanism的预设计（先不在 V48.56 启用）

建议在 strict audit通过后，定义一个单轴、role-separated factor，而不是再次多因子叠加：

### Material admission coordinate

用 absolute deployability boundary：

`b_dep(a) = R_dep(a) - gamma_material`

material rescue event：

`nominal below boundary AND candidate crosses boundary`。

### Local order coordinate

在相同 hard class内用 smooth `Delta R_dep` / smooth q order，而不是硬二值承担所有 magnitude。

### Relative preservation coordinate

保留 `sigmoid(R_dep_nom)-sigmoid(R_dep_cand)` 作为 secondary non-inferiority，不再把它和 absolute deployability混叫一个 DEP。

### GAP role

默认改成 diagnostic / anti-oracle/order term：

- 作为 oracle-artifact penalty；或
- 在 material-deployable candidates之间 tie-break；
- 不直接作为 hard veto，除非新的独立 ablation证明 hard role有必要。

这条路线和 v48.21–25 的 generic safe-benefit/admission-head不同：它不引入新的 learned router/head，而是把论文原生 `R_dep` material boundary 与部署 decision role对齐。

---

## 14. 本轮需要回答的新研究问题

1. **一个 certificate statistic在数学上“连续/可归一化”，是否就意味着它应该进入 hard certificate？** 目前答案是否；decision role优先。
2. **oracle-to-deployable gap是 safety boundary，还是 uncertainty/consistency diagnostic？** V48.56需要给出实证答案。
3. **当前 R_dep rescue support是真实 teacher physics还是 post-min floor artifact？** strict shadow直接回答。
4. **构造 oracle artifact是否必须依赖 branch-intent fixture？** 若是，论文必须透明区分 synthetic artifact construction和physical teacher margin；若否，可以进一步做 paper-pure teacher ablation。
5. **absolute deployability boundary与relative nominal preservation是否应该是两个不同坐标？** 当前证据强烈支持分离，下一 performance 版本需要做因果验证。

---

## 15. 数据集与论文证据链建议

你现在已经有 Safe/Near/Contact 各自 train/val/test/calibration，这一点对论文很重要，但用途应是：

> **severity-stratified evidence，证明同一个 planner primitive across continuum work；而不是给模型 regime identity。**

建议最终表格至少拆：

- Safe：nominal utility preservation/non-inferiority；
- Near：material rescue recall + harmful false admission；
- Contact：deployable recovery/secondary-collision/stabilization；
- 所有 regime共享一个 policy/gate semantics；
- calibration/test scene-disjoint；
- 机制选择只用 train/dev/calibration，不用 test。

V48.56 strict shadow先只重建 calibration/adaptation是正确的，因为当前目的是 teacher semantics诊断；**一旦 teacher semantics冻结，最终论文数据集应把同一 teacher contract同步重建到正式 train/val/test/calibration，不能让不同 split混用 legacy/strict teacher。**

---

## 16. 工程落地

V48.56 代码新增/修改：

- `tools/audit_v48_56_teacher_component_semantics.py`
- `src/ocrap/simulation/teacher/margins.py`
- `src/ocrap/config/defaults.py`
- `configs/default.yaml`
- `configs/v48_56_teacher_component_semantic_audit.yaml`
- `scripts/run_v48_56_teacher_component_semantic_audit.sh`
- `scripts/build_v48_56_strict_teacher_calibration_shadow.sh`
- `scripts/run_v48_56_strict_teacher_shadow.sh`
- `tests/test_v48_56_teacher_component_semantics.py`
- `V48_56_DCP_DRFC_BCDE_TCSA_DESIGN_CONTRACT.json`
- `V48_56_TEACHER_COMPONENT_SEMANTIC_AUDIT.uploaded-v48.55.json`
- `OC-RAP-v48.56-DCP-DRFC-BCDE-TCSA-next-commands-ZH.txt`
- `ALGORITHM_CHANGELOG.md` / mirrored `ALGORITHM_CHANGELOG_V48.md`

兼容策略：默认仍是 `legacy`，所以历史 V48.50–55 replay不会因新版本代码默默改变 teacher label；只有显式 strict shadow才启用 `strict_min_slack`。

新增路径回归：V48.47–V48.56 targeted algorithm/certificate/transport tests **67 passed**；其中 V48.56新增 5 tests passed。`compileall`通过；全部 **133 个 `.sh`** `bash -n`通过。

全历史 `pytest`不是 clean target：该上传代码包保留若干与 V48.56 无关的历史 packaging/test debt（例如测试引用已不在包内的 v48.45.3/45.4/45.5 operator command文件；部分 tools namespace在pytest环境下的旧导入预期）。这些不是本轮变更造成，故没有通过“补假文件/改旧测试”来掩盖；V48.56 changed-path regression为 clean pass。

---

## 17. 最终推荐

**V48.56 = TCSA + strict teacher shadow，不是 centering。**

如果要把当前研究主线保持在高水平会议/期刊应有的机制标准，下一步最有价值的不是再提高一个 recall数字，而是先证明：

> **模型训练时被称为“benefit / deployability / gap”的每个量，与最终 deployed admission真正消费的 decision notion是一致的。**

V48.55已经把“尺度/坐标类型”这个较浅层解释基本排除；现在正好到了把核心机制从经验 calibration推进到 **decision-semantic correctness** 的阶段。只有这层正确，后续 evidence centering才会是一个干净、可解释、可写进论文的最后阶段，而不是补丁。
