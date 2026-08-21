# OC-RAP V48.56 结果归因与 V48.57 CMRI 设计报告

## 1. 结论摘要

V48.56 没有授权进入 Boundary-Complete Evidence Centering。相反，它把前一轮“decision-role semantics”假设进一步收紧：**relative recovery improvement 与 absolute deployability admission 不能继续塞在同一个 noncompensatory component max-veto 里。** DZBA 把 `R_dep=0` 的正确物理边界放到了错误的逻辑位置，造成 Near/Contact safe-positive 100% false-veto；GOR 的 ranking semantic intuition 仍有局部支持，但 standalone deployment 无收益。

按预注册分支，component-construction family 到此应停止。本轮最强的新证据来自 v48.56-A 的 source audit：Near/Contact 中 teacher 明确认定有正 recovery gain 的 candidate，在模型的 candidate-vs-nominal predicted `R_dep` 上反而大幅为负，而 Safe 明显更正常。因此当前 dominant bottleneck 已从“component semantic mismatch”进一步收窄为：

> **counterfactual candidate-conditioned latent source drift：部署模型在比较不同 candidate 时，同时改变了 recovery consequence 与用于积分的 predicted root measure，破坏了 candidate-vs-nominal source comparability。**

V48.57 只测试这一条因果假设：**CMRI (Common-Measure Root Invariance)**。它把同一 scene-time 的唯一 nominal predicted root posterior 作为所有 candidate recovery aggregation 的共同 latent-world measure；不改 root head、不加参数、不重新校准 logits、不使用 teacher、regime label 或新阈值。

---

## 2. V48.56 严格按 B−A → C−A → D−B−C+A

### 2.1 B−A：DZBA 被明确否证

Precision certificate 的关键变化：

| Metric | Near A | Near B | Contact A | Contact B |
|---|---:|---:|---:|---:|
| cert recall | 0.3333 | **0** | 0.0500 | **0** |
| fixed-DRAC candidate safe-positive AUC | 0.4979 | 0.3534 | 0.6330 | 0.5903 |
| native DEP harmful false-safe (dev) | 0.3824 | 0.0216 | 0.4120 | **0** |
| native DEP safe-positive false-veto (dev) | 0.5263 | **1.0** | 0.5946 | **1.0** |

这不是一个“安全更保守、稍微牺牲 recall”的正常 trade-off。Contact B 直接进入 abstain-all；Near/Contact 的 safe-positive false-veto 同时达到 1.0，说明 absolute DEP boundary 已经支配整个 max-veto。

因此应区分两个结论：

1. **保留：** `R_dep=0` 是 material deployability boundary。teacher zero-crossing audit 中存在大量安全 rescue，说明这个边界不是伪命题。
2. **拒绝：** 把 `R_dep_candidate<0` 等价地变成 candidate-vs-nominal harm component，是错误角色分配。

所以不应该做 `0.5` 周围 tolerance/temperature sweep。那只是给 architecture error 调松紧。

### 2.2 C−A：GOR 的“语义直觉”与“算法效果”必须分开

在 fixed-DRAC labels 下：

- Near candidate safe-positive AUC：`0.4979 → 0.5169`；
- Contact：`0.6330 → 0.6310`，几乎不变。

这说明“GAP 不应在缺少 standalone material boundary 时被强行当作不可补偿 hard feasibility coordinate”仍然是合理的 semantic lesson。

但部署结果不支持把 C 保留成算法：Near recall 下滑，Contact recall 归零，harmful UCB 变差。因此：

- **不恢复 legacy GAP-hard 的理论地位；**
- **也不继续优化 GOR 本身。**

它只应作为未来重新分离 improvement/admission architecture 时的设计约束，而不是当前可继续加权的 mechanism。

### 2.3 D−B−C+A：不是 synergy，而是 X 对 Y 的结构性遮蔽

D 与 B 的 candidate/proposal/final deployment 指标逐位一致。工程审计排除了“D 实际跑成 B”：checkpoint/training provenance 不同，Y factor 也进入配置。真正原因是 native certificate preservation 会把 DEP/DRS 直接写回 component logits；X 的 absolute DEP boundary 一旦开启，就在 max-veto 上形成 floor，Y 对 GAP 的改变无法再穿透最终 decision。

因此 interaction term 不能解释为“DZBA + GOR 互补”，而应解释为 **masking/cancellation under an over-constrained veto**。

D 没有形成 fixed-DRAC labels 下 Near+Contact 的 source/native/safety-recall Pareto improvement，所以按照 V48.56 预注册规则：

**centering = NOT AUTHORIZED；component-construction family = STOP；进入 predicted-root uncertainty/source decomposition。**

---

## 3. 更进一步的证据链：从 component semantics 到逻辑层分离

V48.50–56 的链条现在可以收紧成：

`Exact-only → BC-FC → physical teacher → structural imitation → privileged physical distillation → coordinate scaling → decision-role retyping → single-veto role separation failure`

这一轮真正新得到的不是“DEP 也不行”，而是一个更抽象的 falsification：

> **一个 component max-veto 不能同时回答“candidate 是否比 nominal 改善”和“candidate 自身是否绝对可部署”。**

legacy conflict 全由 GAP 参与；DRAC 去掉 GAP-hard 后，fixed-DRAC conflict 转移到 DEP。这说明 conflict 的载体可以变，**冲突结构本身来自 improvement/admission 的逻辑混合。**

因此高水平论文主线不应继续问“下一项 component 应该怎么 normalize/threshold”，而应问：

> 在 observation-consistent counterfactual planning 中，哪些量定义共享 latent source，哪些量定义 action-conditioned consequence，哪些量定义 relative improvement，哪些量定义 absolute admission？

V48.57 只切其中第一刀：source vs consequence。

---

## 4. 当前 dominant bottleneck：predicted recovery source

### 4.1 v48.56-A preliminary R_dep source audit

从 Precision `standard_prediction_cache_v48.json` 重算：

| Regime | abs sign acc | false-veto | relative sign acc | teacher-positive n | teacher-positive predicted relative R_dep median | capture@+0.015 |
|---|---:|---:|---:|---:|---:|---:|
| Safe | 0.8656 | 0.1148 | 0.6572 | 446 | **+0.2808** | **0.7578** |
| Near | 0.6679 | 0.3275 | 0.4713 | 74 | **−0.4648** | **0.3108** |
| Contact | 0.4485 | 0.5481 | 0.4493 | 175 | **−0.5607** | **0.1371** |

这比“pred_adv median 为负”更具体：错误集中在 **teacher-positive recovery candidates 的 relative source representation**。模型很少 false-safe，却系统性 false-veto，Contact 最严重。这也解释为什么把绝对 DEP boundary 再硬化会把问题放大而不是修复。

### 4.2 为什么先查 root source，而不是立刻查 centering

当前模型的 root posterior 来自 candidate-conditioned input/prefix。对同一 scene-time 的不同 counterfactual candidate，最终 OC-MERO 比较实际上是：

`F(p_a, C_a, M_a) - F(p_nom, C_nom, M_nom)`。

这同时改变 `p` 与 `(C,M)`。如果 `p` 表示“当前观测下我们处于哪个 latent future/root”的 belief，那么 counterfactual action 应主要改变 future consequence，而不应重写比较双方的起始概率测度。否则 relative recovery gain 混入 action-conditioned source drift。

这正是 V48.57 要最小化检验的假设。

---

## 5. V48.57：CMRI 的算法定义

### 5.1 Common-measure principle

完整 scene-time candidate group `G={a_0=nominal,a_1,...}`：

`p_ref(z) = p_theta(z | x_nom)`

`R_cm(a) = OC-MERO(p_ref, C_a, M_a)`

然后所有 native certificate / native advantage / component evidence 都从 `R_cm(a)` 的同测度比较构造。

这不是把 candidate root probability 平均化：candidate-set mean 会随 proposals 增删改变，反而破坏 counterfactual invariance。nominal anchor 与当前观测/当前执行策略对应，且对 alternative proposal set invariant。

### 5.2 明确不变的东西

- root decoder/head 参数与 raw root logits；
- root supervision/loss；
- q-hard BC-FC；
- smooth NAP/native advantage；
- teacher/component labels（exact v48.56-A semantics）；
- proposal top-k=5；
- calibration protocol/scene split；
- shared regime-agnostic deployment；
- thresholds；
- candidate-specific `C`, `M`, option validity。

所以 B−A 的唯一 scientific factor 是 recovery integration measure。

### 5.3 fail-closed 条件

CMRI 只有在以下条件同时成立时启用：

1. group size > 1；
2. 恰好一个 nominal；
3. 所有 candidate 的 `root_valid` support mask 完全一致；
4. grouped inference/training 能看到完整 scene-time group。

否则逐 candidate 回退原始 root logits。`predict_sample` singleton 永远不会猜一个 nominal。

另一个更深的前提是 **root slot identity alignment**。即使 support mask 一样，slot 0/1/... 在 candidate 与 nominal 之间也必须表示可比较 latent root。这个假设不能仅靠代码声明，所以 source audit 直接测 `root_signature` / `root_future_signature` 的 identity cosine、nearest-slot identity 与 best-vs-identity gap，并把弱 alignment 设为 STOP condition。

---

## 6. V48.57 实验：不要再造人为 2×2

### A/reference

优先复用 v48.56-A。`tools/check_v48_57_reference_reuse.py` 校验 source checkpoint、protocol seal、dataset manifest、no-test、A semantic factors；失败则 launcher 自动 fresh exact A。

### B/Main

只开：

`model.direct_recovery_evidence_common_measure_root_mass=true`

其余保持 A。

### Mechanistic audits

B 的 Precision 与 Balanced checkpoint 各做 root-source decomposition；如果 A checkpoint 可读，再做 A audit。主要看：

- `pred candidate→nominal root JS` 是否明显高于 teacher root drift；
- CMRI 是否降低 relative R_dep error、提升 sign accuracy/AUC/teacher-positive capture；
- root support/slot alignment coverage；
- CMRI 后 native DEP 与实际 deployed evidence 是否一致；
- improvement 是否最终传到 Near+Contact certificate/dev，而不是只停在 diagnostic metric。

### GO / STOP

**GO CMRI** 需要同时满足：

- mechanism audit 支持“excess predicted root drift”；
- common-measure substitution 对 Near 和 Contact 的 relative source geometry 都有实质改善；
- B/Main 对 Near+Contact 形成可解释的 source/native/deployment Pareto，Safe 无明显破坏。

**STOP CMRI**：root slot/support 不可比、teacher root 本身强 action-dependent、source substitution 无收益、或 source 改善不能穿透 deployment。STOP 后进入 `margin / C / source substitution` 分解；仍不允许 root-logit temperature/recalibration、threshold search、component normalization。

**Centering 仍是条件分支，不是 v48.57 默认下一步。** 只有 upstream/source/native geometry 真正改善后 `pred_adv` 仍出现残余系统负偏，才有资格重新讨论。

---

## 7. 下一轮若 CMRI 成功但 final gate 仍受阻：预留而不混入本轮

V48.56 已经强烈提示未来可能需要显式两层 contract：

1. **relative recovery improvement / ranking layer**；
2. **absolute feasibility admission layer**，其中 `R_dep=0` 才作为真正 material boundary。

但这一层不在 v48.57 实现。原因是把 CMRI 与 two-stage admission 同轮加入会破坏单轴因果归因，又退回 trick stack。只有 CMRI/source 问题被独立验证后，才决定是否需要该架构。

---

## 8. 保留、停止、增强

### 保留

- observation-consistent root / post-prefix observation kernel / OC-MERO 主体；
- q-hard material BC-FC + smooth local order；
- native certificate/advantage preservation；
- shared regime-agnostic planner；
- scene-disjoint adaptation-dev / certificate verification；
- Safe 作为 nominal preservation / non-interference stratum，Near/Contact 作为 alias/oracle-artifact/root-uncertainty mechanism strata。

### 明确停止

- TCBC / component normalization / RMS / scale / DRS regression weight；
- DZBA 及其 tolerance sweep；
- GOR 作为 standalone algorithm 及 GAP reliability/threshold search；
- root-logit recalibration/retraining/temperature sweep；
- threshold relaxation / grid densification；
- proposal top-k / macro family expansion；
- exact-only NAP、hard magnitude DEFC、MC-NCP tolerance；
- PSA/CSE/IPBD physical teacher/student/distillation family；
- aggressive oversampling、generic ranking stack、learned residual、broad encoder tuning；
- regime-conditioned router/policy/threshold/budget。

### v48.57 唯一增强

- counterfactual source–consequence factorization through CMRI；
- 不是额外 loss，不是额外 head，不是阈值；
- 用 source audit 给机制建立可证伪的中间因果量。

---

## 9. 工程实现与运行优化

实现位置：

- `src/ocrap/models/ocrap.py`：common-measure projection；raw/recovery root logits 双通道；direct/native path wiring；
- `src/ocrap/models/inference.py`：checkpoint reconstruction、recovery root probability diagnostic、singleton fail-closed；
- `src/ocrap/cli/train.py`：flag/metadata 保存；
- training / adaptation / model-contract / cache scripts：全链 flag contract；
- `tools/audit_v48_57_root_source_decomposition.py`：source causal audit；
- `tools/check_v48_57_reference_reuse.py`：A reuse fail-closed；
- `tools/compare_v48_57_cmri.py`：single-axis attribution；
- `tests/test_v48_57_common_measure_root_invariance.py`：projection、fail-closed、proposal-set invariance、legacy identity tests；
- `scripts/run_v48_57_dcp_drfc_bcde_cmri_{arm,two_gpu}.sh`：正式 launcher。

v48.56 telemetry 显示 GPU0 平均 util 约 86%，GPU1 约 10%，显存并非瓶颈。v48.57 因此不做模型缩减，而是：复用合法 A；B 的 Balanced/Precision 同时分配 GPU0/GPU1；post-run source audit 也双 GPU 并行。这样优化的是调度，不改变模型容量/算法。

正式指令：

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs bash scripts/run_v48_57_dcp_drfc_bcde_cmri_two_gpu.sh
```

运行完成后上传：

- `ocrap_v48_57_dcp_drfc_bcde_cmri_main.zip`
- `OC-RAP-v48.57-CMRI-audits.zip`

下一轮首先读 source decomposition，再读 B−A deployment；不要在看到单一 recall 改善时提前宣布 CMRI 成功。
