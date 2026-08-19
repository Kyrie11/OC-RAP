# OC-RAP v48.54 结果归因与 v48.55 TCBC 设计

## 结论摘要

v48.54 **只产生 Main 目录是设计行为，不是消融漏跑或工程失败**。v48.54 是单轴 A/B：A 优先语义复用 v48.53-A，只有 B/Main（IPBD）需要重新训练。上传的 `OC-RAP-v48.54-A-reference-reuse-contract.json` 为 `valid=true` 且 `errors=[]`；Main 的 authoritative status 为 RC20、pipeline valid，certificate/Natural gate真实执行，没有 RC30/Traceback/OOM/必需 artifact 缺失。因此可以可靠做 A/B 算法归因。

v48.54 的核心结论是：**IPBD 作为统一主机制失败，不能进入 Boundary-Complete Evidence Centering。** IPBD 对 Contact 有真实局部正信号，但对 Near 造成系统性崩塌，说明 selected-option physical margin 不是无效信息，却也不是可以通过共享 witness 无条件注入的 privileged signal。按照 v48.54 预注册 stop rule，physical-margin distillation family 进入 STOP，下一步转向 component correctness / normalization，而不是继续调 IPBD，也不是重开 root-logit recalibration。

下一版设计为 **v48.55 DCP-DRFC-BCDE-TCBC — Coordinate-Typed Component Boundary Calibration**。它不改变 q-hard deployment certificate，而用严格 2×2 判断两个更基础的问题：离散 DRS 是否不应承担 continuous magnitude regression；连续 DEP/GAP 是否因为 pooled raw scale 不一致而需要 train-only、regime-free、线性的尺度规范化。

## 1. v48.54 工程完整性

### 为什么只有 Main

v48.54 launcher 的实验设计是：

- A = 已验证的 v48.53-A q-hard BC-FC + smooth-NAP reference；
- B/Main = A + training-only IPBD。

若 reference semantic contract 通过，则不重新生成 A 目录。此次 contract 显示：

- source checkpoint SHA 一致；
- Safe / Near certificate / Contact certificate / Near dev / Contact dev 五个 manifest SHA 一致；
- shared rule、observation-class execution、top-k=5、safe-benefit、fit/verify scene-disjoint、no-regime-conditioning、no-test 全部通过；
- `valid=true`、`errors=[]`。

因此只生成 `ocrap_v48_54_dcp_drfc_bcde_ipbd_main` 是正常预期。

### Main 是否为工程有效结果

Main 中：

- authoritative exit code = 20；
- pipeline valid = true；
- calibration/certificate/Natural gate已执行；
- factor contract与 IPBD 设计匹配；
- `test_roots_read=false`；
- 日志未发现 Traceback、ENGINEERING FAILURE、CUDA OOM、Killed、segfault 或缺失文件导致的工程中断。

所以 RC20 是**算法拒绝**，不是工程失败。

## 2. v48.54 A/B 归因

### Precision / Near-contact

| metric | A | IPBD Main | Δ B-A |
|---|---:|---:|---:|
| certificate recall | 0.333 | 0.000 | -0.333 |
| harmful-selected UCB90 | 0.042 | 0.247 | +0.205 |
| candidate safe-positive AUC | 0.448 | 0.320 | -0.128 |
| proposal safe-positive AUC | 0.491 | 0.384 | -0.108 |
| development recall | 0.375 | 0 | -0.375 |
| development precision | 0.088 | 0 | -0.088 |
| development joint sign | 4/19 | 0/19 | -4/19 |
| dev DRS harmful false-safe | 0.419 | 0.706 | +0.287 |
| dev DRS safe-positive veto | 0 | 0.211 | +0.211 |
| safe-positive opportunity median | 0.369 | 0.125 | -0.244 |
| safe-positive pred-adv median | -0.174 | -0.485 | -0.311 |

这是系统性 collapse，而不是仅 final centering 变差。上游 DRS/component geometry、ranking、joint sign 与最终 evidence同时退化。

### Precision / Contact

| metric | A | IPBD Main | Δ B-A |
|---|---:|---:|---:|
| certificate recall | 0.050 | 0.000 | -0.050 |
| harmful-selected UCB90 | 0.351 | 0.340 | -0.011 |
| candidate safe-positive AUC | 0.632 | 0.736 | +0.103 |
| proposal safe-positive AUC | 0.611 | 0.688 | +0.077 |
| dev DRS harmful false-safe | 0.789 | 0.560 | -0.229 |
| dev DRS safe-positive veto | 0.297 | 0.270 | -0.027 |
| dev exact/native positive | 7/37 | 4/37 | -3/37 |
| safe-positive pred-adv median | -0.386 | -0.236 | +0.150 |

Contact 的提升是真实的：candidate/proposal discrimination、DRS harmful specificity 和 pred-adv centering均有改善。但这些收益没有恢复 material certificate recall，而且与 Near 的全面崩塌同时发生。

### component-level 现象

IPBD 还导致 component probability出现明显过旋转：

- Near certificate GAP safe-positive false-veto从约 0.19 推到 1.00；
- Contact certificate GAP safe-positive false-veto从约 0.10 推到约 0.97；
- 同时 deployability/DRS 的 false-safe / false-veto 方向在 Near/Contact 并不一致。

这说明共享 factor representation并不是只差一个最终 threshold，而是 component coordinates 的监督几何本身发生了跨严重度冲突。

## 3. 是否应该正式进入 Boundary-Complete Evidence Centering？

**不应该。**

v48.54 的 prereg 条件是：先看到 ranking / physical boundary evidence提高，并且 q-hard specificity保持，只有 final `opportunity/pred_adv` 仍负偏，才能把 final evidence centering认定为下一 dominant bottleneck。

当前 Near 没有满足任何这个前提：ranking下降、DRS specificity下降、safe-positive veto增加、joint sign清零，最终 evidence也下降。因此 upstream certificate/component correctness 尚未闭环。此时做 evidence centering会把 upstream geometry error 与 downstream centering混在一起，失去因果可解释性。

## 4. IPBD 与 physical-margin family 的结论

### 应明确否定的命题

1. **“physical hard certificate 直接替换 q-hard certificate 会更正确”——否定。** v48.53 C/D 已否定。
2. **“teacher/student/deployment 内部结构相同就能修复 physical certificate”——否定。** v48.53 D 的 specificity最差。
3. **“只把 teacher 物理化、不改 student/deployment 也会更正确”——否定。** v48.52 PSA 系统性负向。
4. **“privileged physical signal只要不进入 deployment，就不会伤害统一 planner”——否定。** v48.54 IPBD deployment完全不变仍造成 Near collapse。
5. **“Contact ranking变好即可说明 unified algorithm变好”——否定。** v48.54 Contact AUC显著提高但 Near崩塌且 Contact recall归零。

因此 physical-margin distillation family应在主线中正式 STOP。后续可作为机制/negative ablation保留，不应继续 weight/temperature/sampling 搜索。

### 仍然成立的机制

截至 v48.54，最可靠的机制仍然是：

- observation consistency；
- shared, regime-agnostic recovery policy；
- `q-hard` / exact decision coordinate承担 material sign；
- smooth q geometry承担 hard-equivalence class 内 continuous ordering；
- deployed decision boundary必须进入 upstream calibration；
- privileged physical realization不能机械替换或复制 deployed decision invariant。

## 5. v48.50→v48.54 的递进主线

- **v48.50**：exact-only NAP损失 Near ordering/recall，证明 exactness ≠ information completeness。
- **v48.51**：BC-FC 证明 hard sign + smooth order 的职责分离有效。
- **v48.52**：PSA 证明 semantic teacher physicalization不等于可学习/可部署 correctness。
- **v48.53**：CSE 证明 structural imitation不等于 decision equivalence。
- **v48.54**：IPBD 证明 privileged physical signal即使只用于训练也可能产生 shared-representation cross-severity negative transfer。

因此论文不应继续围绕 “physical certificate structure” 加限定词，而应收敛到更一般的原则：

> **Invariant-Preserving Boundary-Complete Decision Equivalence**
>
> 一个统一 planner 应保持部署时真正消费的 decision invariants；辅助/privileged coordinates只有在不破坏这些 invariants并能跨 normal→critical continuum保持 Pareto improvement 时才应进入共享学习。

v48.55 对这一主线增加的是 coordinate-type calibration，而不是新 planner module：

> **discontinuous coordinates contribute calibrated boundary/sign information; continuous coordinates contribute scale-normalized distance/order information.**

## 6. 新 dominant bottleneck

v48.54 后，dominant bottleneck 比“physical sensitivity vs q-hard specificity的无损传递”更具体：

> **heterogeneous component coordinates 的 supervision geometry 与 cross-regime scale consistency。**

当前 factor head把 DRS、deployability、gap-quality 都做 raw SmoothL1 magnitude regression，但：

- DRS 是量化/离散 root-mass boundary；
- DEP/GAP 是连续非线性坐标；
- pooled train index 的 RMS 约为 DRS 0.419、DEP 0.223、GAP 0.356。

统一使用 raw magnitude regression会让相同 loss weight表达不同的物理/统计尺度，而且离散 DRS被迫承担 continuous distance regression，和 v48.51 “hard sign不应承担 continuous magnitude” 的机制证据不完全一致。

## 7. v48.55 TCBC 设计

### Factor X：DRS boundary-only magnitude contract

保持：

- DRS component BCE；
- DRS hard veto；
- q-hard native/deployment DRS；
- 所有现有 thresholds。

只取消 DRS 在 explicit component SmoothL1 magnitude regression中的权重：

- X off regression reliability = `1,1,1,0,0`
- X on regression reliability = `0,1,1,0,0`

这不是删除 DRS supervision，而是把它从“boundary classification + continuous magnitude”收敛为“boundary classification”。

### Factor Y：DEP/GAP pooled RMS linear canonicalization

scale 只从 pooled adaptation-train teacher index计算，Near/Contact不分别估计；dev/certificate/test不参与。

对 DEP/GAP：

`target_k = 0.10 * raw_component_margin_k / pooled_train_RMS_k`

当前 reference train index给出的典型 scale：

- DEP ≈ 0.2230
- GAP ≈ 0.3564

DRS 在 Y 轴 scale 固定为 0.10，因此 `0.10*raw/0.10 = raw`，保持 identity。

这个 transform：

- 线性；
- 不饱和；
- 零点严格不变；
- component内排序严格不变；
- 不改变 hard veto；
- 不使用 regime-conditioned scale。

它与 v48.40 已失败的 `frontier_tanh` 明确不同。

### 2×2

| Arm | DRS boundary-only X | DEP/GAP RMS normalization Y |
|---|---:|---:|
| A | × | × |
| B | ✓ | × |
| C | × | ✓ |
| D/Main | ✓ | ✓ |

A 优先语义复用 v48.53-A，只需新跑 B/C/D。B/C分别独占 GPU0/GPU1 并行；D再用双 GPU 并行 Balanced/Precision。

## 8. 下一轮判据

必须先读 B-A，再 C-A，再 interaction。

- **B-A**：如果 Near DRS harmful false-safe / safe-positive veto改善或至少不退化，同时 recall/ranking保持，则支持 discontinuous DRS不应做 continuous magnitude regression。
- **C-A**：如果 DEP/GAP false-veto 与 false-safe在 Near/Contact共同改善，并且 ranking不塌，则支持 pooled raw-scale mismatch 是 causal bottleneck。
- **D interaction**：不能仅看数值正负。只有 D 的绝对 component/native geometry、UCB、Near/Contact ranking达到/超过 A，才可宣称互补。
- **若 component geometry已跨 regime Pareto改善，但 final opportunity/pred-adv仍负**：下一版才正式进入 Boundary-Complete Evidence Centering。
- **若 TCBC 无法改善 component Pareto**：STOP normalization family，转 teacher component correctness audit（DEP/GAP定义、源标签、normalization），再做 root-uncertainty diagnosis；root-logit retraining仍不重开。

## 9. 工程与泄漏约束

v48.55 增加：

- train-only scale artifact绑定 teacher-index SHA；
- scale artifact明确 `strategy_regime_conditioning=false`、`test_roots_read=false`；
- factor cache identity包含 target mode、canonical scales与 regression reliability；
- `V48_55_TCBC_CONTRACT.json` 在 certificate access前 fail-closed验证每个 candidate；
- A reuse使用 semantic identity，忽略只含 transient creation metadata 的 protocol-seal byte SHA；
- RC20仍阻断 Safe/test/closed-loop，只有 D/Main RC0才执行 gate-generated next commands。


## 10. 最终工程验证

最终代码状态下，v48.47–v48.55 核心算法/证书/transport 回归 62 passed；再加 stage-transfer、engineering/protocol、terminal/idempotency、OCAF 与历史 v48.40/v48.41/v48.44 component path，明确 exit 0 的不重复统计为 **148 passed、1 skipped**。`compileall` 通过；仓库内 **130 个 shell 脚本**与根 two-GPU 指令全部 `bash -n` 通过。合成 post-gate 验证 RC20 严格阻断 `NEXT_COMMANDS`，只有 RC0 + 合法 D/Main TCBC factor + train-only scale/preflight 才授权执行。当前容器没有 `/data0/...` 与 A30，因此不声称真实 v48.55 end-to-end 已运行。
