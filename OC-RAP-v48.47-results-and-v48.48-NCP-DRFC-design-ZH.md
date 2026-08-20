# OC-RAP v48.47 结果归因与 v48.48 NCP-DRFC 设计报告

## 1. Executive summary

v48.47 提供了三条有效算法证据和一条明确工程故障证据：

- A、B、D/Main 均为 authoritative RC=20、pipeline_valid=true，因此是有效算法负结果；
- C 为 RC=30，Balanced 在 DRFC witness 第一个训练 backward 时因 GPU0 已被其他进程占用约 20.86 GiB 而 OOM；certificate/gate 未执行，不能用于 DRFC-alone 归因；
- B/DWOK 明确学低了 observation loss，但没有移动 DRS/deployability false-veto，因此 DWOK 不吸收；
- D 中 DRFC frontier loss 大幅下降，但最终 DRS/deployability proxy 几乎不移动，形成了“upstream native frontier 学得动、downstream proxy 不跟”的强接口诊断；
- 下一版不增加 head，而删除 proxy-of-proxy：用 Native Certificate Preservation (NCP) 把 paper-native OC-MERO 的 predicted DRS / R_dep 直接、单调地接入 non-compensatory admission，并通过 clean DRFC-alone 2x2 验证接口瓶颈。

## 2. v48.47 pipeline validity

| Arm | Factor | authoritative RC | pipeline valid | certificate/gate | 结论 |
|---|---|---:|---|---|---|
| A | reference | 20 | true | executed/evaluated | 可归因算法负结果 |
| B | DWOK | 20 | true | executed/evaluated | 可归因算法负结果 |
| C | DRFC | 30 | false | not executed | engineering fail，禁止算法归因 |
| D/Main | DWOK→DRFC | 20 | true | executed/evaluated | 可归因算法负结果 |

C 的失败日志：GPU 0 总容量 23.60 GiB，仅余 24.44 MiB；一个 pre-existing process 占约 20.86 GiB，Precision 约 1.68 GiB，而失败进程本身仅约 506 MiB（PyTorch allocated 124 MiB）。D 的同一 DRFC stage 成功完成，因此不是 DRFC loss 数值爆炸。

## 3. v48.46 → v48.47 可比性与最可靠结论

v48.47-A 是 v48.46-B 的 paper-consistent observation-class reference，核心 Precision certificate 指标逐项一致：Near recall 0.1111 / UCB90 0.2274 / dep false-veto 14/16，Contact recall 0 / UCB90 0.2773 / dep false-veto 30/31。这提供了很好的跨轮次 anchor。

### 3.1 DWOK：明确不吸收

v48.47-B 的 decision-weighted observation calibration 把 Balanced witness validation observation loss约从 0.7284 降到 0.6462，但最终：

- Near DRS false-veto仍 11/16；deployability仍14/16；recall仍0.1111；
- Contact DRS仍20/31；deployability仍30/31；recall仍0；development sign仍1/37；
- Precision Near harmful UCB90反而 0.2274→0.2847；Contact虽 0.2773→0.2539，但没有 capture 改善。

结论：observation kernel 对自己的 supervision 可优化，但它不是当前 admission bottleneck。保留 observation-consistency 作为 deployability 定义；停止继续把 observation weighting/transport 当性能抓手。

### 3.2 DRFC：当前版本不能直接吸收，但“直接训练 frontier”方向有诊断价值

D/Main 中 frontier validation loss从约0.8166降到0.4538（约44%），margin anchor也略改善，说明 margin_head 可以被直接的 candidate-relative OC-MERO objective 移动。

相对 B（固定已有 DWOK checkpoint后增加DRFC）：

- Contact development sign 1/37→3/37；certificate recall 0→0.05；DRS false-veto 20/31→19/31；
- 但 Contact harmful UCB90 0.2539→0.3416，deployability仍30/31；
- Near dep false-veto 14/16→16/16，recall0.1111→0；虽然 UCB90下降到0.1717。

所以 DRFC conditional-on-DWOK 有局部 capture/sign 信号，但产生明显、方向不一致的风险/false-veto tradeoff。C 工程失败意味着 DRFC-alone 主效应仍未知，不能把D的改善归因给DRFC本身，也不能计算严格 D-B-C+A interaction。

### 3.3 Proposal generation 仍不是主瓶颈

v48.46 final certificate 中 top-5 proposal oracle 对 Near safe-positive groups 9/9、Contact 20/20 可达；Contact positive-group any-hit约96.9%、oracle-best hit约90.6%。v48.47所有有效arm仍有同样的 proposal-positive group population，没有出现“好动作消失”的证据。

因此不再扩大 top-k、candidate family、macro数量，也不做 aggressive positive proposal sampling。

## 4. 当前主瓶颈：native certificate → learned proxy interface

v48.47-D 给出一个关键的机制证据：

1. DRFC 只训练 margin_head，并直接通过 differentiable OC-MERO 优化 candidate-relative DRS/deployability frontier；其 validation objective 显著下降；
2. factor/OCAF stage 冻结上游 witness，然后把 OC-MERO 信息压成 detached recovery compatibility signature；
3. final non-compensatory DRS/deployability harmful coordinates仍由独立 learned component proxy重新预测；
4. certificate 中原本 14/16、30/31 的 deployability false-veto几乎不动。

因此最合理的当前假设不是“margin_head还不够大”，而是 paper-native certificate 到最终 selector 之间存在 **proxy-of-proxy attenuation / sign-scale re-encoding**。继续加 learned residual会重复 v48.44 ROCT 和更早 residual 系列的失败模式。

## 5. 论文主线应如何升级

保留核心：Observation-Consistent Recoverability 解决 branch-wise oracle recoverability 与 deployed observation-conditioned recovery policy 不一致的问题。

弱化经验 claim：现有 v48.43/v48.46/v48.47 不支持“observation aliasing 是 Near/Contact 当前主要失败来源”。它应被描述为一种必须满足的 deployment semantics / anti-oracle structural constraint，而不是所有性能差距的主因。

升级为：**Decision-Sufficient Observation-Consistent Recoverability with Native Certificate Preservation**。

- Structural sufficiency：OC-MERO 在不可区分 observation class 内强制 compatible shared recovery；
- Decision sufficiency：模型必须在会改变 admission 的 DRS/R_dep frontier上可识别；
- Certificate preservation：这些 paper-native recovery coordinates 到 CRISP/OCAF non-compensatory veto 的映射必须保持方向、零边界和单调性，不允许第二个自由 proxy重新翻转。

这套性质对 Safe/Near/Contact统一，不读取 regime id。

## 6. v48.48 NCP-DRFC

### 6.1 Native Certificate Preservation (NCP)

NCP 不增加可学习参数。对每个 candidate直接从模型自己的 OC-MERO 输出得到：

- hard predicted observation-class DRS：`sum_i p_i * 1[max_l q_i,l >= 0]`；
- deployability quality：`sigmoid(R_dep)`。

相对 nominal形成与 component teacher 同符号的 harmful margin：

`H_DRS = DRS_nom - DRS_candidate - eps_DRS`

`H_DEP = sigmoid(Rdep_nom) - sigmoid(Rdep_candidate) - eps_DEP`

正值表示 candidate 在对应 recovery coordinate 上恶化超过 tolerance。它们直接替换 final component path 中 DRS/deployability 两个 learned proxy coordinate；gap和其它独立 evidence仍保持原机制。

性质：

- monotone：candidate DRS/R_dep增加不会使对应 harmful evidence增加；
- zero-boundary preserving：同 teacher/component veto tolerance；
- non-compensatory：benefit不能抵消一个正的native recovery veto；
- parameter-free：不是另一层head/residual；
- regime-agnostic。

### 6.2 严格 2x2

| Arm | NCP | DRFC | 目的 |
|---|---|---|---|
| A | off | off | v48.47 reference |
| B | on | off | certificate→proxy interface主效应 |
| C | off | on | 补回v48.47-C缺失的 clean DRFC主效应 |
| D/Main | on | on | 检验DRFC是否只有preservation后才能传到最终admission |

强因果读法：

- B>A：支持 proxy-interface bottleneck；
- C witness loss下降但 final无效，而D>B且D>C：强支持旧proxy attenuation；
- C单独有效、B无效：吸收DRFC，不吸收NCP；
- B单独有效、C无效：吸收NCP，停止DRFC；
- B/C/D均无效：执行stop rule，转 teacher margin normalization / recovery constraint scales / continuous recovery-option teacher coverage，不再改selector capacity。

## 7. 与内部 CCF-A readiness 的差距

这些是项目内部目标，不是任何CCF-A venue官方阈值。

### Safe

当前结果只证明 standard calibration 可运行，并没有 scene-disjoint Safe policy certificate，因此不能宣称论文目标“safe不损失nominal utility”已经被验证。最终应补 paired closed-loop：NUP/progress/comfort non-inferiority、intervention低、FRA受控。同一selector、同一rule，不做Safe专用策略。

### Near-contact

当前 reference Precision：recall 0.111、harm UCB90 0.227、DRS false-veto11/16、deployability14/16。风险已经接近可接受区间，主要差距是安全正例被系统性 veto。

v48.48 screening：dep FV<=10/16、DRS FV<=8/16、recall>=0.20、UCB<=0.25。更高的paper-readiness：recall约0.25-0.33、precision LCB>=0.40，并有TTC/clearance的paired/bootstrap CI改善。

### Contact

reference Precision：recall0、UCB0.277、dep false-veto30/31、development sign1/37；D虽recall0.05/sign3/37，但UCB0.342不可接受。因此距离论文“碰撞后指标改善”仍很大。

v48.48 screening：dep FV<=24/31、development sign>=6/37、recall>=0.10且UCB<=0.25。后续paper-readiness继续向 recall0.20-0.30、secondary collision absolute reduction>=2pp、post-contact TTC约+0.2s、stable-stop/rejoin和impact severity改善推进。

## 8. C engineering fix与性能策略

v48.47-C root cause不是cache/NPZ：tensor cache命中且materialization约1.25s。下一版：

- 仍两张卡同时运行两个arm：A@GPU0+B@GPU1，然后C@GPU0+D@GPU1；
- 默认每arm内Balanced/Precision **serial**，因此每张GPU同一时刻只有一个训练variant；
- arm启动前用nvidia-smi做free-memory lease，serial默认>=12GB，parallel debug默认>=20GB；不足则等待，超时RC30并记录占用进程；
- 默认要求GPU0!=GPU1；
- 保留persistent mmap tensor cache / witness fast path / telemetry / expandable_segments；
- 不做OOM后自动换batch/loss/epoch的“隐形实验修改”。

## 9. Stop list

继续禁止：threshold-grid densification、top-k expansion、candidate/macro width扩张、aggressive positive oversampling、hardest-negative population distortion、generic pairwise/listwise stacking、full joint Stage2、learned admission residual、v48.38 one-sided tails、v48.39 unbounded factors、v48.40 frontier-tanh、v48.41 full component factorization、v48.42 partial pooling/rank skip、v48.43 POET transport、v48.45 joint SOWR、v48.46 generic staged witness、v48.47 DWOK、broad encoder fine-tuning、regime-conditioned policy/router/threshold/budget。
