# OC-RAP v48.35.2 结果审计与 v48.36 OCAF 优化方案

## 1. 审计范围与结论

本轮交叉审阅了论文、上一轮大模型建议、代码包、`reports.zip`、v48.35.2 主实验及上传的 2×2 消融结果。当前权威主实验不是一次普通工程失败：`pipeline_valid=true`、`certificate_executed=true`、`gate_evaluated=true`、`certificate_exit_code=20` 且 `test_roots_read=false`。因此 **RC=20 是有效的算法拒绝**，不能再归因于缓存、候选缺失或一次偶发脚本错误。

最重要的算法结论是：候选池存在可用机会，但 v48.35.2 的 evidence bridge 不能可靠预测“候选动作在当前连续场景压力下会怎样改变 deployability、recovery gap 与 DRS”。阈值、temperature、gamma 或更大的 barrier penalty只能改变准入数量，不能修复排序方向。

## 2. 面向 CCF-A 的三类 regime 投稿目标

这些数值不是 CCF 的官方硬门槛，而是为了让论文达到强主结果、可复现统计证据和审稿说服力所建议的内部 acceptance targets。

### Safe：nominal utility 与最小干预

Safe 不是“也能跑”，而是要证明 recoverability primitive 不会破坏正常驾驶。建议以 paired same-scene non-inferiority 为主：NUP 的配对置信区间下界优于 -1%（最多预注册 -2%）；route progress 下降不超过 0.5%；碰撞、off-road、舒适性不显著退化；干预率控制在 1--2% 内；FRA 不高于 1%。Safe 结果必须在 Near/Contact 主 gate 通过后再读取，避免为了 nominal utility 调参而污染证书。

### Near-contact：在接触前提升低尾安全余量

核心主张应落在 min-TTC 低分位、最小净空、碰撞/急刹、DRS 与 ODG，而不是只报告平均 utility。建议 min-TTC p05/LCB 至少提升 0.2 s 或最小净空提升 0.1 m，并给出 paired bootstrap CI；证书 fit/verify precision LCB 分别至少 0.50/0.40，safe recall 至少 0.25--0.33，harmful-selected UCB 不高于 0.22/0.25，且 NUP 损失不超过 2--3%。

### Contact：降低二次碰撞并改善撞后可恢复性

核心结果应包含 secondary collision/re-contact、post-contact TTC、overlap duration、impact $\Delta v$、stable stop、yaw、route progress/rejoin，以及 DRS/ODG。建议二次碰撞绝对下降至少 2 个百分点，post-contact TTC 提升至少 0.2 s，并在 overlap/$\Delta v$、稳定停车、路线重入上给出有置信区间的改进；证书 fit/verify precision LCB 至少 0.50/0.40，recall 至少 0.20--0.30，harmful-selected UCB 不高于 0.22/0.25。

## 3. 当前 Near-contact 与 Contact 的投稿成熟度

### Near-contact：promising but not submission-ready

Balanced 的 candidate safe-positive AUC 为 0.868，learned evidence positive AUC 为 0.833，说明候选级机会识别存在明确正向信号；oracle proposal support 有 9 个 safe-positive groups，oracle precision LCB90 为 0.846。实际共享规则只选 4 组，其中 1 个 safe-positive、1 个 harmful，precision LCB90 仅 0.078、recall 仅 0.111，平均 teacher advantage 为 -0.135。Precision 版本完全 abstain。Near 的主要缺陷不是没有候选，而是组内 top-1 相关性接近零/为负、共享规则 coverage 太低、跨场景稳定性不足。

### Contact：early signal only，远未达到投稿程度

Contact proposal support 更强：20 个 oracle safe-positive groups，oracle precision LCB90 为 0.924，证明候选生成不是根本瓶颈。Balanced 仅选 5 组，1 个 safe-positive、1 个 harmful，precision LCB90 为 0.062、recall 0.05；虽然选中动作的平均 teacher advantage 为 +0.124，但 5 个动作全部是 macro 7，存在严重 macro collapse。Precision 的 learned evidence positive AUC 仅 0.350，已出现方向反转并完全 abstain。Contact 当前缺陷是 **scene-conditioned action effect 学错方向**，不是阈值太保守。

## 4. RC=20 的根本原因

1. **Development shared-rule fit 是直接 gate 层。** Balanced 在 adaptation-dev 的 Contact 上选 6 组却 0 个 safe-positive、3 个 harmful，harmful-selected UCB90 为 0.732，平均 advantage -0.281；Near 选 0。Precision 两个 strata 都选 0。两者均有 6 项约束失败。
2. **action-only bridge 对 Contact 严重欠条件化。** v48.35 的 `physical_relative` 只看 candidate-minus-nominal 的 prefix parameters、macro、prefix states 与 controls。相同制动/转向增量在不同净空、接触姿态、相对速度、可重入走廊下会产生完全不同后果，但 bridge 无法表达这种乘性交互。
3. **冻结 source experts 的共识先验仍可能主导方向。** 两个历史 bucket expert 都对每个候选执行，虽然没有显式 regime ID，但错误共识会压制小校准器。v48.36 将先验 scale 从 1.0 降到 0.5，把主导权交给可学习的连续 action-effect residual。
4. **macro collapse 是症状。** Balanced Contact 全部选择 macro 7，不应通过 regime-specific macro policy 修补；应让场景几何与动作作用共同决定 margin。
5. **阈值调优无法修复反向 AUC。** Precision Contact positive AUC 0.350、evidence top-1 correlation -0.193，继续扫 threshold/gamma/temperature 只会在“全 abstain”和“有害准入”之间移动。

## 5. 保留、修改与升级

### 保留

- recoverability 作为一等规划目标，OC-MERO/DRS/ODG/FRA/NUP 的论文主线；
- candidate pool 与 oracle-support 审计；
- 五个 signed component 的连续物理语义；
- non-compensatory smooth frontier，禁止 benefit 补偿不安全 component；
- 一个网络、一套共享规则、scene-disjoint calibration/certificate；Near/Contact 仅作最坏分层审计；
- gate 前不读测试集、gate 后才授权 Safe/stress；
- factor/identity 分阶段训练与 exact runtime-order metric。

### 修改

- 将 action-only `physical_relative` 升级为 **OCAF (Observation-Conditioned Action Frontier)**：候选相对动作 $\phi_i$ 与 nominal-anchor observation pressure $\psi_0$ 通过 $[h_i,h_i\odot s_0]$ 交互；不输入 regime ID。
- 场景输入只包含部署时可观察的 ego/agents/BEV/route/map/dynamic context，排除 utility、hard violation、harm proxy、feasibility、nominal/time 审计标量和未来 target。
- 所有 action-to-context 输出路径无 bias，保证 $\phi_i=0\Rightarrow c_i=0$；scene 不能凭空制造 candidate benefit。
- 使用 raw signed action path 保留动作幅值，同时用 RMS-gated normalized direction 改善数值条件，避免 LayerNorm 抹平强弱制动。
- factor 与 identity 阶段都训练 interaction bridge；final calibration 默认关闭，若开启则只做 admission 校准并保留已有 OCAF 权重。
- 强化 benefit listwise、component tail、component margin regression、frontier pairwise 与 safe-utility listwise，但不引入任何 regime-specific loss 或策略分支。

### 值得继续优化的正向信号

- Near candidate/evidence AUC 0.83--0.87，说明机会识别并未完全失败；
- Contact 有 20 个 held-out oracle safe-positive groups，候选覆盖充分；
- Balanced Contact 少量选中动作的平均 advantage 为正，说明正确方向偶尔已被捕获；
- non-compensatory frontier 与共享 rule 的论文语义清晰，应继续作为 novelty 的核心，而不是回退到 case-specific policy。

## 6. 工程错误审计与修复

- 上传 ZIP 混合多个版本：v48.36 使用独立 completion event、attempt ID、原子状态与权威 resolver。
- 旧消融 100k 参数上限将 267,774 参数的 legacy-context 模型误判为错误，并因 `set -e` 阻断 B/C：v48.36 改用 exact trainable-prefix contract，参数上限默认关闭，三项消融独立执行。
- legacy `traincontact.json` 与 canonical `train_contact.json` 样本数不同：新增 canonical dataset-root contract，运行前拒绝 alias。
- gate 后命令旧版引用：v48.36 只生成 v48.36 Safe/stress wrapper。
- stale gate/failure/completion marker：权威 resolver 同时验证 RC、marker、attempt ID、NEXT_COMMANDS 与 blocked 状态。
- training/inference config 漂移：interaction hidden/dropout、consensus prior、context source 均写入 checkpoint 并由版本专用 model contract 检查。
- scene shortcut、zero-action、action magnitude、non-compensatory upper bound 均新增单元测试。
- 校准脚本曾引用缺失的 v48.36 shared-rule/metric-contract 工具：已补齐并通过版本工具 `--help` 与依赖闭包。
- 初版 2×2 的训练契约错误地要求所有 arm 都具备 OCAF 与 non-compensatory cap：已按每个 arm 的 context/prior 动态校验，避免把科学对照误报为 RC=30。
- resume checker 中残留的 `physical_relative`/v48.36.1 元数据已改为严格的 v48.36 OCAF 契约。
- 原始 TeX 的 `\usepackage[hidelinks]` 是编译阻断，已修为 `\usepackage[hidelinks]{hyperref}`；上传材料缺少 `.bib`，因此引用仍需作者补齐。

完整逐项状态见 `OC-RAP-v48.36-engineering-error-audit.csv`。

## 7. v48.36 OCAF 实验判读顺序

1. 先运行主实验。RC=30 只按工程失败处理；RC=20 是有效算法拒绝；只有 RC=0 才进入 Safe/stress。
2. RC=20 时先看 development shared-rule：若 Contact positive AUC 与 top-1 correlation 明显提升但 coverage 仍不足，可继续优化 loss/regularization；若仍接近随机或反向，不要调 threshold，继续改 action-observation interaction。
3. 主实验后运行 2×2：A action-only+soft，B OCAF+soft，C action-only+frontier，D OCAF+frontier。这样分别验证表示与非补偿前沿贡献，不拆 regime。
4. RC=0 后运行 Safe paired non-inferiority 与 held-out stress；严格使用生成的 `NEXT_COMMANDS.txt`。

## 8. 诚实边界

本地环境没有用户的真实 WOMD/Waymax 数据根目录、两张 A30 和已训练 source checkpoint，因此本交付只完成代码、协议、静态/单元回归与结果审计，**不声称 v48.36 已经得到 RC=0**。论文结果表仍需用新主实验、Safe paired 与 held-out stress 实测填充。
