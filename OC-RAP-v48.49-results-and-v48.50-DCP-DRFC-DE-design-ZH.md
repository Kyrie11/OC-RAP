# OC-RAP v48.49 authoritative 结果归因与 v48.50 DCP-DRFC-DE 设计报告

## Executive conclusion

本轮最重要的结论不是“DCP-DRFC 需要更多网络容量”，而是 **certificate interface 的 decision equivalence 尚未闭环**。上传的 v48.49 A/B/C/D 四臂全部为 authoritative RC=20、pipeline/certificate/Natural gate 完整执行、test roots 未读取，因此可以做真实的 2×2 算法归因。

v48.49 的结果把两个假设清楚地区分开：**MC-NCP 被否定**；**NAP 有有效机制信号，但只解决了 ordering 的一部分**。C 是唯一稳定正向的 arm：Near Precision recall 从 A 的 0.111 提升到 0.222；Contact candidate safe-positive AUC 从 0.620 提升到 0.681，proposal safe-positive AUC 从 0.529 提升到 0.627。但 Contact safe-positive `opportunity>=0.5` 仍为 0/37，Near 也只有 1/19，因此不能说 positive sign 已被修复。

代码审计发现关键 semantic mismatch：teacher/evaluator PCD 是 `DRS_hard × sigmoid(R_dep) × exp(-relu(gap))`，而 v48.49 NAP 实际用 smooth `shared_feasible_mass` 替代 `DRS_hard`。旧 DRFC 也使用 smooth DRS + teacher root probabilities，而 inference native DRS 使用 hard event + model-predicted root probabilities。**同一个 zero crossing 并不等价于同一个 decision coordinate。**

因此 v48.50 收敛到 **Decision-Equivalent Certificate Transport**：以 v48.49-C 为 reference，关闭本轮失败的 MC-NCP；用 upstream DEFC 修 train/inference certificate coordinate mismatch，用 downstream E-NAP 修 exact teacher/evaluator PCD mismatch。

## 1. 论文 idea / motivation / algorithm 主线

核心问题是“当前动作是否保留未来真实可部署的恢复能力”，不是在 contact 后追加 controller。关键 failure mode 是 **oracle-to-deployable recoverability gap**：oracle 可以按 latent future/root identity 为不同分支选不同 recovery；真实系统执行 prefix 后只能依据 post-prefix observation，因此 observationally indistinguishable futures 必须共享兼容 recovery。

主链：`recovery-sufficient roots → post-prefix observation equivalence → shared legal recovery option → OC-MERO deployable certificate → calibrated CRISP admission`。

建议 CCF-A 叙事统一成 **Observation-Consistent, Decision-Sufficient Recoverability**：Observation consistency 约束 hidden future identity 不得进入 recovery choice；Decision sufficiency/equivalence 约束 physical certificate 到 admission 的 sign/order/boundary 不得被 proxy 再编码。Safe/Near/Contact 是同一 primitive 的 evaluation strata，而不是 regime-conditioned policy。

## 2. 数据集能支持什么论证

12 个 train/val/calibration/test × safe/near_contact/contact 主报告均无 construction failure。critical split 具有 oracle artifact、observation conflict、recovery-option diversity，能支撑 oracle-vs-deployable failure mode；Safe 更接近 nominal/non-critical 分布，用于 nominal utility/non-inferiority。

关键性质见 `dataset_12split_regime_summary.csv`。train Near/Contact 的 deployable margin 比 calibration/test 更负（均值约 -1.79，而 calibration/test 更靠近零边界），支持 **decision-boundary transport/calibration** 而非不同 regime 的网络。Near/Contact 有非零 incompatible observation aliases 与 recovery-option diversity；Safe 的 artifact/alias incompatibility 基本为零。

12 个 report 均 `supports_waymax_runtime_claim=true`、`supports_womd_primary_claim=false`，因此 report 本身不足以证明 TeX 中 WOMD primary provenance，投稿前必须审 dataset builder/source manifests。

## 3. v48.49 authoritative 2×2

| Arm | 机制 | Near recall | Near harmful UCB90 | Contact recall | Contact harmful UCB90 | 结论 |
|---|---|---:|---:|---:|---:|---|
| A | NCP+DRFC reference | 0.111 | 0.043 | 0 | 0.310 | reference |
| B | + MC-NCP | 0 | 0.591 | 0 | 1.000 | **reject** |
| C | + NAP | **0.222** | **0.112** | 0 | 0.567 | **唯一正向机制信号** |
| D | MC-NCP + NAP | 0 | 0.465 | 0 | 0.770 | 被 MC-NCP 拖垮 |

Balanced 方向一致：只有 C 在 Near 得到 1/9 positive（recall 0.111），A/B/D 为 0；Contact 四臂仍为 0。

### 有效设计

NCP/DRFC 主诊断仍成立；NAP 的 native recovery value 确实带来 ordering 信息；proposal/candidate availability 没有新证据表明是 dominant bottleneck，因此 top-k/candidate width 继续冻结。

### 无效/停止设计

MC-NCP reject：B 相对 A 的 Near DRS safe-positive false-veto 0/16→7/16、gap 5/16→14/16；Contact DRS 0/31→4/31、gap 6/31→28/31。D 与 B 同方向。历史 stop list 中 POET、SOWR、DWOK、unbounded factors、learned admission residual、top-k/candidate expansion、aggressive oversampling、generic pair/listwise stacking、broad encoder fine-tuning 同样不再重复。明确禁止 regime-conditioned router/policy/threshold/budget。

## 4. 当前 dominant bottleneck

Near 已有排序能力但 absolute boundary 仍偏；Contact 是最大 empirical gap：C candidate safe-positive AUC 0.681、proposal safe-positive AUC 0.627，说明不是完全没有可分信息，但 recall=0、development safe-positive opportunity≥0.5=0/37，是典型 **ordering-positive / centering-negative**。

最值得验证的根因是 decision-coordinate mismatch：teacher/eval PCD 用 hard DRS而 v49 NAP 用 smooth DRS；旧 DRFC prediction 用 teacher root probabilities 而 inference 用 model-predicted probabilities；旧 DRFC 未同时校准 exact gap-quality 与 exact PCD candidate-vs-nominal margin。

## 5. v48.50 DCP-DRFC-DE

四臂固定 v48.49-C base：NCP=true、DRFC=true、NAP=true、MC-NCP=false；source/proposal/calibration/risk protocol 不变。

**X = DEFC**：不加 head。prediction forward 使用 hard DRS `Σ p_pred 1[q_best>=0]`、`sigmoid(R_dep)`、`exp(-relu(gap))`、exact PCD；teacher 用对应 teacher native coordinates。hard DRS forward 精确，backward 用 straight-through sigmoid surrogate。

**Y = E-NAP**：`V_exact=DRS_hard×sigmoid(R_dep)×exp(-relu(gap))`；`benefit_margin=V_exact(candidate)-V_exact(nominal)-0.015`。parameter-free deterministic overwrite。

| Arm | DEFC | E-NAP | 解释 |
|---|---:|---:|---|
| A | off | off | v48.49-C reference：old DRFC + smooth NAP |
| B | on | off | upstream decision-equivalent calibration only |
| C | off | on | downstream exact PCD transport only |
| D/Main | on | on | full v48.50 |

## 6. 下一轮因果读数与 stop rule

先看 B-A 是否修 native false-veto/坐标校准；再看 C-A 是否把 exact PCD 的 absolute sign 拉回且不丢排序；最后看 D 是否将二者转成 certificate recall + harmful risk control。若 exact C 反而明显丢排序，说明 smooth boundary mass 有 tie/ranking resolution 价值，但不能拥有 admission sign；后续应做 **sign-preserving continuous refinement**，而不是恢复 MC-NCP。

若 B/D 仍无法推动 Contact development positive sign，则停止 selector/OCAF/threshold/top-k 搜索，转入 teacher PCD correctness、margin/constraint normalization、predicted root-probability calibration、recovery-option coverage、teacher PCD recomputation audit。

预注册 screen：Near D Precision recall≥0.25、UCB90≤0.25、deployability FV≤6/16；Contact 第一阶段 dev joint sign≥6/37、certificate recall≥0.10、UCB90≤0.25、deployability FV≤14/31。

## 7. CCF-A readiness

Novelty/story 已有强会潜力：oracle-to-deployable gap 是清楚的 deployment-semantics failure；observation consistency 与 decision equivalence 是两个互补原则；方法不靠 regime routing，也不靠继续堆 head 制造复杂度。

但 empirical closure 未完成：Near 需稳定 sign/recall/precision；Contact 当前 recall=0 是最大差距，后续必须证明 secondary collision、post-contact TTC/stable-stop、yaw、route rejoin、impact severity/Δv 等闭环收益；Safe 当前没有 scene-disjoint paired closed-loop non-inferiority。投稿前还需多 seed/confidence intervals、FRA/calibration curves、核心 baselines、oracle artifact case studies、完整 ablation。

## 8. 工程落地与审计

已完成 DEFC loss、train/witness/config/checkpoint plumbing、exact NAP/inference reconstruction、diagnostic-only native margins、新 v48.50 comparator、regime-conditioning model contract、nested witness stage isolation、v48.50 运行脚本与 fail-closed post-gate wrapper。

工程修复包括：初版 v48.50 OUTPUTDIR 仍指向 v48.49 的版本串线；DEFC prediction mask 不应依赖 teacher NaN/finite missingness；实现过程中一次文本替换误伤旧 v48.47 `teacher_mask`，均已修复。

验证：算法相关 v48.47–v48.50 regression **35 passed**；terminal/stage-isolation **19 passed**；`compileall` PASS；114 个 `scripts/*.sh` 全部 `bash -n` PASS；RC20/RC0 post-gate synthetic contract PASS。更广 legacy batch 的 4 个 v48.31 failures 在原 v48.49 包也存在，因此不声明 full historical suite pass。

## 9. 下一步执行

运行交付的 `OC-RAP-v48.50-DCP-DRFC-DE-two-GPU-run-commands-ZH.txt`。D/Main RC0 才允许执行 controller-generated Safe paired non-inferiority + stress/closed-loop；RC20 明确 BLOCKED。运行结束后上传四个 v48.50 run ZIP 与 2×2 audit，下一轮按 B→C→D 的因果顺序做归因。
