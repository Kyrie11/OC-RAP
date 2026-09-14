# OC-RAP 三 Regime 指标、Near-Contact Baseline 审计与安全加速报告

日期：2026-09-14  
审计范围：上传的 `iclr2027_conference.tex` / `.bib`、`dataset_properties_bundle.zip`、`OC-RAP.zip`，以及 `CLEAN_COMMANDS.md` Section 4 的外部 baseline 训练/测试入口。

## 1. 结论摘要

论文目标与代码总体一致：同一个规划器/恢复语义跨 **Safe / Near-Contact / Contact** 工作，不把 regime label 当作 learned router 输入；正的 deployable recoverability 是 recovery reserve，负值是 recovery debt。评测应以真正的 closed-loop physical outcome 为主，机制诊断（FRA/DRS/ODG 等）与主结果分开。

本轮审计的关键结论：

1. **Near-Contact 的各 external baseline 最终 benchmark 指标是统一计算的。** MARC、RACP、Robust Scenario MPC、PSF、DR-CVaR、CPSF，以及 supplementary 的 Flow Planner、Plan-R1、BeTopNet，都走 `src/ocrap/simulation/closed_loop_runner.py` 的同一套 Waymax closed-loop metric engine；baseline 自己的 risk/cost/logit 只决定选哪个 executable candidate，不会替代 benchmark clearance/TTC/collision/off-road 指标。因此没有发现“某一个 baseline 用了不同物理指标口径”的问题。
2. 修复了三个实际指标实现问题/脆弱点：
   - NUP 以前直接使用 `utility[0]` 当 nominal utility；现在用显式 `is_nominal` 找到的 nominal candidate id。canonical builder 中 a0 本来就是 candidate 0，因此标准结果应保持不变，但候选重排后不再出错。
   - Near 的 2 m clearance / 3 s TTC 阈值以前硬编码；现在从 `regime_thresholds.tau_d` / `tau_ttc` 读取（默认仍为 2 m / 3 s），保证 metric 与 regime 定义一致。
   - Near exposure rate 聚合以前可能用整段 rollout step 数间接加权；若个别 state 缺失有效 geometry/TTC，会产生分母偏差。现在分别记录 clearance/TTC 的有效 observation count，并按真实有效样本数重建 exposure rate。
3. 修复了 Near **offline evaluator 的无用计算**：`flow_planner` 和 `plan_r1` 未被列入 pure learned methods，导致它们即使只用 checkpoint logits 选 candidate，也额外构建完整 observation-conditioned risk profiles。现在它们与 Diffusion/BeTop 等 learned planners 一样跳过这段无用 risk forecast，同时修正 summary 中的 source attribution。
4. 原 Safe/Near launcher 虽然已经有 `wait -n -p` 动态补位代码，但存在一个**全局 train/prepare barrier → 全局 test barrier**，不满足“一个 baseline 训练+测试结束后，下一个 baseline 马上顶上”的要求。现在 Safe/Near 改成 **per-baseline full pipeline slot**：checkpoint/train/prepare → optional offline → closed-loop test 完成后才释放 slot；释放后立即在同一 GPU slot 启动下一个 baseline。
5. Safe/Near/Contact 以及 `run_external_baselines_all.sh` 默认统一为 **6 个动态 slot，GPU0 三个、GPU1 三个**。GPU slot 构造为 `0,1,0,1,0,1`，因此任何时刻上限仍严格是每张 GPU 3 个任务；谁先结束，下一项就在谁释放的 GPU 上启动。
6. 不做会改变算法定义的“加速”：不减少 Flow 的 4-step midpoint ODE、不减少 Plan-R1 token/codebook/stage、不降低 BeTop topology K、不减 Robust Scenario MPC beam/mode，也不改 24 candidate / 40-step closed-loop protocol。这样加速不会把 baseline 变成另一个方法。
7. 本环境没有挂载真正的 `/data0/.../WOMD` TFRecords，因此不能诚实给出端到端 wall-clock speedup 百分比。完成的是代码级瓶颈审计、等价优化与回归测试；实际时间占比应在 WOMD 机器上直接读取已有 timing breakdown。

## 2. 论文与数据集的 regime 语义

论文的中心不是三个独立专家，而是一条 signed deployable recoverability 轴：

- **Safe**：应尽量不干预 nominal，同时维持基本物理安全。
- **Near-Contact**：仍未接触，但 recovery reserve 很低；核心是低尾部空间/时间裕量以及是否长时间处在 critical exposure。
- **Contact**：发生实际 overlap 后，正问题变成偿还 recovery debt：尽快分离、避免 re-contact/secondary collision、稳定并保持可行 free space。

canonical constructor 的 budget 为 24 candidate prefixes、8 latent roots、12 recovery options；离线 dataset quality filter 可将每 scene-time group 保留数缩到 Safe/Near <= 8、Contact <= 9，但 online closed-loop 仍可评分完整 24 candidates。

上传的数据性质包的 held-out test 统计为：

| Regime | Samples | Scenes | Scene-time groups | Negative deployable | Oracle artifact | Oracle recoverable | Incompatible alias pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safe | 3,216 | 175 | 402 | 6.90% | 0.00% | 93.10% | 0.00% |
| Near | 4,723 | 250 | 595 | 48.80% | 24.41% | 75.61% | 20.92% |
| Contact | 6,687 | 209 | 747 | 44.40% | 21.80% | 77.40% | 14.05% |

三套 test property report 的 missing/finite/shape failure 都是 0；OC-MERO stored-vs-recomputed max absolute error 为 Safe `8.9e-16`、Near `1.18e-7`、Contact `1.16e-7`。

### 2.1 必须修正文稿/数据 provenance 的不一致

上传 TeX Appendix 写的是：Safe held-out test 来自标准 WOMD validation，而 Near/Contact held-out test 来自 `validation_interactive`。但上传的 **实际 test property reports 三者全部记录为**：

`.../validation/validation_tfexample.tfrecord@150`

并非 `validation_interactive`。这不是指标代码问题，而是论文 provenance claim 与实际构建 artifact 不一致。投稿前应以最终要发布/复现实验的数据源为准统一 TeX、CLEAN commands、dataset manifest 和表格说明。

## 3. 每个 regime 应计算哪些指标

### 3.1 Safe：主指标

Safe 关注“不要出事，也不要不必要地破坏 nominal”。论文主指标应是：

1. **Collision scene rate ↓**  
   scene 内从初始 state `t0` 到所有 closed-loop post-step states，只要 Waymax overlap metric 任一时刻为真，该 scene 记 1。最终对 scene 求均值。使用 scene rate 而不是 step rate，可避免长 rollout 对总体碰撞率权重更大。

2. **Off-road scene rate ↓**  
   与 collision scene rate 相同，在 scene 内任一时刻 Waymax off-road 为真则记 1，再对 scene 求均值。

3. **Bounded nominal-utility preservation (NUP) ↑**  
   对每个 decision：

   `regret = U_nominal - U_selected`

   `NUP = exp(-max(0, regret) / sigma_u)`

   若 selected utility 不低于 nominal，则 NUP=1；只惩罚 utility 损失。closed-loop NUP 是 decision-level mean。当前修复后 `U_nominal` 来自显式 nominal candidate，而不是默认数组第 0 项。

4. **Intervention rate ↓**  
   canonical protocol 中 a0/candidate 0 是 nominal proposal，选择非 0 candidate 即 intervention；对 decisions 求比例。当前 canonical builder 保证 nominal a0 语义，因此实现正确。若未来允许 arbitrary candidate reordering，建议进一步把该字段也改成显式 `is_nominal`，与本次 NUP 修复一致。

Safe 可辅助报告 comfort（acceleration/jerk/yaw-rate）、route progress、macro switch 等，但不应挤进主安全结论。

### 3.2 Near-Contact：主指标

Near 应包含 Safe 四项，同时加入真正反映“安全边界 headroom”的低尾部指标：

1. **Scene minimum clearance p05 ↑**  
   每个 simulator state 用 ego OBB 与所有有效 actor OBB 计算最小 signed clearance；主 clearance 取 `max(0, signed_clearance)`。先对每个 scene 取时间最小值，再在 scenes 之间取 5% quantile。也就是说统计单元是 **scene minima 的分布**，不是把所有 state 混在一起取 p05。

2. **Scene minimum TTC p05 ↑**  
   同样先每个 scene 取最小 TTC，再对 scene minima 取 p05。当前 TTC 是**常速度、冻结 heading/footprint 的 swept-OBB SAT TTC proxy**：当前接触/重叠为 0；在最大 horizon 内不预测到碰撞返回 99 s。论文/表格中应称 footprint-aware constant-velocity TTC proxy，不应称“ground-truth future TTC”。

3. **Critical-TTC exposure duration ↓**  
   使用 interval 左端点状态 `t0 ... t(N-1)`，统计 `TTC <= tau_ttc` 的 interval 数乘 `dt`。默认 `tau_ttc=3 s`，`dt=0.1 s`。使用 N+1 个 state 对应 N 个 interval，避免把 terminal state 错算成额外 0.1 s。

我建议 Near 的 **secondary physical diagnostics** 同时保留，但不要和论文主列混在一起：

- clearance exposure duration/rate (`clearance <= tau_d`, 默认 2 m)
- critical-TTC episode count / longest episode
- near-zero clearance exposure (`<= 0.05 m`)；注意这不等价于 Waymax overlap
- clearance deficit AUC：`sum(max(0, tau_d-clearance))*dt`
- TTC deficit AUC：`sum(max(0, tau_ttc-TTC))*dt`
- non-collision scene-min clearance p05（帮助区分“因真的撞了所以 clearance=0”和未碰撞的近失事件）
- terminal clearance / clearance gain、comfort、route progress

而以下属于 **mechanism diagnostics**，不应被误当成 external-baseline physical endpoint：FRA_exec、FRA_cand、DRS、ODG、teacher audit regret 等。本轮已让 summarizer 显式输出 `primary_endpoints` / `mechanism_diagnostics` / `secondary_metrics` 三组，同时保留旧 flat keys 兼容已有脚本。

### 3.3 Contact：主指标

Contact 的核心是“实际接触之后是否恢复”，因此实现里正确地要求 **observed Waymax overlap** 才进入 post-contact 指标。仅仅属于 `test_contact` bucket 不等于在 baseline rollout 中实际发生过 contact。

建议主指标：

1. **Off-road scene rate ↓**。
2. **Post-contact terminal clearance ↑**：以首次实际 observed overlap 为 anchor，最后一个有效 post-contact state 的 exact OBB non-negative clearance。
3. **Normalized post-contact free-space AUC ↑**：`sum(max(0, clearance_t))*dt / post_contact_duration`，减少 rollout 长度差异影响。
4. **Escape scene rate ↑**：默认要求 clearance >= 0.5 m 连续至少 3 states，且该 window 内无 overlap。
5. **Re-contact scene rate ↓**：首次 observed contact episode 之后又出现新的 overlap episode。
6. **Secondary-overlap scene rate ↓**：首次 contact 后出现与**不同 Waymax stable object slot** 的 overlap；不能用“第二个 overlap episode”替代，因为那可能只是撞回同一对象。对象 identity 不可用时应 NaN，不应当作 0。
7. **New stable-stop quality scene rate ↑**：仅对初始确实在运动的 scene，尾部至少 5 states 速度 <=0.5 m/s、无 overlap、无 offroad，且 yaw-rate <=0.25 rad/s。
8. **Post-contact overlap duration ↓**：anchor 后的 overlap interval 数 × dt。

可辅助报告 penetration depth/AUC、clearance gain、time-to-escape、time-to-stable-stop、longest overlap run 等。

**重要统计口径**：当前 `test_contact` 是 counterfactual contact-surrogate cohort；generic physical metrics 可以对完整 paired cohort 比较，但 `post_contact_*` 只能在实际 observed-contact eligible scenes 上解释。除非所有方法 observed-contact eligibility 都是 100%，否则不能把这些 conditional diagnostics 描述成完全 paired 的统一 post-impact benchmark。

## 4. 几何与时间指标实现审计

### 4.1 Clearance

`min_oriented_box_signed_clearance` 使用 oriented rectangle exact signed distance：

- 正数：两个 OBB 边界的真实 Euclidean gap；
- 0：touch；
- 负数：SAT penetration；
- 主 `min_clearance_m = max(0, signed)`。

实现先用 circumscribed-circle lower bound 做 broad phase，只对仍可能改善当前最优值的 actor 做 exact polygon distance，因此这个 broad phase 是**等价剪枝**，不是近似替换。

### 4.2 TTC

`min_oriented_box_ttc` 对每个 actor 用 swept Separating Axis Theorem：相对线速度固定，heading/footprint 固定。circumscribed-circle TTC 只作为 lower-bound broad phase；最终候选仍算 exact swept OBB SAT，因此 broad-phase 不改变结果。

### 4.3 Duration / rate

Exposure 和 overlap duration 都使用 interval 左端点，N+1 states 对应 N intervals。Near 本轮修复后 clearance/TTC exposure rate 使用各自有效 observation count 做分母；duration 仍为 count×dt。

### 4.4 Low-tail aggregation

`scene_min_clearance_m_p05` 和 `scene_ttc_s_p05` 是 scene-level minima 之后的跨-scene quantile，这与“lower-tail safety across scenarios”的论文含义一致；不能改成所有 time steps 的 pooled p05。

### 4.5 Paired uncertainty

方法共享同一 test scene 时，置信区间/显著性比较应以 **scene 为 resampling unit** 做 paired bootstrap，对每个 scene 先形成 method-control delta，再 bootstrap scenes。不要 bootstrap decisions/time steps，否则会低估相关性带来的不确定性。

## 5. Near-Contact external baselines：逻辑、训练/测试瓶颈与安全加速

Near 默认 main-table 6 个 + supplementary 3 个，共 9 个。以下“训练”对 main 6 个非学习方法实际上只是共同 train/val 数据契约验证和 registration，不会产生 `.pt`。

### 5.1 MARC (`marc_lite`)

**逻辑**：semantic multi-policy family + policy-conditioned mode response；根据 mode-conditioned ego futures 找动态 branch point，构造 shared prefix + contingent tail tree；用 CVaR/risk tolerance 做 family/candidate selection。连续 LP/iLQR 被 common executable lattice 精确枚举替代。

**主要时间**：每次 replan 的 candidate feature 构造 + observation-conditioned multimodal risk profiles；之后是 branch point、mode risk、CVaR 和 tree scoring。没有神经网络训练。

**安全加速**：共享一次非学习 registration；保持 predictor/risk profile 每 candidate 一次，不重复 teacher future；6-slot pipeline 并发。不要减少 mode/候选或改 CVaR alpha 来换速度。

### 5.2 RACP (`racp_lite`)

**逻辑**：shared-plan + belief-weighted contingent tails，2:1 的 normalized shared/contingent timing；多模态 belief、collision-risk cost、non-anticipative shared prefix。CommonRoad/Frenet + CasADi/OSQP branch MPC 被 executable candidate tree 枚举替代。

**主要时间**：common risk profiles + shared/contingent tail scoring；候选/模式组合处理比简单 safety filter 重。

**安全加速**：与 MARC 相同，缓存/共享 observation-side invariants；不缩短 branch horizon、不减少 modes。

### 5.3 Robust Scenario MPC (`robust_scenario_mpc`)

**逻辑**：多模态 scenario 概率、pairwise mode distinguishability，在模式可区分前强制 non-anticipative input tying，可区分后允许 mode-dependent recourse；所有模式 hard safety constraint + expected cost。连续 nonlinear MPC/tube 被 bounded beam search over executable candidate-tree tuples 替代。

**主要时间**：在 common risk profile 之后，**pairwise compatibility + beam search** 是 main 6 中最可能的 policy-selection CPU 热点。当前 `scenario_mpc_pairwise_beam_size=128`。

**安全加速**：保持同一个 beam=128 和相同 tie thresholds；只做 vectorized pair checks、预计算 candidate/mode pair compatibility、避免 Python 重建不变量。不能为了加速直接把 beam 128 改成 32/64，否则 baseline 算法和结果都会变。

### 5.4 Predictive Safety Filter (`predictive_safety_filter`)

**逻辑**：若 nominal proposal 在 finite horizon 内满足 stage safety/input constraint 并进入 terminal backup-safe set，就原样接受；否则从 executable candidates 中找 minimum-input-deviation 的安全修正。不是 CBF，旧 CBF alias 仅为命令兼容。

**主要时间**：自身 selector 基本是 O(candidate×horizon) 的 stage/terminal checks，较轻；通常 common observed-risk profile/candidate feature 更占时。

**安全加速**：保留 nominal-first feasibility semantics；control deviation、stage min、terminal backup margin 可 NumPy vectorize。不要把 terminal backup certificate 删除成“只看当前 clearance”。

### 5.5 DR-CVaR Safety Filter (`dr_cvar_safety_filter`)

**逻辑**：source DRCVaRHalfspace affine loss、per-obstacle/per-horizon DR-CVaR safe halfspaces、Wasserstein radius/alpha-tail CVaR、MPC Q/QT/R tracking objective；先 hard halfspace admission，再从 common executable lattice 中投影 minimum-cost candidate。

**主要时间**：observation-only multimodal context / samples 和 candidate-halfspace evaluation。代码已经做了一个关键安全优化：affine DRCVaRHalfspace 用代数等价 closed form，而不是为大量小问题启动 CVXPY。

**安全加速**：继续保持 closed-form/vectorized halfspace；不要重新引入每 candidate/horizon 的 CVXPY。也不要减少 `dr_cvar_num_samples=20` 或 horizon=10，除非单独作为近似 ablation 报告。

### 5.6 Conformal Predictive Safety Filter (`conformal_predictive_safety_filter`)

**逻辑**：held-out raw-future conformal calibration；每 prediction horizon 用 joint-agent L2 nonconformity，delta/T Bonferroni、(N+1)-th infinity sentinel 和 exact finite-sample quantile；runtime 形成 per-horizon conformal tube，执行 Eq.7 separation constraints，并做 minimum-deviation projection。

**主要时间**：第一次 calibration 需要从 held-out calibration cohort + raw WOMD future 构建统计，通常是显著的一次性 CPU/JAX/IO 开销；runtime 的 tube constraint 本身相对轻。

**安全加速**：严格 fingerprint 校验后复用 `conformal_calibration.json`；只有 config/dataset/source/delta/H/T/unit 改变才重算。不能用 test labels 校准。当前 launcher 已按这些字段验证 artifact。

### 5.7 Flow Planner (`flow_planner`, supplementary learned)

**实现配置**：d_model=256、4 layers、8 heads、20-step future、11-step history、32 agents、70 map polygons、4 midpoint ODE sample steps；100 epochs、batch 48、AMP、fused optimizer、pinned/persistent workers。

**逻辑**：CondOT x-start flow matching、overlapping trajectory tokenization、distance-scaled joint scene/trajectory attention、nearest-neighbor CFG，4-step explicit-midpoint ODE；生成一次 native trajectory，再投影到 24 executable candidates，而不是每 candidate 单独跑 ODE。

**训练瓶颈**：scene encoder + flow decoder/attention + dataloader preprocessing，100 epochs 是主要 wall time。

**测试瓶颈**：每 replan 4 次 midpoint ODE evaluation + scene encoding + 24-candidate projection；Waymax state/candidate feature 也有固定成本。

**本轮安全加速**：offline evaluator 现在把 Flow 正确识别为 pure learned，跳过它完全不使用的 observation-risk profile；scheduler 让它的长训练优先启动并和其它 baseline pipeline 重叠。保留 4 ODE steps；减少 sample steps 会改变 planner，不属于 free optimization。

### 5.8 Plan-R1 (`plan_r1`, supplementary learned)

**实现配置**：128 dim、6 layers、8 heads、1024-token motion codebook、20-step future、11-step history；32-epoch predictor pretrain + 5-epoch planner alignment（总 37），batch 12。

**逻辑**：trajectory-token LM + source-style iterative token reconstruction；stage 2 使用 frozen predictor / KL regularization / VD-GRPO-like group-centering adapter，再映射到 common executable candidate lattice。

**训练瓶颈**：1024-token codebook distance/tokenization + 6-layer transformer；两阶段 32+5 epoch 是主成本。代码中已有 vectorized corresponding-corner codebook distance、frozen predictor `no_grad` 等优化。

**测试瓶颈**：scene encoding + autoregressive/token output + candidate projection。

**本轮安全加速**：offline evaluator 不再无谓构建 observed-risk profiles；长训练优先并发。不要减 token vocab、layers、pretrain/planner stage 来伪装成等价加速。

### 5.9 BeTopNet (`betopnet`, supplementary learned)

**实现配置**：128 dim、4 layers、8 heads、K=32 topology agents、history 21、source map 256×20、25 epochs、batch 20；topology loss weight=50。

**逻辑**：actor behavior-braid topology + nearest-lane map topology，iterative actor/map topology prediction，K=32 local attention；planning selector把 normalized candidate confidence 与论文 short-term repulsive potential (`t_b=3`, `lambda_m=0.5`) 合并。

**训练瓶颈**：256 map polylines 的 preprocessing/attention、32 actor topology 和 topology loss。

**测试瓶颈**：source-scene tensor 构建 + topology attention + short-term contingency score。已有 candidate-independent actor/map sorting/cache，避免为 24 candidates 重复相同 scene work。

**安全加速**：复用 scene/topology invariant；不减少 K=32、map context 或 decoder depth，除非作为单独效率 ablation。

### 5.10 Optional legacy severity minimization

`RUN_LEGACY_NEAR=true` 才运行。它是 pre-impact unavoidable-collision severity planner，不属于 Near main six；使用 observed target + constant-velocity extrapolation和 collision/post-impact severity projection。不要把其内部 severity objective 当作 Near benchmark safety metric；最终仍应由统一 closed-loop metric engine 比较。

## 6. 共性训练/测试时间瓶颈

### 6.1 训练/准备阶段

1. 非学习 baseline 的 train/val registration 扫描：现在一套 regime 只扫一次，而不是每方法一次。
2. Learned source bridge：NPZ/WOMD scene/history/agent/map preprocessing 与数据搬运。
3. Flow：100 epoch flow model。
4. Plan-R1：codebook/tokenization + 32+5 stage。
5. BeTop：大 map/topology preprocessing + attention。
6. CPSF：一次性 calibration raw-WOMD future scan。

### 6.2 Closed-loop 测试阶段

runner 已记录的 timing buckets 可直接定位真实机器瓶颈：

- `state_history`：Waymax current state → spliced raw/history；
- `candidate_features`：每次 replan 构建 24 executable candidates/features；
- `policy_selection`：baseline-specific risk/model/optimizer；
- `teacher_labels` / `audit_labels`：不是 deployable planner latency，且正常 sparse/selected 模式已经避免全量 teacher；
- `waymax_step_metrics`：Waymax step + physical metric collection；
- `other_overhead_s`：IO/Python/serialization 等未单列部分。

出版 latency 的边界是 `state_history + candidate_features + policy_selection`，并且 learned model 边界显式 CUDA synchronize。**6-worker throughput run 的 timing 是资源竞争下的数，不应作为 publication latency。** 应在结果跑完后用 `scripts/profile_external_baselines_latency.sh` 一卡一进程复用 checkpoint 重新测 mean/p50/p95。

## 7. 本轮代码修改

### 7.1 Metric correctness

`src/ocrap/simulation/closed_loop_runner.py`

- NUP 改为显式 nominal id。
- Near threshold 改为 config-driven `tau_d` / `tau_ttc`。
- 增加 `clearance_exposure_observed_count` / `ttc_exposure_observed_count`。
- Aggregate exposure rate 按 metric-specific有效 observation 分母重建。

`tools/summarize_external_closed_loop.py`

- 保留旧 flat keys，新增明确的 `primary_endpoints`、`mechanism_diagnostics`、`secondary_metrics`，避免后处理误把 teacher/mechanism metric 放入 main table。

### 7.2 Near offline acceleration/correctness

`src/ocrap/external_baselines/evaluate.py`

- `flow_planner/flowplanner/plan_r1/planr1`（以及 diffusion aliases）加入 learned sets；Flow/Plan-R1 offline evaluation 不再生成未使用的 observation-risk forecast。
- Learned result attribution 不再误标成 rule/optimizer path。

### 7.3 6-slot full-pipeline dynamic scheduler

`scripts/run_external_baselines_safe.sh` / `near.sh`

旧行为：

`所有 baseline prepare/train 完成` → barrier → `所有 baseline closed-loop`。

新行为：

`baseline_i: train/prepare -> optional offline -> closed-loop` 占据一个 slot；完成后立即释放 slot，下一 baseline 在该 slot 对应 GPU 上启动。

Learned methods 优先入队，让最长训练尽早开始并与较短 non-learning controls 重叠；这只改变调度，不改变任何数据、模型或选择结果。

`scripts/run_external_baselines_contact.sh`

- Contact 六个都是 non-learning controller adapters；共同 registration 仍只做一次（重复六次更慢且无科学意义），随后 6 个 closed-loop baseline 同时跑，完成即补位 legacy extra method（若启用）。

`scripts/run_external_baselines_all.sh`

- 默认也统一为 `USE_DYNAMIC_SCHEDULER=true`、`JOBS_PER_GPU=3`、`MAX_PARALLEL=6`，并向子 regime 传递动态 scheduler 设置。

`CLEAN_COMMANDS.md`

- 原 Section 4 命令完全兼容，不需要新增参数；文档明确 slot 生命周期是完整 baseline pipeline。

### 7.4 Resume/force semantics

- Safe/Near 如果 closed-loop artifact 已完整，且不是 `--retrain`，可直接复用并跳过 learned checkpoint prepare。
- `--retrain` 时不再被旧 complete closed-loop artifact 提前 short-circuit，确保真的 retrain + retest。
- CPSF calibration 仍按 fingerprint/dataset/source/delta/H/T/unit 严格复用或重算。

## 8. 调度行为的精确定义

命令：

```bash
bash scripts/run_external_baselines.sh \
  --regime near \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 \
  --max-scenarios 0 --womd-role validation
```

默认 `GPU_SLOTS=(0 1 0 1 0 1)`。初始最多启动 6 个 baseline pipeline；每张 GPU 最多 3 个。`wait -n -p` 返回最先结束任务的 PID，scheduler 查到它释放的 GPU，并把下一 queued baseline 直接放到同一 GPU。因此不会因为另一张卡上的慢任务而等待一个完整 batch。

Near 当前 learned-first queue 的开头通常为：Flow→GPU0、Plan-R1→GPU1、BeTop→GPU0，然后再填入 main controls，直到 6 slots 满。后续任意一个 baseline 的整个 pipeline 完成，下一项立即补上。

动态 refill 现在默认开启；需要支持 `wait -n -p` 的 Bash，推荐 Bash 5.1+。如果机器非常老，应升级 Bash；显式设置 `USE_DYNAMIC_SCHEDULER=false` 仍可回退 fixed batch，但那不满足本次“完成即补位”的要求。

## 9. 验证

完成以下静态/回归检查：

- `bash -n`：wrapper + Safe/Near/Contact/all launchers 通过。
- `python -m compileall -q src tools` 通过。
- 新增 launcher contract tests：验证 Safe/Near 不再存在阶段级 global barrier、full pipeline dynamic refill 存在、all-runner 继承 6-slot defaults。
- 新增 Near exposure denominator test：验证 metric-specific有效观测分母。
- 全仓测试：见最终交付时的 pytest 结果。

由于 review container 没有 `/data0/.../WOMD` 原始 TFRecords，本轮没有伪造端到端时间数据。建议在真实机器完整跑一次后，从每个 `closed_loop_<method>.json` 的 `timing` 字段比较 `state_history / candidate_features / policy_selection / waymax_step_metrics`，即可决定下一轮是否值得继续优化某个具体 hot path。

## 10. 推荐最终论文报告方式

- Safe main table：Collision / Off-road / NUP / Intervention。
- Near main table：Safe 四项 + scene-min clearance p05 / scene-min TTC p05 / critical-TTC exposure duration。
- Contact main table：Off-road + terminal clearance / normalized free-space AUC / Escape / Re-contact / Secondary overlap / stable-stop quality / overlap duration，并同时显示 observed-contact eligibility。
- Mechanism table/appendix：FRA/DRS/ODG、teacher selector regret、artifact-specific diagnostics。
- 所有共享-test-scene方法的 method-vs-control uncertainty：scene-level paired bootstrap。
- Publication latency：单 GPU、单 process、reuse checkpoints 的独立 profile pass；不要引用 Section-4 六并发 throughput run 的 latency。
