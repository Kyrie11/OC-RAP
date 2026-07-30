# OC-RAP Near-Contact 外部 Baseline 复现审计与双 GPU 优化报告

## 1. 审计范围与结论

本次审计覆盖：

- 论文：`post-collision.tex`
- 代码：`OC-RAP.zip`
- 数据报告：`reports.zip`
- 重点脚本：`scripts/run_near_contact_external_baselines.sh`
- 重点实现：
  - `src/ocrap/external_baselines/observed_risk.py`
  - `src/ocrap/external_baselines/policies.py`
  - `src/ocrap/external_baselines/models.py`
  - `src/ocrap/external_baselines/data.py`
  - `src/ocrap/external_baselines/train.py`
  - `src/ocrap/external_baselines/evaluate.py`
  - `src/ocrap/simulation/closed_loop_runner.py`

### 总结性判断

1. **论文的核心 idea 是清楚且有区分度的。** OC-RAP 不是普通的风险最小化或分支可恢复性判断，而是识别“oracle 每个隐藏分支可分别选择恢复动作，但部署系统无法由后缀观测辨别分支”的虚假可恢复性。OC-MERO 先按后缀观测等价关系约束共享恢复动作，再做 lower-tail 聚合；CRISP 将该量作为候选动作的校准准入约束。

2. **三个物理数据桶是 safe、near-contact、contact；论文内部还使用 normal、low-headroom、occluded、near-contact、post-contact 等可重叠标签。** 目录名是互斥实验桶，regime 标签则是样本属性，两者不应混为一谈。

3. **near-contact 是最能支撑论文中心论点的数据桶。** 测试集 oracle artifact 比例约 24.41%，不兼容观测别名对比例约 20.92%，`R_orc-R_dep` 均值约 0.422；safe 桶相应三个量均为 0，无法检验 observation-consistency；contact 桶也含大量 artifact，但叠加了 post-contact 稳定性目标。

4. **当前命名为 MARC、RACP、GameFormer、DRO-CVaR、Predictive Safety Filter 的代码，不是这些论文或官方仓库的严格复现。** 更准确的表述是：
   - `marc_lite`：MARC-inspired semantic multipolicy scoring adapter；
   - `racp_lite`：RACP-inspired prior-predictive multimodal risk adapter；
   - `gameformer_lite`：GameFormer-like level-k candidate-ranking network；
   - `dro_cvar_filter`：Wasserstein-inspired dispersion-penalized CVaR surrogate；
   - `predictive_safety_filter`：candidate-lattice backup/barrier heuristic。

5. **如果论文表格直接写 MARC、RACP、GameFormer、DR-CVaR、PSF，并声称“复现”，目前不够严谨。** 可以保留为 source-inspired adapters，但需要在正文、表注和附录中明确说明没有使用官方求解器、官方数据接口或官方训练目标。

6. **已提供优化代码。** 性能优化保持数据、场景数、候选数、评估指标和全局 batch size 不变；另外包含 3 个会改变错误旧结果但提高正确性的修复：MARC 约束选择、航向字段索引、GameFormer 负向位移航向计算。

7. **静态检查与测试结果：178 tests passed。** 当前环境没有 `/data0/senzeyu2/dataset/OCRAP`、WOMD TFRecord 和可用训练 GPU，因此未在真实数据上给出训练/闭环 wall-clock 加速比；报告只给出可由调用次数严格推导的优化量和应在服务器上执行的验证方案。

---

## 2. 论文 idea 与算法理解

### 2.1 问题定义

对候选可执行前缀动作 `a`，数据包含：

`(h_t, a, z, o, g, m*)`

- `h_t`：当前场景历史；
- `a`：1 s 左右的候选可执行前缀；
- `z`：恢复充分的隐藏根/未来模式；
- `o`：执行前缀后的可观测结果；
- `g`：恢复动作/恢复 option；
- `m*`：给定根与恢复 option 的 teacher margin。

### 2.2 Oracle recoverability

Oracle 已知隐藏根身份，可对每个根分别选最优恢复 option：

1. 每个 root 内先对 option 取最大值；
2. 再对 root 风险做 lower-tail 聚合。

这会产生 hindsight/branch-identity leakage：两个执行后仍观测等价的根可以分别选互相冲突的恢复动作。

### 2.3 Deployable recoverability

部署系统只能依赖执行后的观测。对观测等价的 roots，必须共享兼容的恢复 option，再进行 lower-tail 聚合。因此理论上：

`R_dep <= R_orc`

当观测等价 roots 的最优恢复 option 不兼容时，二者严格分离，这正是论文定义的 oracle artifact。

### 2.4 OC-MERO 与 CRISP

- **OC-MERO**：输入 roots、root 概率、观测兼容矩阵、恢复 options 和 margin，输出 deployable recoverability、oracle recoverability 和 gap。
- **CRISP**：将 deployable recoverability 作为准入约束，而不是唯一优化目标；名义动作满足约束时保持名义动作，否则在安全、伤害、恢复与效用约束下选替代动作。
- calibration split 只应用于冻结 admission threshold，不能参与模型选择后的 test 调参。

### 2.5 数据时间尺度

论文定义的默认设置为：WOMD + Waymax、10 Hz、1 s 历史、1 s executable prefix、4 s recovery horizon；每个样本当前报告中固定为 11 futures、8 roots（有效 roots 均值约 6–6.5）、12 recovery options。

---

## 3. 三个数据桶的性质

以下为 test split 的核心统计：

| 数据桶 | 样本/组/场景 | 候选/组 | Hard violation | Oracle artifact | Negative deployable | Oracle recoverable | mean `R_dep` | mean gap | 不兼容 alias pair | 主要作用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| safe | 3216 / 402 / 175 | 8.00 | 0.93% | 0.00% | 6.90% | 93.10% | 0.988 | 0.000 | 0.00% | 名义效用、舒适性、无不必要干预 |
| near-contact | 4723 / 595 / 250 | 7.94 | 1.61% | 24.41% | 48.80% | 75.61% | -0.690 | 0.422 | 20.92% | 论文中心：虚假恢复准入与低余量交互 |
| contact | 6687 / 747 / 209 | 8.95 | 2.12% | 21.80% | 44.40% | 77.40% | -0.572 | 0.298 | 14.05% | 接触后稳定、二次碰撞、恢复可部署性 |

### 3.1 Safe

- `R_orc == R_dep`，gap 与 incompatible aliases 都为 0；
- 适合训练/评估正常驾驶模仿、舒适性和 NUP；
- 不适合作为 observation-consistency 核心证据；
- 如果 safe 上 OC-RAP 明显频繁介入，通常说明阈值、效用尺度或 calibration 有问题。

### 3.2 Near-contact

- test、val、calibration 全部为 near-contact + occluded；
- test 中 92.21% 为 low-headroom；
- oracle artifact 约 24.41%，不兼容 alias pairs 约 20.92%；
- `R_dep` 显著低于 `R_orc`，是最有辨识度的主实验桶；
- unknown corridor ratio 约 0.47，说明闭环中观测不确定性是实质因素，而不是仅靠标签构造出来的 gap。

### 3.3 Contact

- 所有样本含 post-contact / counterfactual post-contact 属性；
- 同样有 observation artifact，但任务还要求稳定停车、控制 yaw rate、避免 secondary overlap、重新加入路线；
- near-contact 的 MARC/RACP/风险过滤器不能替代专门的 post-impact controller baseline。

---

## 4. Near-contact 各 split 的分布审计

| split | 样本/组/场景 | 候选/组 | feasible | hard violation | artifact | negative deployable | oracle recoverable | `R_dep` | `R_orc` | gap | low-headroom | incompatible alias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 13324 / 1800 / 600 | 7.40 | 87.86% | 8.94% | 18.94% | 55.31% | 63.63% | -1.794 | -1.469 | 0.326 | 72.34% | 16.15% |
| val | 3445 / 433 / 176 | 7.96 | 93.50% | 0.87% | 24.59% | 50.42% | 74.17% | -0.801 | -0.386 | 0.414 | 89.67% | 20.39% |
| calibration | 6039 / 765 / 316 | 7.89 | 86.19% | 3.48% | 23.99% | 44.83% | 79.17% | -0.509 | -0.102 | 0.406 | 94.19% | 19.57% |
| test | 4723 / 595 / 250 | 7.94 | 89.94% | 1.61% | 24.41% | 48.80% | 75.61% | -0.690 | -0.268 | 0.422 | 92.21% | 20.92% |

### 4.1 重要分布问题

train 明显比 val/test/calibration 更“物理困难”：

- train hard violation 8.94%，test 1.61%；
- train mean `R_dep=-1.794`，test 为 `-0.690`；
- train oracle recoverable 63.63%，test 为 75.61%；
- train 候选数更少，artifact 与 incompatible-alias 比例也更低。

这不一定是错误，但必须解释来源。可能原因包括：

- train 使用更强的 targeted stress sampling；
- 数据生成参数或代码版本不一致；
- train 的 candidate failure 导致平均候选数下降；
- scene/time 选择策略不同；
- train 含少量 post-contact 样本，而 val/test/calibration 没有。

### 4.2 建议的数据协议

1. 检查四个 split 的 dataset fingerprint、builder config、git commit 和 teacher config 是否一致。
2. 按 `scene_id` 保持严格 disjoint；当前单目录报告中的 `leakage_scenes=[]` 是正信号。
3. 把 `candidate_count`、hard violation、root-valid count、future-source composition 纳入版本化 contract。
4. 训练可使用 group-level stratified sampler，使 artifact/non-artifact、low-headroom 强度和 macro family 的比例接近目标 test，而不是直接重采样单候选。
5. calibration 只拟合阈值、风险半径和 baseline 超参数，test 只能运行一次冻结配置。
6. reports 中“calibration/test split empty”的 warning 是因为每个报告只扫描单独的 split 目录；既然实际有 `train_* / val_* / calibration_* / test_*` 四套根目录，这条 warning 不是跨目录协议失败，建议在报告工具中增加 `--protocol-roots` 聚合模式，避免误读。

---

## 5. Baseline 与原论文/开源代码映射

| 当前方法 | 对应来源 | 官方代码 | 当前复现等级 | 可否作为严格复现 |
|---|---|---|---|---|
| `marc_lite` | Li et al., **MARC**, arXiv:2308.12021 | 本次检索未确认作者官方仓库 | 思想适配器 | 否 |
| `racp_lite` | Mustafa et al., **RACP**, arXiv:2402.17387 | `KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions` | 思想适配器 | 否 |
| `expected_risk_filter` | 期望风险类通用 baseline | 不对应单一模型 | 合理的通用消融 | 可以，但不要绑定具体论文实现 |
| `cvar_risk_filter` | CVaR 风险规划类通用 baseline | 不对应单一模型 | 合理的通用消融 | 可以，但要说明风险模型与调参 |
| `dro_cvar_filter` | Wasserstein DR-CVaR / DR risk-constrained MPC 文献 | 当前未接入求解器 | 仅 dispersion surrogate | 否 |
| `predictive_safety_filter` | Wabersich/Zeilinger 系列 PSF；Tearle et al. racing PSF, arXiv:2102.11907 | 当前未接入原 MPC/安全集实现 | 候选栅格启发式 | 否 |
| `oracle_recovery_filter` | 论文自身 Eq. oracle recovery | 内部 teacher upper bound | 定义级诊断 baseline | 是诊断上界，不是 deployable 外部模型 |
| `gameformer_lite` | Huang et al., **GameFormer**, ICCV 2023 | `MCZhi/GameFormer` / `MCZhi/GameFormer-Planner` | architecture-inspired adapter | 否 |

### 推荐的论文命名

在论文表格中改成：

- `MARC-inspired adapter`
- `RACP-inspired adapter`
- `Expected risk`
- `CVaR risk`
- `DR-CVaR surrogate`
- `Predictive safety-filter surrogate`
- `Branchwise oracle upper bound`
- `GameFormer-like level-k planner`

只有真正接入官方仓库、保持核心优化问题与监督目标后，才建议去掉 `-inspired / -like / surrogate`。

---

## 6. 各 baseline 的严谨性分析

### 6.1 `marc_lite`

MARC 原论文核心包括：

1. 对每个 semantic ego policy 构建 policy-conditioned critical scenarios；
2. 依据 scene divergence 形成动态 branchpoint 的场景树；
3. 通过 bi-level risk-aware contingency optimization 联合求解短期公共动作和长期分支动作。

当前代码做的是：

- 按已有 `macro_name` 分组；
- 每个 macro 选一个候选代表；
- 使用 utility、公共前缀 proxy、backup margin、expected/CVaR risk、collision probability、smoothness 的手工线性组合评分；
- 没有按候选 ego policy 重预测场景；
- 没有真实 scenario-tree trajectory optimization；
- 没有 bi-level solver；
- 动态 branchpoint 是候选相对 nominal 的几何相似度 proxy。

**结论：**保留了“多语义策略 + 风险 + common prefix”的外形，不是 MARC 算法复现。

原实现还有一个约束错误：先在每个 macro 中选 unconstrained representative，再在 representatives 中打分，可能在存在 admitted candidate 时执行 rejected candidate。优化版已改为只在 admitted pool 内选 representative；没有 admitted candidate 时才走 feasible fallback。

### 6.2 `racp_lite`

RACP 原论文/官方仓库的要点是：

- 对其他道路参与者的潜在 policies 维护 Bayesian belief；
- 根据 multimodal intents 生成长期 contingent plans；
- belief 进入短期公共计划的优化代价；
- 使用 probabilistic risk metric 控制效率/鲁棒性；
- 官方代码包含 `MPC_branch.py`、`PredictiveControllers.py`、CommonRoad/Frenet planner 适配。

当前代码：

- 使用固定的 7 模态先验；
- 所有场景、actor、候选共享固定权重 `[0.34,0.14,0.14,0.10,0.10,0.10,0.08]`；
- 没有在线 Bayesian update；
- 没有 actor-specific belief；
- 没有 long-horizon branch MPC；
- entropy 主要由固定权重决定，对候选缺乏辨识度。

**结论：**是 prior-predictive risk scoring，不是 RACP 复现。要严格比较，应优先包装官方 RACP 代码而不是继续增加手工权重。

### 6.3 `expected_risk_filter` / `cvar_risk_filter`

这两个 baseline 的算法定义本身成立：对同一观测预测分布分别使用期望和 upper-tail CVaR 聚合风险，再做阈值过滤和效用选择。

需要补足：

- 7 个 mode 是启发式而非学习预测；
- 一个 mode 同时作用于所有 observed actors，不表达每个 actor 的独立/相关意图组合；
- 圆形包络 clearance 与 sigmoid collision proxy 未校准；
- 阈值和 utility/risk 权重是手工值；
- 应在 calibration split 上拟合后冻结。

**结论：**可以作为“基于同一观测风险模型的 expectation/CVaR 消融”，但不能宣称代表完整的 SOTA risk-aware planner。

### 6.4 `dro_cvar_filter`

当前公式是：

`risk = empirical_CVaR + epsilon * weighted_std / alpha`

真正的 Wasserstein DR-CVaR 通常要求：

- 明确定义经验分布周围的 Wasserstein ambiguity set；
- 优化 ambiguity set 内最坏分布下的 CVaR/风险约束；
- 使用 dual reformulation、LP/SOCP/SDP 或可证明的近似；
- ambiguity radius 通过独立数据和置信水平设定。

当前实现没有最坏分布优化，也没有 Kantorovich dual/robust MPC 约束，因此 `DRO-CVaR` 命名过强。

**结论：**应改名 `CVaR + dispersion penalty` 或实现真正 DR-CVaR 求解器。

### 6.5 `predictive_safety_filter`

PSF 的关键通常是：

- 验证 proposed action 是否存在满足动力学和约束的安全延拓；
- 不安全时用 MPC/QP 给出最小侵入替代输入；
- 依赖 terminal safe/invariant set 或 backup trajectory；
- 讨论 recursive feasibility / constraint satisfaction。

当前实现仅做：

- 最大加速度/转向 gate；
- stopping/backup margin；
- 相对 nominal barrier inequality；
- 在预生成候选 lattice 中按分数选候选。

没有控制投影、动力学约束求解、terminal set，也没有形式保证。

**结论：**是 safety-filter-inspired candidate selector，不是 predictive safety filter 的严谨复现。

### 6.6 `oracle_recovery_filter`

这是论文需要的非部署上界：使用 `m_star`、hard、harm 与 branchwise oracle margin 做选择。它不是外部模型，也不能与 deployable method 在同一信息条件下解释。

原代码将所选候选 root 0 的 best option 填为单个 `selected_option`。但 branchwise oracle 的定义允许每个 root 选择不同 option，因此单一 DRS option 没有严格语义。建议：

- oracle 的主指标使用 branchwise oracle success / `R_orc`；
- deployable DRS 标记为 N/A；或者
- 另报“用共享 option 重新评估 oracle 所选候选”的 diagnostic DRS，但不要把它称作 oracle 执行动作。

### 6.7 `gameformer_lite`

原 GameFormer：

- Transformer encoder 建模 scene elements；
- hierarchical decoder；
- level `k` 使用 level `k-1` 的交互轨迹结果；
- 联合预测 ego 与多个 agent 的 multimodal futures；
- 每层都有响应关系监督；
- 官方 planner 还包括 feature processing、path planning、model query、trajectory refinement。

当前实现：

- LSTM 编码 ego/neighbor history；
- candidate-set Transformer；
- modal queries 和 iterative level-k ego trajectory；
- neighbor context 主要是静态观测 token，没有真正预测并反馈各 neighbor 的未来响应；
- trajectory loss回归的是候选自身 prefix，而非 ego+neighbors joint future GT；
- policy target 是 logged nominal candidate；
- 没有官方 lane/crosswalk element encoder 与原训练/规划 pipeline；
- 没有调用官方 checkpoint 或仓库。

**结论：**网络结构受到 GameFormer 启发，但任务、输入、输出和监督均有显著变化，不能视为严格复现。

优化版修复了：

`atan2(dy, clamp(dx, min=1e-3))`

对于 `dx<0` 会错误破坏第二/第三象限航向。现在使用标准 `atan2(dy, dx)`。

---

## 7. 发现的代码正确性问题

### 7.1 Smoothness 使用了错误字段

OC-RAP `prefix_states` schema：

`[x, y, vx, vy, heading, yaw_rate, speed, length, width]`

原 `_control_smoothness_cost` 使用第 2 列作为 yaw，实际是 `vx`。这会把纵向速度变化当作转向平滑度惩罚。优化版改为第 4 列 `heading`，并添加单元测试。

### 7.2 MARC 可能选择 rejected candidate

已如 6.1 所述，优化版保证：只要存在 admitted candidate，最终候选一定来自 admitted set。

### 7.3 GameFormer heading 计算错误

负 `dx` 被 clamp 到正小值，导致航向错误。优化版已修复。

### 7.4 Oracle 的 DRS/selected option 语义

尚未直接删除该字段，以避免破坏当前输出 schema；但论文与后处理应按 6.6 重新定义。

### 7.5 Utility 与 risk 未标准化

当前 linear score 混合：utility、风险、backup margin、collision probability、smoothness、deviation。数据报告中 utility 尺度宽，而风险通常在较小范围。手工权重会高度依赖数据版本。

建议：

- calibration 上按 group 做 robust normalization；
- 所有 scaler 只从 train 拟合；
- baseline 超参数只从 calibration 选择；
- test 不再修改；
- 公开完整 search space、selection metric 和 random seed。

---

## 8. 已实现的结果保持型加速

### 8.1 数据分组不再完整展开 NPZ

原 `group_sample_paths`：

- 每个 NPZ 完整 `load_npz` 一次读取 scene/time；
- 排序时再次完整 `load_npz` 读取 candidate index。

优化版：

- 优先从 manifest 读取 `scene_id/time_index/candidate_index`；
- fallback 只通过 `np.load` 读取三个小字段；
- 使用 LRU cache；
- 不再为分组扫描展开大型 history/map/root arrays。

位置：`src/ocrap/external_baselines/data.py:179-214`。

### 8.2 Architecture-aware Dataset

原 dataset 对每个 GameFormer 样本还会构造：

- actor topology；
- map topology；
- teacher branch tensors（按配置）；
- 其他模型不需要的数组。

优化版只构造当前 arch 使用的张量：

- GameFormer：history + prefix；
- BeTop：topology；
- teacher branch context：必须显式开启。

位置：`data.py:515+`。

### 8.3 观测风险 profile 按 group 共享

原 near-contact 非学习离线评估中，7 个 baseline 每个都对全部候选重复构造 `observed_risk_profile`，记录所选候选时又重复一次。

对于 `N` 个候选：

- 原调用数：`7N + 7`；
- 优化后：`N`。

因此 profile 构造调用数降低：

- 平均约 8 个候选：63 -> 8，即 **7.875 倍减少**；
- 24 个候选：175 -> 24，即 **7.29 倍减少**。

这是调用次数，不等价于整体 wall-clock 加速倍数。

位置：`evaluate.py:306-316`、`policies.py` 的 `precomputed_profiles`。

### 8.4 `torch.inference_mode()`

GameFormer 离线推理使用 `torch.inference_mode()`，避免 autograd metadata。模型预测数值不变。

### 8.5 两卡 DDP 保持全局 batch size

原 config 的 `batch_size=64` 在两卡 DDP 下是每卡 64，实际 global batch 变为 128，优化轨迹改变。

优化版明确使用：

- `global_batch_size=64`
- 两卡时每 rank batch = 32
- total DataLoader workers = 8，两 rank 各 4

near-contact train 有 1800 groups，可被 2 整除，因此每个训练 group 恰好使用一次，不需 DDP padding duplicate。

注意：即使 global batch 与样本顺序语义一致，GPU 并行 reduction 顺序不同，不能承诺与单卡 bitwise identical；应要求统计/指标等价，而不是逐 bit 相同。

### 8.6 验证集不再重复 padding

val 有 433 groups，普通 `DistributedSampler(drop_last=False)` 会补成 434，重复一个 group，可能改变 best epoch。

优化版 `_DistributedEvalSampler`：

- rank 0 取 `0,2,4,...`；
- rank 1 取 `1,3,5,...`；
- 433 个 group 每个只评一次；
- 最后 reduce 总和与样本数。

`DDP(broadcast_buffers=False)` 避免不等长验证 shard 的 per-forward buffer collective；当前网络没有 BatchNorm。

### 8.7 CPU/GPU 并发离线评估

- MARC/RACP/风险 filters 是 NumPy CPU workload，不占 CUDA；
- GameFormer 离线评估在 GPU 0；
- 两者并发运行，避免 CPU baseline 占一个无意义的 GPU context。

### 8.8 动态双 GPU 闭环队列

原脚本按固定两两 batch 等待：一个短 job 完成后，GPU 可能等待同 batch 的长 job。

优化脚本使用 Bash 5 `wait -n -p`：

- 首先启动预计较慢的 oracle 与 GameFormer；
- 任意 GPU 完成后，立即领取下一个 method；
- 按实际 wall-clock 动态均衡；
- Bash < 5 自动退化为静态两卡 batch。

### 8.9 控制 CPU 线程过量订阅

每个闭环进程限制 OMP/MKL/OpenBLAS/NumExpr threads，防止两个 GPU job 各自占满全部 CPU cores。

---

## 9. 哪些加速没有默认开启

为避免改变训练或测试结论，以下未默认开启：

- AMP / bf16 / fp16；
- TF32；
- `torch.compile`；
- 减少 epoch；
- 减少 test scenes、max steps、candidate count、recovery options；
- 增大 replan interval；
- 降低 audit frequency；
- 近似 teacher label；
- 改变风险 mode 数或 root 数；
- 量化、剪枝；
- 以 val/test 混合调参。

这些可以做单独的 speed ablation，但不能与严格主结果混在一起。

---

## 10. 双 GPU 优化脚本用法

```bash
cd /path/to/OC-RAP-near-contact-optimized

export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export CUDA_DEVICES=0,1
export RUN=runs/near_contact_external_baselines_optimized

bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

常用覆盖：

```bash
# 强制重训 GameFormer
FORCE_RETRAIN_GAMEFORMER=true \
GAMEFORMER_TRAIN_GPUS=2 \
GAMEFORMER_GLOBAL_BATCH_SIZE=64 \
GAMEFORMER_NUM_WORKERS_TOTAL=8 \
CUDA_DEVICES=0,1 \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

只做闭环：

```bash
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
TRAIN_GAMEFORMER_IF_MISSING=false \
GAMEFORMER_CHECKPOINT=/path/to/best.pt \
CUDA_DEVICES=0,1 \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

### Oracle 协议

- deployable methods：`label_mode=selected`，先用 observable features 选动作，再只对执行动作做 teacher audit；
- oracle：`label_mode=all`，因为其定义需要在选择前看到所有 candidate teacher labels；
- 不要把 oracle 的运行时间与 deployable methods 当作同等在线计算成本。

---

## 11. 真实服务器上的等价性与加速验证

建议先运行一次旧版与新版：

1. 固定 dataset、checkpoint、seed、CUDA/cuDNN/PyTorch 版本；
2. 旧版单卡和新版双卡训练均保持 global batch=64；
3. 比较每 epoch：
   - train/val loss；
   - target index 分布；
   - checkpoint epoch；
   - test policy selection；
4. 规则 baseline 在 correctness bug fix 关闭/未涉及的 methods 上，应做到逐 group selected index 完全相同；
5. MARC 与 GameFormer heading 修复后允许结果变化，但必须记录为 correctness correction；
6. 闭环比较：场景 ID、起始 time index、候选数、steps、selected macro trace、FRA/ODG/DRS/NUP；
7. 从输出 `timing` 比较：candidate generation、teacher labeling、policy selection、Waymax stepping；
8. 至少重复 3 次报告 median wall-clock 和 peak GPU memory。

推荐分别发表两类结果：

- **Legacy-compatible speed comparison**：不启用 correctness fix，验证纯调度/IO 加速；
- **Corrected baseline comparison**：启用本包修复，作为最终论文结果。

当前提供的完整包是 corrected + optimized 版本。

---

## 12. 要达到“严格外部复现”还需完成的工作

### MARC

- 每个 ego semantic policy 条件化预测 future scenarios；
- 构建真实 scenario tree；
- 动态 branchpoint 由 scenario divergence 决定；
- 实现 bi-level risk-aware trajectory optimization；
- 按原论文 horizon、risk tolerance、约束和求解器报告。

### RACP

- 优先直接包装官方仓库；
- 将 CommonRoad/Frenet state 与 OC-RAP/Waymax state 做明确转换；
- 保留 Bayesian policy belief update；
- 保留 branch MPC 与 probabilistic risk objective；
- 只在适配层改变 map/vehicle interface，不改算法核心。

### GameFormer

- 使用官方 encoder/decoder 或官方 checkpoint；
- 提供 map/lane/crosswalk scene elements；
- 对 ego 与 neighbor futures 做 joint multimodal supervision；
- 保留各 reasoning level 的响应学习；
- 明确候选 proposal/refinement 在 OC-RAP 中如何替代；
- 不把“仅候选分类 + 自轨迹重构”称作 GameFormer full reproduction。

### PSF

- 动力学模型；
- state/input constraints；
- terminal safe/invariant set；
- backup trajectory；
- minimally invasive control projection；
- infeasibility/fallback 与 recursive feasibility 说明。

### DR-CVaR

- 明确 Wasserstein metric、ambiguity set、support assumptions；
- 实现 worst-case CVaR dual reformulation；
- ambiguity radius 只在 calibration 上确定；
- 报告 solver tolerance 与 infeasibility handling。

---

## 13. 修改文件

- `configs/external_baselines/near_contact_gameformer_lite.yaml`
- `scripts/run_near_contact_external_baselines_2gpu_optimized.sh`
- `src/ocrap/external_baselines/data.py`
- `src/ocrap/external_baselines/evaluate.py`
- `src/ocrap/external_baselines/models.py`
- `src/ocrap/external_baselines/policies.py`
- `src/ocrap/external_baselines/train.py`
- `tests/models/test_external_observation_only_policies.py`

测试：

```text
178 passed, 5 warnings
```

warning 均为原项目中的 PyTorch Transformer/numerical test warning，不是本次改动导致的测试失败。

---

## 14. 参考来源

- Tong Li et al., “MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving,” arXiv:2308.12021, 2023.
- Khaled A. Mustafa et al., “RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions,” arXiv:2402.17387 / IEEE T-IV, 2024.
- Official RACP repository: `KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions`.
- Zhiyu Huang et al., “GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving,” ICCV 2023.
- Official repositories: `MCZhi/GameFormer`, `MCZhi/GameFormer-Planner`.
- Ben Tearle et al., “A Predictive Safety Filter for Learning-Based Racing Control,” arXiv:2102.11907, 2021.
- Alireza Zolanvari and Ashish Cherukuri, “Wasserstein Distributionally Robust Risk-Constrained Iterative MPC for Motion Planning,” arXiv:2310.04141, 2023.
