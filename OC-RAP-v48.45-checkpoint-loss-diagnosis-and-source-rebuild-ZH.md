# v48.45 checkpoint 遗失诊断与可归因 source 重建方案

## 结论

本次上传的 A/B/C 三个结果不是算法 gate failure，也不是 OOM。三者均在任何 adaptation/certificate/gate 之前，于 `source_checkpoint_contract` 阶段以工程 RC=30 退出。共同缺失的是历史 source run 中 Balanced/Precision 的 `best.pt`。因此这三包不能用于 SOWR 算法归因。

历史 v48.13 source 无法从当前归档精确复现：当前代码仍有测试引用 `scripts/train_ocrap_v48_13_terra.sh`，但该历史脚本已经不在包中。故不应构造一个新的 checkpoint 再称其为“恢复的 v48.13”。

## 正确实验设计

建立一个新的 source identity：`ocrap_v48_45_source_rebuild_s7`。

1. **S0 shared recovery backbone/witness**：从 scratch 训练一次，使用 pooled `train_safe + train_near_contact + train_contact`，development 使用对应三个 val。训练论文核心的 root / margin / observation / OC-MERO shared-option witness；direct proposal loss 关闭。
2. **S1 source policy/evidence heads**：冻结同一个 S0 checkpoint，只训练 Balanced/Precision 各自的 direct proposal/evidence heads；两者可在 GPU0/GPU1 并行。
3. 写 `SOURCE_REBUILD_COMPLETE.json`，记录 S0 与 Balanced/Precision source checkpoint 的 SHA256。之后 A/B/C/D 的 source preflight 强制检查这些 hash，禁止某个 arm 偷偷使用不同 source。
4. 再运行 v48.45 2x2：A=无 SOWR，B=margin/root witness，C=obs kernel，D=both。A/B/C/D 除这两个开关外保持 v48.44-D dual-ROCT、top-k=5、shared rule、risk budget 等不变。

## 可比性边界

因为历史 checkpoint 已丢失，新 source 与旧 v48.44/v48.45 historical source 不同，所以：

- **可以**做新 source round 内的 A/B/C/D 因果比较；
- **不能**把新 A 的绝对数值与旧 v48.44 A/D 当作“只差 SOWR”的严格对照；
- 如果新 source round 最终成为论文主结果，最终 paper baselines/ablations 应统一基于该 source identity 重跑或明确区分 source generation。

## 其他审计发现

`MAX_PARALLEL_ARMS=2` 不是“总共两个训练进程”。v48.45 每个 arm 内部会同时启动 Balanced(GPU0) + Precision(GPU1)，因此 `MAX_PARALLEL_ARMS=2` 最多产生四个训练进程（每卡两个）。首次 source-rebuild round 推荐 `MAX_PARALLEL_ARMS=1`，此时仍充分使用两张 GPU，但总训练进程恰好两个。

当前 v48 lineage 仍保留历史 `direct_delta_adapters` 双适配器 checkpoint geometry，并在旧 source policy 路径中存在 bucket-based expert selection。这不是本次 checkpoint 修复新增的，也不应和 SOWR source-loss recovery 混在同一轮重构。若论文最终要求严格消除所有 regime-conditioned policy internals，应在 SOWR 归因结束后做一次独立、预注册的 shared-adapter refactor，并重跑完整对照。
