# OC-RAP v48.49 RC30 工程失败分析与 v48.49.1 Hotfix

## 结论

本轮不能进行 v48.49 新因素算法归因。A 是有效 RC20 reference；B/C/D 均在 `v48_47_recovery_frontier` 的模型构造阶段以相同 ValueError 失败，属于单一、可复现的 stage-local flag isolation 工程错误。

## 证据

- A：authoritative RC20，pipeline valid，certificate/Natural gate 已执行。
- B/C/D：authoritative RC30，pipeline invalid；Balanced/Precision 均 exit 1。
- B/C/D factor contract 在父级正确记录 NCP=true；B 开 MC-NCP、C 开 NAP、D 两者都开。
- 失败 stage 的历史脚本却强制 NCP=false，同时没有屏蔽新增 MC-NCP/NAP，于是产生 `DCP=true + NCP=false`。模型 dependency guard 因而 fail-closed。
- 三臂未出现独立 OOM/NaN/path/CUDA failure signature。

## 修复

在 `scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh` 整个子进程中统一屏蔽 NCP/MC-NCP/NAP，并在 train 调用处再次显式屏蔽；stage JSON 同步记录三者为 false。父级 arm 不受影响，后续 factor stage 会继续使用其原始 v48.49 配置。

## 算法状态

本 hotfix 不修改 DCP-DRFC 算法。由于 B/C/D 没有完成 adaptation/certificate，当前没有新证据支持修改算法、阈值、top-k、loss 或任何 regime-specific 机制。应先用 hotfix 完整重跑原 2x2，再恢复算法归因。

## 下一步

删除这次不完整的 v48.49 结果目录，替换为 hotfix 代码，然后继续使用原来的 `OC-RAP-v48.49-DCP-DRFC-two-GPU-run-commands-ZH.txt`。不要修改命令。
