# v48.52 重建说明：DCP-DRFC-BCDE-PSA

本版本由用户重新上传的 v48.51 DCP-DRFC-BCDE 代码重新构建，不依赖上一轮已经失效的 v48.52 临时产物。

## 算法差异

v48.52 以 v48.51-B（BC-FC + smooth NAP）为 reference。唯一新的算法因素是 Physical Sign Alignment (PSA)：

- student/deployed hard sign 仍由 model q 的 hard `q_best>=gamma` DRS 给出；
- teacher q 仍负责选择 observation-consistent legal recovery option；
- teacher root 的物理恢复成功改由所选 option 上 `m_star>=0` 判断；
- smooth q-boundary DRS/PCD 继续只负责 continuous magnitude / local ordering；
- BC-NAP、exact-only NAP、v48.50 old DEFC、MC-NCP 全部关闭；
- Safe / near-contact / contact 不进入 policy/router/threshold/budget/loss，仍共享同一 planning primitive。

这样 v48.52 不再把一个新的 learned module 叠到主算法，而是修正 BC-DE 中 teacher material sign 与真正 physical certificate 的语义一致性。

## 运行设计

默认先验证已有 `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_51_dcp_drfc_bcde_ablation_B` 是否可以作为 A/reference 复用。复用条件包括 protocol seal、source checkpoint SHA、authoritative status、factor contract、gate semantics、Balanced/Precision BC-FC witness stage；任一不一致即 fail-closed 回退 fresh A/B。

若历史 reference 可复用，只训练新的 PSA Main，并让 Balanced/Precision 分别在 GPU0/GPU1 并行。若不可复用，则 A/B 两个 arm 分占两张 GPU，每个 arm 内 Balanced/Precision 串行，保持干净单轴归因。

## 运行优化

v48.51 telemetry 表明 GPU 利用率明显偏低，因此 v48.52 对 standard calibration 加入当前原子 calibration attempt 内部的 prediction cache。cache 绑定 checkpoint SHA 与 inference-config SHA；pooled pass 已经算过的样本，在后续 Near/Contact calibration 中直接复用同一原始 float score，不重复 model forward。没有启用 AMP/TF32、候选重排或跨 scene 合批。

## 结果读取

只读 `Main - reference`：先看 teacher/native physical sign 与 DRS false-veto 是否改善，再看 final opportunity/pred-adv centering，最后才看 certificate recall。若 physical sign 改善而 final centering仍未闭环，下一步才进入 Boundary-Complete Evidence Centering；若 physical sign本身没有改善，则停止 transport/loss-weight搜索，转 DEP/GAP/root-probability/teacher correctness audit。
