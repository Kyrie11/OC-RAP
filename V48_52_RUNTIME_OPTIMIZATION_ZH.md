# v48.52 Runtime Optimization

v48.51 telemetry 显示：两张 24GB GPU 峰值显存都约 919 MB，平均 GPU utilization 约 20%，约四分之三采样低于 30%。因此耗时并非只有“四臂数量多”这一原因，pipeline 还存在明显的小粒度 inference / Python / NPZ I/O / 重复 forward 开销。

v48.52 采用两项不改变实验数值定义的优化：

1. **减少无意义的实验重复**：v48.51 已经给出 BC-NAP 的负 interaction，因此 v48.52 不再做 2×2 四臂，只做 PSA 单轴；历史 v48.51-B 通过严格 identity seal 后可以作为 reference 复用。
2. **standard calibration prediction cache**：同一 checkpoint/config 下 pooled calibration 已产生的原始 score，在 Near/Contact 子校准中原样复用。cache 仅位于当前 calibration 临时目录，绑定 checkpoint SHA256 与 inference-config SHA256；cache miss 仍调用原模型，cache hit 不重新 forward。

没有新增 AMP、TF32、跨 scene batching、候选重排、dtype 修改或模型结构变化。由于当前环境没有目标 A30 和 `/data0/...` 数据，实际 wall-time 加速比例必须由下一轮 telemetry 测量，不能在这里外推具体百分比。
