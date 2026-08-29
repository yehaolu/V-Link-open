# RoboTwin2 C2 开源裁剪设计

## 目标

将仓库收敛为可公开复现的 RoboTwin2 C2 主方案，保留“数据校验 → Query Warmup → C2 Full → 在线推理 → RoboTwin2 评测”的完整闭环，删除不再参与主方案的消融和研究分支。

## 唯一支持的方案

- 三相机输入。
- Action Query 与 Segmentation Query 均为 `3 × 5 × 5 = 75` 个 token。
- Query 在 VLM 中保留，并通过带 RoPE 的 C2 Bridge 注入 Action DiT。
- 使用 grouped Action Query depth head 与 grouped segmentation head。
- Warmup 阶段从 GR00T-N1.6-3B 基座训练 20k steps。
- Full 阶段从 Warmup 的 `checkpoint-20000` 继续训练 120k steps。
- Full 阶段损失权重为 action `1.0`、depth `0.1`、segmentation `0.002`。
- VLM 使用 rank-128 LoRA；Full 阶段训练 projector、Action DiT、VLLN 与 C2 Bridge。

## 开源接口

RoboTwin2 示例目录只暴露以下用户流程：

1. `validate_dataset.py` 校验训练数据。
2. `train_query_warmup.sh` 启动 Query Warmup。
3. `train_c2_full.sh` 从 Warmup checkpoint 启动正式训练。
4. `eval_server.sh` 启动策略服务。
5. `eval_client.sh` 运行 RoboTwin2 客户端评测。

Shell 入口只允许用户覆盖数据、模型、输出路径，GPU、随机种子、batch size、学习率、训练步数、端口和日志配置。方法本身的固定结构参数不再作为消融开关暴露。

## 裁剪边界

删除以下内容及其专用实现、配置、测试和文档：

- 非 75-query 的 query-number ablation。
- `seg_only`、`action_only`、`no_task_heads`、`no_query_injection`。
- A3U、tri-plane、PETR、BiKRM、Spatial Forcing、L12 bridge/self-distillation、aligned geometry。
- 重复或指向历史 checkpoint 的部署脚本。
- 生成物、缓存和本地实验报告。

保留 GR00T 通用基础能力以及主方案训练、加载和推理所必需的代码。源码裁剪以静态引用和导入验证为准，不因名称相似而删除仍被主链路调用的公共模块。

## 路径与可复现性

- 删除开发机绝对路径。
- 所有仓库文件通过脚本目录推导 `REPO_ROOT`。
- 数据集与基座模型路径由明确的环境变量提供，并在启动时校验。
- Warmup 与 Full 使用稳定、可预测的输出目录和 checkpoint 衔接规则。

## 安全与验证

当前目录没有 Git 历史，因此删除前在仓库外保存被删除源码的恢复归档和文件清单。完成后执行：

- Python compile/import 检查。
- Shell `bash -n` 检查。
- 启动参数与两阶段 checkpoint 衔接测试。
- 主方案相关单元测试。
- 全量可运行测试，并区分环境/依赖问题与裁剪回归。

