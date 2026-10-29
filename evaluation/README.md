# 对话助手验证

本工程只验证多轮对话、资料检索与回答，不运行手机对比研究。拆分前旧研究实验的原报告作为历史保留，当前研究脚本和测试在独立phone-comparison-agent中。

私人原题运行、人工评审、检索实验与成本结果未公开，不作为公开文档链接目标。公开机制与本轮边界见 [Agent](../docs/AGENT.md)、[资料覆盖](../docs/CORPUS.md)、[修复验证](../docs/REPAIR_20261024.md)。测试数量与检索指标不代表推荐准确率。

pytest验证会话和资料行为；scripts.evaluate_chat运行六类对话验收；scripts.evaluate_intent检查需求变更。scripts.evaluate_user_chat仅使用用户原题，默认只打印计划，显式--live运行真实模型与资料工具。所有原输出保留，失败不换题或删除。

拆分后验收已排除原对比工作流的两个检查。总项数以每次新结果为准，不混用拆分前288项的历史数字。

封存评测使用 `scripts.evaluate_sealed_chat`，先准备 JSON 数组：每项有唯一 `id`、`kind: "conversation"` 和 `turns`（普通 ChatRequest 字段，例如 message/search_mode）。先执行 `.venv/Scripts/python.exe -m scripts.evaluate_sealed_chat seal --cases <题目.json> --output <新目录> --max-provider-requests 12 --origin external`。封存只读文件，不调用模型；题目、代码、运行配置摘要和知识快照绑定后才能运行。当前没有独立保管的盲测题，开发者看过或编写的题应使用 `--origin developer`。

只有另行明确场景和调用预算后才执行同一目录的 `run --output <封存目录> --live`。每个目录只允许运行一次，失败或中断也不能重跑；请求、响应、来源、用量与失败原样保留，人工评审保持 pending。声明 external 不证明独立盲测，自动检查也不是推荐准确率。
