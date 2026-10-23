# 对话助手架构

首页与 `/api/chat` 是独立入口。用户输入 → 增量理解 → 条件合并和路由 → 模型决定工具 → 受边界约束的执行 → 回答校验 → SQLite 保存。

| 模块 | 职责 |
|---|---|
| web、chat_store | HTTP、持久化、revision 并发保护 |
| chat、chat_intent、conversation_policy | 会话编排、需求增量、候选范围与路由 |
| chat_agent、tool_loop、agent_tools | 单轮 Agent 状态、模型工具循环、工具参数和预算 |
| chat_answer、selection_presentation | 共享回答校验、资料兜底、展示 |
| selection_catalog、selection_requirements | 候选目录、用途和硬条件核对 |
| selection_prices、evidence_rules、source_rules、topics | 无网络的价格、证据、来源身份与主题规则 |
| knowledge、knowledge_ingest、knowledge_models | 检索、入库与独立数据模型 |
| phone_sources、public_search、tools、model_api | 公开资料与模型适配 |
| config、chat_prompts | 集中配置和提示词 |

知识模型不导入检索/入库，展示不导入网页抓取或候选目录，公开来源的证据识别不依赖会话业务。模块导入图（含延迟导入）由离线测试检查为无环。目录模型的 card 方法保留兼容入口，价格和展示计算已拆出。

SQLite 持久化不等于 LangGraph checkpoint。每次请求加载独立会话和目录副本，图执行后以 revision 保存；会话内发现的型号不污染全局目录。Agent 和固定资料路径共享 AnswerComposer，无密钥时仍可摘录已有资料。

wheel 只包含运行代码、静态页面和公开索引；测试、评测脚本及历史评审留在开发仓库。默认随包 98 款 BM25，外部索引通过 CHAT_KNOWLEDGE_INDEX_PATH 配置。本项目没有跨工程运行依赖。

已加入共享访问密钥、远程无密钥拒绝、每IP限流和SQLite持久化的每日请求/模型调用/token限额；按UTC+8切换。共享密钥不是多用户账户系统；token控制不等于人民币账单硬停。源码已公开，GitHub Actions 执行跨平台及 Docker 检查，具体结果见运行记录；服务部署需另行配置。

当前实现还会把本轮语义focus_topics与完整条件分开保存，压缩最终请求但保留完整来源注册表。相机列表按标签/数值方向绑定镜头角色，屏幕和触控频率、有线/无线充电分别核对。真实12次请求的失败保存在 runs/resolution-20261023/live；后续修复只做离线回放，不把它改判为新版本真实通过。
