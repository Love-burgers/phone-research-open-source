# 对话助手架构

首页与 `/api/chat` 是独立入口。用户输入 → 增量理解 → 条件合并和路由 → 模型决定工具 → 受边界约束的执行 → 回答校验 → SQLite 保存。

| 模块 | 职责 |
|---|---|
| web、chat_store | HTTP、持久化、revision 并发保护 |
| chat、chat_intent、conversation_policy | 会话编排、需求增量、候选范围与路由 |
| chat_agent、tool_loop、agent_tools | 单轮 Agent 状态、模型工具循环、工具参数和预算 |
| chat_answer、selection_presentation | 共享回答校验、资料兜底、展示 |
| selection_catalog、selection_requirements | 候选目录、用途和硬条件核对 |
| feature_requirements、source_retention | 有限非数值属性四态核对与有界来源淘汰 |
| phone_identity、evidence_registry | 统一完整型号/别名边界，会话/本地/网页的合并、局部刷新和当前问题缺口 |
| selection_prices、evidence_rules、source_rules、topics | 无网络的价格、证据、来源身份与主题规则 |
| knowledge、knowledge_ingest、knowledge_models | 检索、入库与独立数据模型 |
| phone_sources、public_search、tools、model_api | 公开资料与模型适配 |
| config、chat_prompts | 集中配置和提示词 |

知识模型不导入检索/入库，展示不导入网页抓取或候选目录，公开来源的证据识别不依赖会话业务。模块导入图（含延迟导入）由离线测试检查为无环。目录模型的 card 方法保留兼容入口，价格和展示计算已拆出。

SQLite 持久化不等于 LangGraph checkpoint。每次请求加载独立会话和目录副本，图执行后以 revision 保存；会话内发现的型号不污染全局目录。Agent和固定资料路径共用按属性更新的条件、单轮task、证据资格与AnswerComposer。公开事实自动入库在会话成功保存后进行；索引写入以进程间互斥读取最新磁盘版本，拒绝旧日期回滚同源事实。索引和会话是两个存储，不承诺跨文件事务。

wheel包含运行代码、静态页面、公开索引与本次公开补录JSONL；私人评审和运行记录不打包。默认102款/478片段BM25，可选本地BGE已接入，外部索引通过CHAT_KNOWLEDGE_INDEX_PATH配置且不自动同步随包更新。本项目没有跨工程运行依赖。

已有共享访问密钥、远程无密钥拒绝、每IP限流和SQLite每日请求/模型调用/token限额，按UTC+8切换。共享密钥不是多用户账户系统，token控制不是人民币账单结算。[已核验CI](https://github.com/Love-burgers/phone-research-open-source/actions/runs/39953196441)的Ubuntu、Windows、container均成功；Docker已验证构建和phone-chat --help，不等于容器完整聊天验收，也不覆盖未推送的修复。

本轮focus_topics与累计条件分开保存；相机按镜头角色、屏幕/触控频率和有线/无线充电分别核对。工作来源表每款最多八条，按条件覆盖与日期保留，历史回答拥有独立来源快照。历史真实失败不因离线修复改判为真实通过。
