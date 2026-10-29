# 发布复核记录

## 2026-10-29同步检查

用户已明确授权更新GitHub。此次以远端main为基线，按149项公开文件清单准备提交，保留原提交历史；学习材料、密钥、会话、模型缓存、私人评审与运行记录不上传。

提交前本地完整Python回归839项通过；隔离公开副本实际导入副本代码，Python回归704项、前端23项通过，45个内部链接无错误。公开副本的wheel与sdist构建成功并检查了私有路径和凭据；这些是离线机制与构建检查，不是新增真实模型质量验收。本次远端CI以新提交实际运行结果为准，不沿用旧CI结论。

## 此前准备记录

本轮只修改本地文件，没有自动提交、推送、发布或更改GitHub About/Topics。原有拆分修改和未跟踪成果继续保留；不要用本地初始Git基线的全部差异直接覆盖公开仓库。

后续联网联合修复新增app/phone_identity.py、app/evidence_registry.py、tests/test_unified_evidence.py及 [联网修复记录](NETWORK_REPAIR_20261024.md)，并修改既有搜索/正文、本地检索、会话/回答、刷新状态与页面三模式选择。原始审查、保存HTML回放、本地UI记录均留在公开清单之外，不因补回归而上传。

2026-10-25集中逻辑修复增加tests/test_logic_invariants.py及[逻辑修复记录](LOGIC_REPAIR_20261025.md)，收敛条件原子化、本轮task、资格/答案校验、刷新归并、查询进展和知识写入。原真实失败、审查与修后重放留在私有runs目录；不把它们改称真实验收通过。

## 本轮应审阅的内容

- 会话与条件：app/chat.py、chat_intent.py、chat_schemas.py、conversation_policy.py、chat_prompts.py。
- 要求与证据：app/selection_requirements.py、feature_requirements.py、source_retention.py、chat_agent.py、chat_answer.py。
- 公开资料：app/knowledge_samples/selection_index.json、coverage_additions_20261024.jsonl、public_source_directory.jsonl。
- 回归与工具：tests/test_selection_state_evidence.py、scripts/check_public_docs.py。
- 文档与包装：README.md、docs/CHAT.md、AGENT.md、ARCHITECTURE.md、RAG.md、EMBEDDING.md、CORPUS.md、REPAIR_20261024.md、PUBLISH_CHECKLIST.md、public-files.txt、evaluation/README.md、MANIFEST.in、pyproject.toml。
- 图片：docs/chat-demo.png为核对过的历史真实截图；docs/api-key-setup.png沿用已公开的空输入配置图，只在配置节展示。

完整公开候选路径见 [public-files.txt](public-files.txt)。它以已公开版本文件为基线，加本轮必要文件；已排除Python/pytest缓存，不自动包含本地其他未跟踪文件。此清单用于复核与安全链接检查，不是发布命令。尚未公开的私人测试、会话、原始请求、日志、人工评审、实验和成本报告不得因为“修链接”加入。

此前603/468/17及650/515/18是历史记录。当前完整Python683项、公开副本548项、前端18项分别通过，不相加；149个公开候选文件的内部链接通过检查。公开清单已移除旧runs/chat-sessions.sqlite3条目，原数据库保留。首次副本准备中止的结果不计通过；有效公开测试核对了实际导入路径。打包只使用该公开副本，使用既有依赖做源码外wheel验证，不代表全新依赖环境或真实模型验收。输出留在忽略目录供审阅，不要直接把含私人开发文件的工作目录打包上传。

修后核查追加修正后的最新计数为完整Python723项、公开副本588项、前端18项；683/548是上一阶段记录。149文件及45个内部链接保持不变，构建包重新从本阶段公开副本验证。追加范围为新搜索线索的有界抓取机会、历史价格时间/否定校验、规格比较值归属、回答操作与产品条件分流、分项实测缺口与跨轮状态文案；详见逻辑修复记录。独立真实核查原记录和人工pending状态不变，追加修复没有新增真实调用。

用户网页反馈继续修复后的最新计数为完整Python759项、公开副本624项、前端19项；上述723/588/18保留为上一阶段记录。本阶段增加逐金额机型归属、合法价格纠错、旧会话派生报价撤回、短预算与澄清前置、项目额度可见性与停止后不再请求、换批记忆、临时/持久系列范围及弱品牌偏好。原会话、索引、额度和失败记录保留，没有新增付费或外网调用，也未重启服务。公开包来自本阶段149文件清单，不带私人原会话；离线回归不代表用户网页链路已重新验收。

配置推断与总体问法修复后的最新计数为完整Python809项、公开副本674项、前端20项。最新产品边界允许基于已核实事实的有界初判，保留事实/推断/未知项及测量、价格和硬门槛校验。增加共用CPU/GPU等事实视图、overview、缺引用/结构错误的有界修正及可区分对象的背景引用。旧真实失败仍为旧结果，未追加真实模型/外网调用，用户授权的每日200万配置保持；本阶段公开清单仍149文件、45内部链接，私人会话不入包。

## GitHub页面建议

README已修正实际clone地址，增加CI徽章、聊天主流程图、八轮操作示例、来源与验证范围。首图为历史真实追问截图；它未展示本版完整候选更新，应在后续限定调用数的真实多轮验收后替换，不能用假模型夹具冒充真实演示。

About已有简介，先保留。待明确授权后可设置Topics：`phone-recommendation`、`conversational-ai`、`ai-agent`、`langchain`、`langgraph`、`rag`、`fastapi`、`python`、`sqlite`。本轮未向远端写入。

## 发布前复核

1. 逐项审阅本轮变更与已有成果，确认公开清单中不含密钥、个人会话和私有评审。
2. 使用本工程.venv运行完整Python与前端回归、公开链接检查；私有完整回归与公开副本回归分别计数。
3. 核对构建包，确保新增运行模块/索引/公开补录可用，文档图片和链接可解析；不要直接上传本地runs或全目录归档。
4. 本地结果通过后再由用户明确授权提交/推送。已公开CI成功不代表这些未推送文件已经在远端通过；推送后需观察新的CI。

[修复与验证边界](REPAIR_20261024.md) · [资料来源与缺口](CORPUS.md)
