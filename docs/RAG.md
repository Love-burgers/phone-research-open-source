# 资料检索与 RAG

知识库保存带机型、网址、日期和证据类型的片段；Agent检索相关片段，决定是否继续查公开正文，回答使用同型号来源。检索排名和引用校验各有边界，不能单独证明结论正确。

## 默认与可选配置

| 配置 | 行为 |
|---|---|
| 默认外部索引路径为空 | 随包selection_index.json，102款/478片段，无向量BM25；不下载模型、不请求Embedding |
| CHAT_KNOWLEDGE_INDEX_PATH指向外部BM25索引 | 使用该文件内容，不自动同步随包补录 |
| 外部向量索引及EMBEDDING_PROVIDER=local | 正式接入的本地BGE路径，需事先准备缓存，支持vector/hybrid |
| 外部向量索引及远程Embedding配置 | 查询请求独立服务，资料发送和消费按部署配置控制 |

`CHAT_RETRIEVAL_METHOD` 可选bm25/vector/hybrid，未填沿用索引自身method。只配置向量提供方不会改变随包BM25路径。39款/301片段和98款/455片段是历史外部快照，当前默认以 [资料覆盖说明](CORPUS.md) 为准。

## 导入与检索

JSONL每行一条KnowledgeDocument，至少包含phone_name、title、source_url、text；真实资料还应填写checked_on、published_on（若能确认）、market、evidence_kind；版本价需reference_price与price_variant。先核对完整型号、市场、版本和测试条件。虚构Phone A/B仅为测试语料。

```powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli build --input my-reviewed-phones.jsonl --output runs/my-knowledge-new.json
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/my-knowledge-new.json --phone "完整机型" --need "重视续航"
```

LangChain将正文分成约600字符、80字符重叠的片段，保留稳定document_id/chunk_id，相同来源相同片段去重。输入上限2MiB、单条正文20000字符；索引最多1024片段、20MiB，超出拒绝。没有自动PDF解析或全站爬取。

BM25使用中文二字词组、英文单词和有限主题别名。完整型号先过滤，至多返回四个正相关片段；无相关词返回空。跨机型发现先按型号分散，再核对预算和条件，避免同款多片段挤掉其他型号。价格与评测覆盖仍不均匀，缺资料不是产品不符合。

## 向量与运行时增量

向量路径使用持久化向量、LangChain InMemoryVectorStore和余弦相似度，混合检索用RRF。默认最低相似度0.35，可用 `CHAT_VECTOR_MIN_SCORE` 调整，不是语义正确率。见 [Embedding配置](EMBEDDING.md)。

只有显式配置本地Embedding、加载可写外部索引时，取得的官方参数和明确版本价才可能通过准入后增量入库。保留导入历史，重复片段不重建；摘要和第三方评测不自动写共享索引。随包BM25默认不自动吸收联网结果；本次评测摘要是逐项核对后的有限补录。

每款最多八条工作证据，按本轮价格、硬条件、用途覆盖、正文类型和日期淘汰低价值材料。不同价格/相反属性保留冲突，同页正文更新复用编号；历史回答保留自身来源快照，不把被替换编号改绑其他URL，轨迹记录替换事件。

## 检查与访问边界

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_knowledge --output runs/knowledge-evaluation-new.json
.\.venv\Scripts\python.exe -m scripts.audit_chat_corpus --index app/knowledge_samples/selection_index.json --output runs/corpus-audit-new.json
```

离线回归验证召回、无资料返回、型号隔离及接线；语料审计检查结构与覆盖，不核验全部事实或实时可购性。历史实验与成本记录未公开，不链接私有材料，不将测试数量或检索指标当作推荐准确率。

当前已有共享访问密钥、远程无密钥拒绝及请求/消费限制。共享密钥不是多用户登录，知识库与SQLite也不是多用户权限隔离系统。私人文档、会话、索引和人工评审不随公开清单发布。
